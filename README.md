# YOLO MLOps Project

A full MLOps pipeline for object detection using YOLOv8, FastAPI, MLflow, Prometheus, and Grafana — fully containerized with Docker.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Training](#training)
- [API Reference](#api-reference)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a complete MLOps pipeline for helmet detection on bikes using the YOLOv8 object detection model. It includes:

- **Model Training** — YOLOv8 training with experiment tracking via MLflow
- **Model Serving** — FastAPI REST API for real-time inference
- **Experiment Tracking** — MLflow for logging parameters, metrics, and artifacts
- **Monitoring** — Prometheus metrics scraping + Grafana dashboards
- **Containerization** — Full Docker Compose setup for all services

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Docker Network                       │
│                                                         │
│  ┌─────────────┐     ┌─────────────┐                   │
│  │   FastAPI   │────▶│   MLflow    │                   │
│  │  :8000      │     │   :5000     │                   │
│  └──────┬──────┘     └─────────────┘                   │
│         │                                               │
│         │ metrics                                       │
│         ▼                                               │
│  ┌─────────────┐     ┌─────────────┐                   │
│  │ Prometheus  │────▶│   Grafana   │                   │
│  │  :9090      │     │   :3000     │                   │
│  └─────────────┘     └─────────────┘                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| Service    | Internal Port | External Port | Purpose                  |
|------------|--------------|---------------|--------------------------|
| FastAPI    | 8000         | 8000          | Prediction API           |
| MLflow     | 5000         | 5000          | Experiment tracking      |
| Prometheus | 9090         | 9091          | Metrics scraping         |
| Grafana    | 3000         | 3001          | Metrics visualization    |

---

## Project Structure

```
yolo-mlops-project/
├── api/
│   └── main.py                 # FastAPI app with prediction endpoint
├── src/
│   ├── train.py                # Model training with MLflow logging
│   ├── predict.py              # Local inference script
│   ├── model_selection.py      # Selects best model from MLflow
│   └── utils.py                # Runs multiple training experiments
├── monitoring/
│   ├── metrics.py              # Prometheus metric definitions
│   └── prometheus.yml          # Prometheus scrape config
├── data/
│   ├── raw/                    # Raw dataset
│   ├── processed/
│   │   ├── images/
│   │   │   ├── train/          # Training images
│   │   │   └── val/            # Validation images
│   │   └── labels/
│   │       ├── train/          # Training labels (YOLO format)
│   │       └── val/            # Validation labels (YOLO format)
│   └── dataset.yaml            # Dataset configuration
├── models/
│   └── best.pt                 # Best trained model
├── mlruns/                     # MLflow experiment data
├── tests/
│   └── test_api.py             # API test script
├── Dockerfile                  # API container definition
├── docker-compose.yml          # All services configuration
├── requirements.txt            # Python dependencies
└── yolov8n.pt                  # YOLOv8 nano base weights
```

---

## Prerequisites

- Python 3.10+
- Docker Desktop (with WSL2 on Windows)
- Git

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yolo-mlops-project.git
cd yolo-mlops-project
```

### 2. Create Virtual Environment

```powershell
python -m venv .env

# Windows
.env\Scripts\activate

# Mac/Linux
source .env/bin/activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create Required Folders

```powershell
mkdir models, mlruns
```

### 5. Place Base Model

Download `yolov8n.pt` from [Ultralytics](https://github.com/ultralytics/assets/releases) and place it in the project root. Then copy it as a placeholder:

```powershell
copy yolov8n.pt models\best.pt
```

---

## Configuration

### `data/dataset.yaml`

```yaml
path: data
train: processed/train/images
val: processed/val/images

nc: 2                          # number of classes
names: ["WithHelmet", "WithoutHelmet"]
```

### `monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "fastapi"
    static_configs:
      - targets: ["api:8000"]
```

---

## Running the Project

### Step 1 — Start All Services

```powershell
docker-compose up --build
```

Wait until all containers are running:

```
✔ Container mlflow-server       Started   → http://localhost:5000
✔ Container yolo-fastapi        Started   → http://localhost:8000
✔ Container prometheus-server   Started   → http://localhost:9091
✔ Container grafana-dashboard   Started   → http://localhost:3001
```

### Step 2 — Fix Corrupt Labels (One Time Only)

```powershell
python -c "
import os
def fix_labels(label_dir):
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(label_dir, fname)
        lines = open(path).readlines()
        fixed = []
        for line in lines:
            parts = line.strip().split()
            cls = parts[0]
            coords = [min(max(float(x), 0.0), 1.0) for x in parts[1:]]
            fixed.append(cls + ' ' + ' '.join(f'{c:.6f}' for c in coords) + '\n')
        open(path, 'w').writelines(fixed)
fix_labels('data/processed/labels/train')
fix_labels('data/processed/labels/val')
print('Labels fixed')
"
```

### Step 3 — Shut Down

```powershell
docker-compose down
```

---

## Training

### Single Training Run

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
python src/train.py
```

Default parameters:
| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Batch size | 16 |
| Image size | 640 |
| Epochs | 20 |

### Quick Test Run (1 epoch)

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
python -c "
import os, mlflow
from ultralytics import YOLO
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('YOLO_Test')
model = YOLO('yolov8n.pt')
model.add_callback('on_pretrain_routine_end', lambda *args, **kwargs: None)
with mlflow.start_run():
    results = model.train(data='data/dataset.yaml', epochs=1, imgsz=640, batch=16)
    print('Done - view at http://localhost:5000')
"
```

### Run Multiple Experiments

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
python src/utils.py
```

Runs 3 configurations automatically:

| Config | Learning Rate | Batch | Image Size |
|--------|--------------|-------|------------|
| 1      | 0.01         | 16    | 640        |
| 2      | 0.001        | 32    | 640        |
| 3      | 0.0005       | 16    | 512        |

### Select Best Model

After training, automatically picks the run with the highest `mAP50-95` score:

```powershell
python src/model_selection.py
```

Output:
```
✅ Best Run ID: 92c8c0a17fc9432b82a1cb3b03a2e86c
✅ Best mAP50-95: 0.743
✅ Copied best.pt → models/best.pt
```

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /`

Health check endpoint.

**Response:**
```json
{
  "status": "running",
  "model": "models/best.pt"
}
```

### `POST /predict`

Run object detection on an uploaded image.

**Request:** `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| file  | File | Image file (jpg, png) |

**Response:**
```json
{
  "boxes": [[x1, y1, x2, y2], ...],
  "confidence": [0.87, 0.91],
  "latency": 0.043,
  "model_used": "models/best.pt"
}
```

**Example:**
```powershell
python tests/test_api.py
```

Or via Swagger UI at **http://localhost:8000/docs**

### `GET /metrics`

Prometheus metrics endpoint — scraped automatically every 5 seconds.

---

## Monitoring

### Prometheus

Access raw metrics at **http://localhost:9091**

Available metrics:
| Metric | Type | Description |
|--------|------|-------------|
| `api_requests_total` | Counter | Total number of API requests |
| `api_request_latency_seconds` | Histogram | Request latency in seconds |
| `cpu_usage_percent` | Gauge | CPU usage percentage |

### Grafana Dashboard Setup

1. Open **http://localhost:3001** → login: `admin` / `admin`
2. Go to **Connections → Data Sources → Add → Prometheus**
3. Set URL to `http://prometheus-server:9090` → **Save & Test**
4. Go to **Dashboards → New → Add visualization**
5. Add panels using these queries:

| Panel Title | Query | Visualization |
|-------------|-------|---------------|
| Total Requests | `api_requests_total` | Stat |
| Requests Per Minute | `rate(api_requests_total[1m])` | Time series |
| Average Latency | `rate(api_request_latency_seconds_sum[1m]) / rate(api_request_latency_seconds_count[1m])` | Time series |
| CPU Usage | `cpu_usage_percent` | Gauge |

6. Save dashboard as `YOLO API Monitoring`

### MLflow Experiment Tracking

Access at **http://localhost:5000**

Logged per run:
- **Parameters:** learning rate, batch size, image size, epochs
- **Metrics:** mAP50, mAP50-95, precision, recall, box loss, cls loss
- **Artifacts:** model weights, training plots, confusion matrix

---

## Troubleshooting

### Port Already in Use

```powershell
# Find what's using the port
netstat -ano | findstr :3000

# Kill the process
taskkill /PID <PID> /F
```

### Container Won't Start

```powershell
# Check logs
docker logs <container-name>

# Rebuild from scratch
docker-compose down
docker rmi yolo-mlops-project-api -f
docker builder prune -a -f
docker-compose build --no-cache
docker-compose up -d
```

### MLflow URI Error

Always set the environment variable before training:
```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
```

### Corrupt Labels Warning

Run the label fix script (Step 2 above). These are non-normalized coordinates that YOLO ignores automatically — training still works.

### Model Not Found (`best.pt`)

```powershell
# Use base model as placeholder until training is complete
copy yolov8n.pt models\best.pt
docker restart yolo-fastapi
```

---

## Quick Reference

```
docker-compose up --build        → start all services
python src/train.py              → train model (20 epochs)
python src/utils.py              → run all experiments
python src/model_selection.py    → select best model
python tests/test_api.py         → test the API

http://localhost:8000/docs       → Swagger UI
http://localhost:5000            → MLflow experiments
http://localhost:9091            → Prometheus metrics
http://localhost:3001            → Grafana dashboard
```
