from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
from PIL import Image
import io

app = FastAPI()

model = YOLO("models/best.pt")

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API Requests"
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API latency"
)

@app.post("/predict")
async def predict(file: UploadFile):

    REQUEST_COUNT.inc()

    start = time.time()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model(image)

    latency = time.time() - start
    REQUEST_LATENCY.observe(latency)

    boxes = results[0].boxes.xyxy.tolist()
    conf = results[0].boxes.conf.tolist()

    return {
        "boxes": boxes,
        "confidence": conf,
        "latency": latency
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")