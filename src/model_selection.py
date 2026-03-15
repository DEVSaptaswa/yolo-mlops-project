import os
import mlflow
import shutil
from pathlib import Path


def select_best_model():

    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("YOLO_Object_Detection")
    runs = client.search_runs(experiment.experiment_id)

    best_run = None
    best_map = 0

    for run in runs:
        metrics = run.data.metrics
        # ✅ Check multiple possible metric key names
        for key in ["metrics/mAP50-95B", "metrics/mAP50-95(B)", "mAP50-95"]:
            if key in metrics:
                score = metrics[key]
                if score > best_map:
                    best_map = score
                    best_run = run
                break

    if best_run is None:
        print("❌ No runs found with mAP metric")
        return

    print(f"✅ Best Run ID: {best_run.info.run_id}")
    print(f"✅ Best mAP50-95: {best_map}")

    # ✅ Model is stored under weights/best.pt
    download_path = mlflow.artifacts.download_artifacts(
        run_id=best_run.info.run_id,
        artifact_path="weights",
        tracking_uri="http://localhost:5000"
    )

    print(f"Downloaded to: {download_path}")

    # ✅ Find best.pt inside downloaded weights folder
    best_pt = Path(download_path) / "best.pt"
    last_pt = Path(download_path) / "last.pt"

    os.makedirs("models", exist_ok=True)

    if best_pt.exists():
        shutil.copy(best_pt, "models/best.pt")
        print("✅ Copied best.pt → models/best.pt")
    elif last_pt.exists():
        shutil.copy(last_pt, "models/best.pt")
        print("✅ Copied last.pt → models/best.pt")
    else:
        print(f"❌ No .pt file found in {download_path}")
        print("Files found:", list(Path(download_path).iterdir()))


if __name__ == "__main__":
    select_best_model()