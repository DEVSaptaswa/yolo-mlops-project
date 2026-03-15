import os
import mlflow
import mlflow.pytorch
from ultralytics import YOLO

# ✅ Set BEFORE anything else — Ultralytics reads this env var automatically
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

def train_model(lr, batch, imgsz, epochs):

    # ✅ Set tracking URI inside the function too
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("YOLO_Object_Detection")

    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch)
        mlflow.log_param("image_size", imgsz)
        mlflow.log_param("epochs", epochs)

        # ✅ Disable Ultralytics' built-in mlflow callback to avoid conflict
        model = YOLO("yolov8n.pt")
        model.add_callback("on_pretrain_routine_end", lambda *args, **kwargs: None)

        # Train
        results = model.train(
            data="data/dataset.yaml",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr
        )

        # Log metrics
        metrics = results.results_dict
        for key, value in metrics.items():
            clean_key = key.replace("(", "").replace(")", "")
            try:
                mlflow.log_metric(clean_key, float(value))
            except Exception:
                pass

        # Save and log model
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pt"
        model.save(model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    train_model(
        lr=0.001,
        batch=16,
        imgsz=640,
        epochs=20
    )