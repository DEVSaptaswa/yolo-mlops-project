import mlflow
import mlflow.pytorch
from ultralytics import YOLO

os.environ["MLFLOW_TRACKING_URI"] = "file:///D:/yolo-mlops-project/mlruns"

mlflow.set_tracking_uri("file:///D:/yolo-mlops-project/mlruns")
mlflow.set_experiment("YOLO_Object_Detection")

def train_model(lr, batch, imgsz, epochs):

    mlflow.start_run()

    # Log parameters
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch)
    mlflow.log_param("image_size", imgsz)
    mlflow.log_param("epochs", epochs)

    # Load YOLO model
    model = YOLO("yolov8n.pt")

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

    metrics = results.results_dict

    for key, value in metrics.items():
        clean_key = key.replace("(", "").replace(")", "")
        mlflow.log_metric(clean_key, float(value))

    # Save model
    model_path = "models/model.pt"
    model.save(model_path)

    mlflow.log_artifact(model_path)

    mlflow.end_run()


if __name__ == "__main__":

    train_model(
        lr=0.001,
        batch=16,
        imgsz=640,
        epochs=20
    )