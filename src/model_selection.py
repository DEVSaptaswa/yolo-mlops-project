import mlflow
import shutil


def select_best_model():

    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("Default")

    runs = client.search_runs(experiment.experiment_id)

    best_run = None
    best_map = 0

    for run in runs:

        metrics = run.data.metrics

        if "metrics/mAP50-95(B)" in metrics:

            score = metrics["metrics/mAP50-95(B)"]

            if score > best_map:
                best_map = score
                best_run = run

    artifact_uri = best_run.info.artifact_uri

    print("Best Model:", artifact_uri)

    shutil.copy(
        artifact_uri + "/model.pt",
        "models/best_model.pt"
    )


if __name__ == "__main__":
    select_best_model()