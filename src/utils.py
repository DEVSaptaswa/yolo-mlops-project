from train import train_model


def run_experiments():
    """Run a grid of training configurations and log each as a separate MLflow run."""

    configs = [
        {"lr": 0.01,   "batch": 16, "imgsz": 640},
        {"lr": 0.001,  "batch": 32, "imgsz": 640},
        {"lr": 0.0005, "batch": 16, "imgsz": 512},
    ]

    for i, config in enumerate(configs, 1):
        print(f"\n[utils] Starting experiment {i}/{len(configs)}: {config}")
        train_model(
            lr=config["lr"],
            batch=config["batch"],
            imgsz=config["imgsz"],
            epochs=20,
        )


if __name__ == "__main__":
    run_experiments()