from train import train_model


def run_experiments():

    configs = [
        {"lr":0.01, "batch":16, "imgsz":640},
        {"lr":0.001, "batch":32, "imgsz":640},
        {"lr":0.0005, "batch":16, "imgsz":512}
    ]

    for config in configs:

        train_model(
            lr=config["lr"],
            batch=config["batch"],
            imgsz=config["imgsz"],
            epochs=20
        )


if __name__ == "__main__":
    run_experiments()