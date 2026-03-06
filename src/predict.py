from ultralytics import YOLO
import cv2


def predict(image_path):

    model = YOLO("models/best_model.pt")

    results = model(image_path)

    results[0].show()

    return results


if __name__ == "__main__":

    predict("test.jpg")