from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

model = YOLO("models/best_model.pt")


@app.get("/")
def home():
    return {"message": "YOLO MLOps API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(image)

    predictions = results[0].boxes.data.tolist()

    return {"detections": predictions}