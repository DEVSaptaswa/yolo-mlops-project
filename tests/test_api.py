import requests
import os

url = "http://localhost:8000/predict"

# Use an existing image from your dataset
image_path = "data/processed/images/val/BikesHelmets444.png"

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Response:", response.json())