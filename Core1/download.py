import os
import urllib.request
from ultralytics import YOLO

MODEL_PATH = "yolov8n-face.pt"
MODEL_URL = "https://github.com/derronqi/yolo-face-detection/releases/download/yolov8/yolov8n-face.pt"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLOv8 face detection model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# Load the model
model = YOLO(MODEL_PATH)

model.save("yolov8n-face.pt")