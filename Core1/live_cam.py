import cv2
import torch
import numpy as np
import json
from ultralytics import YOLO

# Load ROI from JSON file
def load_roi(filename="roi_config.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return np.array(data["roi"], np.int32)
    except FileNotFoundError:
        print("ROI file not found! Select ROI first.")
        return None

# Load YOLOv8 model for face detection
model = YOLO("yolov11s-face.pt")  # Use the smallest YOLOv8 model trained for faces

# Open recorded video file
video_path = "data/raw_footage.mp4"  # <-- Change this to your recorded footage
cap = cv2.VideoCapture(video_path)

polygon_points = load_roi()
if polygon_points is None:
    print("No saved ROI found. Please select ROI first.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Create a mask for the ROI
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
    roi_frame = cv2.bitwise_and(frame, mask)

    # Perform YOLO face detection
    results = model(roi_frame)

    # Draw bounding boxes around detected faces
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int
            confidence = box.conf[0].item()

            if confidence > 0.5:  # Confidence threshold
                cv2.rectangle(frame, (x1+8, y1+8), (x2+8, y2+8), (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Face Detection with ROI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()