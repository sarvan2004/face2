import cv2
import numpy as np
import json
import os
import pandas as pd
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime
import logging
import time

# Setup logging
logging.basicConfig(filename="face_recognition.log", level=logging.INFO)

def load_roi(filename="roi_config_first_row.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return np.array(data["roi"], np.int32)
    except FileNotFoundError:
        print("ROI file not found, using full frame")
        return None

# Load YOLO model
model = YOLO("yolov11n-face.pt")

# Setup known faces directory
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Setup attendance tracking
attendance_file = "attendance.csv"
attendance_today = set()
today_date = datetime.now().strftime("%Y-%m-%d")

if os.path.exists(attendance_file):
    try:
        df = pd.read_csv(attendance_file)
        if "Date" in df.columns and "Name" in df.columns:
            today_df = df[df["Date"] == today_date]
            attendance_today = set(today_df["Name"])
        else:
            print("CSV missing 'Date' or 'Name' columns. Resetting attendance.")
    except Exception as e:
        print(f"Error reading attendance file: {e}")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam opened successfully!")
print("Press 'q' to quit")

# Load ROI (optional)
polygon_roi = load_roi()

# Variables for face tracking
cropped_faces_display = {}
face_embeddings_cache = set()
frame_count = 0

def mark_attendance(name, face_crop):
    global attendance_today, cropped_faces_display
    current_time = datetime.now().strftime("%H:%M:%S")

    if name not in attendance_today:
        attendance_today.add(name)
        new_entry = pd.DataFrame({"Name": [name], "Date": [today_date], "Time": [current_time]})

        write_header = not os.path.exists(attendance_file) or os.path.getsize(attendance_file) == 0
        new_entry.to_csv(attendance_file, mode="a", header=write_header, index=False)

        print(f"✔️ Attendance marked for {name}")

        cropped_faces_display[name] = {
            "image": face_crop,
            "time": time.time()
        }
        return True
    return False

def is_inside_polygon(x, y, polygon):
    if polygon is None:
        return True  # If no ROI, process all faces
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

print("Starting face recognition with EASY settings...")
print("Known faces directory:", KNOWN_FACES_DIR)
known_faces_list = os.listdir(KNOWN_FACES_DIR) if os.path.exists(KNOWN_FACES_DIR) else []
print("Available known faces:", known_faces_list)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from webcam")
        break

    frame_count += 1

    # Use the exact YOLO detection logic from the minimal test
    results = model(frame)
    faces_detected = 0
    faces_recognized = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            if conf > 0.5:
                faces_detected += 1
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                try:
                    # Recognition logic
                    verification = DeepFace.find(
                        img_path=face_crop,
                        db_path=KNOWN_FACES_DIR,
                        model_name="ArcFace",
                        enforce_detection=False
                    )
                    if isinstance(verification, list) and len(verification) > 0 and not verification[0].empty:
                        best_match = verification[0].iloc[0]
                        person_name = os.path.basename(best_match["identity"]).split(".")[0]
                        faces_recognized += 1
                        # Mark attendance (optional)
                        # mark_attendance(person_name, face_crop)  # Uncomment if you want to log attendance
                        cv2.putText(frame, f"{person_name}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Recognition error: {e}")
                    cv2.putText(frame, "Error", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Show detection and recognition count
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Faces Detected: {faces_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Faces Recognized: {faces_recognized}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Known faces: {len(known_faces_list)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Face Recognition EASY - Webcam", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        print("Quitting...")
        break
cap.release()
cv2.destroyAllWindows()
print("Webcam released. Goodbye!") 