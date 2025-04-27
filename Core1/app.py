import cv2
import numpy as np
import json
import os
import pandas as pd
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime


def load_roi(filename="roi_config.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return np.array(data["roi"], np.int32)
    except FileNotFoundError:
        print("ROI file not found")
        return None

polygon_roi = load_roi()
if polygon_roi is None:
    exit("No ROI found")

model = YOLO("yolov11l-face.pt")

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(attendance_file, index=False)

def is_inside_polygon(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

def mark_attendance(name):
    df = pd.read_csv(attendance_file)
    today_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    if not ((df["Name"] == name) & (df["Date"] == today_date)).any():
        new_entry = pd.DataFrame({"Name": [name], "Date": [today_date], "Time": [current_time]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"Attendance Marked for {name}")
        return True
    else:
        return False

cap = cv2.VideoCapture("data/raw_footage.mp4")

processed_faces = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2
            
            if is_inside_polygon(face_center_x, face_center_y, polygon_roi):
                face_crop = frame[y1:y2, x1:x2] 
                
                if face_crop.size == 0:
                    continue

                temp_face_path = "temp_face.jpg"
                cv2.imwrite(temp_face_path, face_crop)

                try:
                    verification = DeepFace.find(img_path=temp_face_path, db_path=KNOWN_FACES_DIR, enforce_detection=False)
                    if isinstance(verification, list) and len(verification) > 0 and not verification[0].empty:
                        best_match = verification[0].iloc[0]
                        person_name = os.path.basename(best_match["identity"]).split(".")[0]

                        if person_name not in processed_faces:
                            if mark_attendance(person_name):
                                processed_faces.add(person_name)

                        cv2.putText(frame, f"{person_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    print(f"⚠️ Face Recognition Error: {str(e)}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.polylines(frame, [polygon_roi], isClosed=True, color=(255, 0, 0), thickness=2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

