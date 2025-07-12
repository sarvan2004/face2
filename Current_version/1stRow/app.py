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


def load_roi(filename="roi_config_first_row.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return np.array(data["roi"], np.int32)
    except FileNotFoundError:
        print("ROI file not found")
        return None

polygon_roi = load_roi()
model = YOLO("yolov11n-face.pt")
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

attendance_file = "attendance.csv"
attendance_today = set()
today_date = datetime.now().strftime("%Y-%m-%d")

if os.path.exists(attendance_file):
    try:
        df = pd.read_csv(attendance_file)
        if "Date" in df.columns and "Name" in df.columns:
            today_df = df[df["Date"] == today_date]
            attendance_today = set(today_df["Name"].values)
        else:
            print("CSV missing 'Date' or 'Name' columns. Resetting attendance.")
    except Exception as e:
        print(f"Error reading attendance file: {e}")

cap = cv2.VideoCapture("raw_footage_first_row.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
target_interval = 0.4
frame_skip = int(fps * target_interval)

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
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    x, y, w, h = cv2.boundingRect(polygon_roi)
    roi_cropped_frame = frame[y:y+h, x:x+w]
    results = model(roi_cropped_frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 += x
            x2 += x
            y1 += y
            y2 += y

            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2

            if is_inside_polygon(face_center_x, face_center_y, polygon_roi):
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                try:
                    face_id_key = f"{x1}_{y1}_{x2}_{y2}_{frame_count}"
                    if face_id_key not in face_embeddings_cache:
                        verification = DeepFace.find(
                            img_path=face_crop,
                            db_path=KNOWN_FACES_DIR,
                            model_name="ArcFace",
                            enforce_detection=False
                        )

                        if isinstance(verification, list) and len(verification) > 0 and not verification[0].empty:
                            best_match = verification[0].iloc[0]
                            person_name = os.path.basename(best_match["identity"]).split(".")[0]

                            face_embeddings_cache.add(face_id_key)

                            mark_attendance(person_name, face_crop)
                            cv2.putText(frame, f"{person_name}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    logging.error(f"Face Recognition Error: {str(e)}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.polylines(frame, [polygon_roi], isClosed=True, color=(255, 0, 0), thickness=10)

    keys_to_remove = []
    for person_name, face_info in cropped_faces_display.items():
        cropped_face_image = face_info["image"]
        display_time = face_info["time"]

        if time.time() - display_time > 2:
            keys_to_remove.append(person_name)
            continue

        height, width = cropped_face_image.shape[:2]
        max_size = 100
        scale_factor = max_size / max(height, width)
        resized_face = cv2.resize(
            cropped_face_image, (int(width * scale_factor), int(height * scale_factor))
        )

        corner_x = frame.shape[1] - resized_face.shape[1] - 10
        corner_y = 10

        frame[corner_y:corner_y + resized_face.shape[0], corner_x:corner_x + resized_face.shape[1]] = resized_face
        cv2.putText(frame, "Present", (corner_x, corner_y + resized_face.shape[0] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    for key in keys_to_remove:
        del cropped_faces_display[key]

    cv2.imshow("Face Recognition Attendance", frame)

    key = cv2.waitKey(int(1000 / fps))
    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
