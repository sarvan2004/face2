from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import json
import os
import pandas as pd
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime, timedelta
import pytz
import logging
import time

app = Flask(__name__)

# --- Load config and ROI ---
def load_config(config_file="recognition_config.json"):
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "face_recognition": {
                "min_confidence": 0.5,
                "max_distance": 0.6,
                "min_face_size": 50,
                "consecutive_frames": 3,
                "quality_threshold": 0.3,
                "high_confidence_threshold": 0.6,
                "history_length": 10,
                "consistency_check_frames": 5
            }
        }

def load_roi(filename="roi_config_first_row.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return np.array(data["roi"], np.int32)
    except FileNotFoundError:
        return None

config = load_config()
face_config = config["face_recognition"]
polygon_roi = load_roi()
polygon_roi = None  # Disable ROI check for all faces
model = YOLO("yolov11n-face.pt")
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
attendance_file = "attendance.csv"

# --- Attendance marking logic ---
def mark_attendance(name):
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.strftime("%H:%M:%S")
    today_date = now.strftime("%Y-%m-%d")
    required_columns = ["Name", "Date", "Time", "Type"]
    recreate_file = False
    if not os.path.exists(attendance_file):
        recreate_file = True
    else:
        try:
            df = pd.read_csv(attendance_file)
            if list(df.columns) != required_columns:
                recreate_file = True
        except Exception:
            recreate_file = True
    if recreate_file:
        pd.DataFrame({col: [] for col in required_columns}).to_csv(attendance_file, index=False)
    df = pd.read_csv(attendance_file)
    today_df = df[df["Date"] == today_date]
    person_entries = today_df[today_df["Name"] == name]
    if isinstance(person_entries, pd.DataFrame) and (person_entries.empty or (person_entries.iloc[-1]["Type"] == "out")):
        new_entry = pd.DataFrame({
            "Name": [name],
            "Date": [today_date],
            "Time": [current_time],
            "Type": ["in"]
        })
        new_entry.to_csv(attendance_file, mode="a", header=False, index=False)
        return "in"
    if isinstance(person_entries, pd.DataFrame) and not person_entries.empty and person_entries.iloc[-1]["Type"] == "in":
        last_in = person_entries.iloc[-1]
        ist = pytz.timezone('Asia/Kolkata')
        last_in_time = ist.localize(datetime.strptime(f"{last_in['Date']} {last_in['Time']}", "%Y-%m-%d %H:%M:%S"))
        now = datetime.now(ist)
        if (now - last_in_time) >= timedelta(hours=1):
            new_entry = pd.DataFrame({
                "Name": [name],
                "Date": [today_date],
                "Time": [current_time],
                "Type": ["out"]
            })
            new_entry.to_csv(attendance_file, mode="a", header=False, index=False)
            return "out"
        else:
            return "already_marked"
    return "already_marked"

def is_inside_polygon(x, y, polygon):
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

def get_face_quality_score(face_crop):
    if face_crop.size == 0:
        return 0
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        sharpness_score = min(laplacian_var / 100, 1.0)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        return (sharpness_score + brightness_score) / 2
    except Exception:
        return 0

def get_best_match_from_deepface_result(result):
    if isinstance(result, pd.DataFrame) and not result.empty:
        return result.iloc[0]
    return None

def recognize_face_with_confidence(face_crop):
    quality_score = get_face_quality_score(face_crop)
    print(f"[DEBUG] Face quality score: {quality_score}")
    if quality_score < face_config["quality_threshold"]:
        print("[DEBUG] Low quality face, skipping.")
        return None, 0, f"Low quality face ({quality_score:.2f})"
    verification = DeepFace.find(
        img_path=face_crop,
        db_path=KNOWN_FACES_DIR,
        model_name="ArcFace",
        enforce_detection=False
    )
    print(f"[DEBUG] DeepFace.find result: {verification}")
    if isinstance(verification, list) and len(verification) > 0:
        best_match = get_best_match_from_deepface_result(verification[0])
        print(f"[DEBUG] DeepFace best match: {best_match}")
        if best_match is not None:
            distance = best_match["distance"]
            person_name = os.path.basename(best_match["identity"]).split(".")[0]
            if distance <= face_config["max_distance"]:
                confidence = 1.0 - (distance / face_config["max_distance"])
                status = f"Distance: {distance:.3f}, Quality: {quality_score:.2f}"
                return person_name, confidence, status
            else:
                print(f"[DEBUG] Distance too high: {distance}")
                return None, 0, f"Distance too high: {distance:.3f}"
        else:
            print("[DEBUG] No match found in DeepFace result.")
            return None, 0, "No match found"
    else:
        print("[DEBUG] No match found by DeepFace.")
        return None, 0, "No match found"

@app.route('/recognize', methods=['POST'])
def recognize():
    print("[DEBUG] Received /recognize request")
    if 'file' not in request.files:
        print("[DEBUG] No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    print(f"[DEBUG] Image loaded, shape: {img.shape if img is not None else None}")
    results = model(img)
    print(f"[DEBUG] YOLO results: {len(results)} result(s)")
    recognized = []
    for result in results:
        print(f"[DEBUG] YOLO result: {len(result.boxes)} box(es)")
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            print(f"[DEBUG] Detected box: {x1},{y1},{x2},{y2} conf={confidence}")
            if confidence < face_config["min_confidence"]:
                print(f"[DEBUG] Skipping box due to low confidence: {confidence}")
                continue
            face_width = x2 - x1
            face_height = y2 - y1
            if face_width < face_config["min_face_size"] or face_height < face_config["min_face_size"]:
                print(f"[DEBUG] Skipping box due to small size: {face_width}x{face_height}")
                continue
            if not is_inside_polygon(x1, y1, polygon_roi):
                print(f"[DEBUG] Skipping box outside ROI")
                continue
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size == 0:
                print(f"[DEBUG] Skipping empty face crop")
                continue
            person_name, recog_conf, status = recognize_face_with_confidence(face_crop)
            print(f"[DEBUG] DeepFace result: name={person_name}, conf={recog_conf}, status={status}")
            if person_name:
                attendance_status = mark_attendance(person_name)
            else:
                attendance_status = None
            recognized.append({
                "name": person_name if person_name else "Unknown",
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "recognition_confidence": recog_conf,
                "status": status,
                "attendance": attendance_status
            })
    print(f"[DEBUG] Returning {len(recognized)} recognized face(s)")
    return jsonify({"recognized": recognized})

@app.route('/attendance')
def get_attendance():
    # Adjust the path if attendance.csv is elsewhere
    return send_file('attendance.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 