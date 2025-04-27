import cv2
import os
import json
import uuid
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO

# Load ROI
def load_roi(filename="roi_config1.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return np.array(data["roi"], np.int32)
    except:
        return None

# Check if point is inside polygon
def is_inside_polygon(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

# Validate cropped face is really a face
def is_real_face(face_img):
    try:
        result = DeepFace.verify(img1_path=face_img, img2_path=face_img,
                                 model_name="ArcFace", enforce_detection=True, silent=True)
        return result["verified"]
    except:
        return False

# Check if new face matches any existing one
def is_duplicate(face_img, known_faces_path):
    try:
        result = DeepFace.find(
            img_path=face_img,
            db_path=known_faces_path,
            model_name="ArcFace",
            enforce_detection=False,
            silent=True
        )
        return len(result) > 0 and not result[0].empty
    except:
        return False

# Save cropped face to random-named folder
def save_face_to_folder(face_crop, base_path, folder_name):
    person_folder = os.path.join(base_path, folder_name)
    os.makedirs(person_folder, exist_ok=True)
    count = len(os.listdir(person_folder))
    cv2.imwrite(os.path.join(person_folder, f"{count+1}.jpg"), face_crop)

# Main Process
video_path = "footage.mp4"
roi_polygon = load_roi("roi_config1.json")
known_faces_base = "face_dataset"
os.makedirs(known_faces_base, exist_ok=True)

cap = cv2.VideoCapture(video_path)
model = YOLO("yolov11n-face.pt")

frame_skip = 5
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if not is_inside_polygon(*face_center, roi_polygon):
                continue

            face_crop = frame[y1:y2, x1:x2]

            # Skip if too small
            if face_crop.size == 0 or face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                continue

            # Skip if not a real face
            if not is_real_face(face_crop):
                print("[-] Not a valid face — skipped")
                continue

            # Check for duplicate using DeepFace
            if not is_duplicate(face_crop, known_faces_base):
                folder_name = str(uuid.uuid4())[:8]
                save_face_to_folder(face_crop, known_faces_base, folder_name)
                print(f"[+] New person added: {folder_name}")
            else:
                print("[=] Duplicate face — skipped")

cap.release()
cv2.destroyAllWindows()