import cv2
import os
import numpy as np
import json
import uuid
from deepface import DeepFace
from ultralytics import YOLO

# Load ROI from JSON
def load_roi(filename="roi_config_first_row.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    return np.array(data["roi"], np.int32)

# Check if point is inside polygon ROI
def is_inside_roi(x, y, roi):
    return cv2.pointPolygonTest(roi, (x, y), False) >= 0

# Save cropped face to person's folder
def save_face_to_folder(face_img, name):
    person_folder = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_folder, exist_ok=True)
    count = len(os.listdir(person_folder))
    cv2.imwrite(os.path.join(person_folder, f"{count+1}.jpg"), face_img)

# Paths and configs
VIDEO_PATH = "raw_footage_first_row.mp4"
KNOWN_FACES_DIR = "first_row_faces"
YOLO_MODEL = "yolov11n-face.pt"
ROI_PATH = "roi_config_first_row.json"

# Load ROI and YOLO
polygon_roi = load_roi(ROI_PATH)
model = YOLO(YOLO_MODEL)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Open video and calculate FPS-based frame skip
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
target_interval = 0.2  # seconds
frame_skip = int(fps * target_interval)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Draw ROI
    display_frame = frame.copy()
    cv2.polylines(display_frame, [polygon_roi], isClosed=True, color=(0, 255, 0), thickness=2)

    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if not is_inside_roi(cx, cy, polygon_roi):
                continue

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            try:
                search_results = DeepFace.find(
                    img_path=face_crop,
                    db_path=KNOWN_FACES_DIR,
                    model_name="ArcFace",
                    enforce_detection=False
                )

                if isinstance(search_results, list) and (len(search_results) == 0 or search_results[0].empty):
                    random_id = str(uuid.uuid4())[:8]
                    save_face_to_folder(face_crop, f"person_{random_id}")
                    label = f"New: {random_id}"
                    print(f"New person added: person_{random_id}")
                else:
                    label = "Matched"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            except Exception as e:
                print("Error in recognition:", str(e))

    # Show video with bounding boxes and ROI
    cv2.imshow("Face Detection", display_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()