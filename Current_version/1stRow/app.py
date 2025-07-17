import cv2
import numpy as np
import json
import os
import pandas as pd
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime, timedelta
import logging
import time

def load_config(config_file="recognition_config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_file} not found, using default settings")
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
            },
            "display": {
                "show_confidence": True,
                "show_distance": True,
                "show_quality": False,
                "tentative_recognition": True
            },
            "logging": {
                "log_level": "INFO",
                "log_errors": True,
                "log_recognition": False
            }
        }

def load_roi(filename="roi_config_first_row.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return np.array(data["roi"], np.int32)
    except FileNotFoundError:
        print("ROI file not found")
        return None

# Load configuration
config = load_config()
face_config = config["face_recognition"]
display_config = config["display"]
logging_config = config["logging"]

# Setup logging
log_level = getattr(logging, logging_config["log_level"].upper())
logging.basicConfig(filename="face_recognition.log", level=log_level)

# Initialize components
polygon_roi = load_roi()
model = YOLO("yolov11n-face.pt")
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Attendance tracking
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

# Video capture setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
target_interval = 0.4
frame_skip = int(fps * target_interval)

# Tracking variables
cropped_faces_display = {}
face_embeddings_cache = set()
face_recognition_history = {}
frame_count = 0

def mark_attendance(name, face_crop):
    print(f"DEBUG: Attempting to mark attendance for {name}")
    global cropped_faces_display
    current_time = datetime.now().strftime("%H:%M:%S")
    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    required_columns = ["Name", "Date", "Time", "Type"]
    recreate_file = False
    if not os.path.exists(attendance_file):
        recreate_file = True
    else:
        try:
            df = pd.read_csv(attendance_file)
            if list(df.columns) != required_columns:
                print(f"DEBUG: CSV columns are {list(df.columns)}, expected {required_columns}. Recreating file.")
                recreate_file = True
        except Exception as e:
            print(f"DEBUG: Error reading CSV: {e}. Recreating file.")
            recreate_file = True
    if recreate_file:
        pd.DataFrame({col: [] for col in required_columns}).to_csv(attendance_file, index=False)
        print(f"DEBUG: Created new attendance file with columns {required_columns}")

    # Load today's attendance
    df = pd.read_csv(attendance_file)
    today_df = df[df["Date"] == today_date]
    person_entries = today_df[today_df["Name"] == name]

    print("DEBUG: person_entries:")
    print(person_entries)
    if isinstance(person_entries, pd.DataFrame) and not person_entries.empty:
        print("DEBUG: Last entry for today:")
        print(person_entries.iloc[-1])

    # If no entries today or last status is 'out', mark as 'in'
    if isinstance(person_entries, pd.DataFrame) and (person_entries.empty or (person_entries.iloc[-1]["Type"] == "out")):
        new_entry = pd.DataFrame({
            "Name": [name],
            "Date": [today_date],
            "Time": [current_time],
            "Type": ["in"]
        })
        new_entry.to_csv(attendance_file, mode="a", header=False, index=False)
        print(f"✔️ Attendance IN marked for {name} at {today_date} {current_time}")
        cropped_faces_display[name] = {
            "image": face_crop,
            "time": time.time()
        }
        return True

    # If last status is 'in', check if 1 hour has passed since last 'in'
    if isinstance(person_entries, pd.DataFrame) and not person_entries.empty and person_entries.iloc[-1]["Type"] == "in":
        last_in = person_entries.iloc[-1]
        last_in_time = datetime.strptime(f"{last_in['Date']} {last_in['Time']}", "%Y-%m-%d %H:%M:%S")
        print(f"DEBUG: last_in_time={last_in_time}, now={now}, diff={(now - last_in_time)}")
        if (now - last_in_time) >= timedelta(hours=1):
            new_entry = pd.DataFrame({
                "Name": [name],
                "Date": [today_date],
                "Time": [current_time],
                "Type": ["out"]
            })
            new_entry.to_csv(attendance_file, mode="a", header=False, index=False)
            print(f"✔️ Attendance OUT marked for {name} at {today_date} {current_time}")
            cropped_faces_display[name] = {
                "image": face_crop,
                "time": time.time()
            }
            return True

    # Otherwise, do nothing
    return False

def is_inside_polygon(x, y, polygon):
    """Check if point is inside polygon"""
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

def get_face_quality_score(face_crop):
    """Calculate face quality score based on sharpness and brightness"""
    if face_crop.size == 0:
        return 0
    
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Simple quality score (0-1)
        sharpness_score = min(laplacian_var / 100, 1.0)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        
        return (sharpness_score + brightness_score) / 2
    except Exception as e:
        logging.error(f"Error calculating face quality: {e}")
        return 0

def get_best_match_from_deepface_result(result):
    if isinstance(result, pd.DataFrame) and not result.empty:  # type: ignore
        return result.iloc[0]  # type: ignore
    return None

def recognize_face_with_confidence(face_crop, face_id_key):
    """Recognize face with comprehensive confidence checks"""
    try:
        # Check face quality
        quality_score = get_face_quality_score(face_crop)
        if quality_score < face_config["quality_threshold"]:
            return None, 0, f"Low quality face ({quality_score:.2f})"
        
        verification = DeepFace.find(
            img_path=face_crop,
            db_path=KNOWN_FACES_DIR,
            model_name="ArcFace",
            enforce_detection=False
        )

        if (
            isinstance(verification, list) and
            len(verification) > 0
        ):
            best_match = get_best_match_from_deepface_result(verification[0])
            if best_match is not None:
                distance = best_match["distance"]
                person_name = os.path.basename(best_match["identity"]).split(".")[0]
                if distance <= face_config["max_distance"]:
                    confidence = 1.0 - (distance / face_config["max_distance"])
                    status = f"Distance: {distance:.3f}, Quality: {quality_score:.2f}"
                    return person_name, confidence, status
                else:
                    return None, 0, f"Distance too high: {distance:.3f}"
            else:
                return None, 0, "No match found"
        else:
            return None, 0, "No match found"
            
    except Exception as e:
        if logging_config["log_errors"]:
            logging.error(f"Face Recognition Error: {str(e)}")
        return None, 0, f"Error: {str(e)}"

def update_face_history(face_id_key, person_name, confidence):
    """Update face recognition history for consistency checking"""
    if face_id_key not in face_recognition_history:
        face_recognition_history[face_id_key] = []
    
    face_recognition_history[face_id_key].append({
        'name': person_name,
        'confidence': confidence,
        'frame': frame_count
    })
    
    # Keep only recent history
    if len(face_recognition_history[face_id_key]) > face_config["history_length"]:
        face_recognition_history[face_id_key] = face_recognition_history[face_id_key][-face_config["history_length"]:]

def get_consistent_recognition(face_id_key):
    """Get the most consistent recognition for a face across multiple frames"""
    if face_id_key not in face_recognition_history:
        return None, 0
    
    history = face_recognition_history[face_id_key]
    if len(history) < 2:
        return None, 0
    
    # Count occurrences of each name
    name_counts = {}
    total_confidence = {}
    
    check_frames = face_config["consistency_check_frames"]
    for entry in history[-check_frames:]:
        name = entry['name']
        confidence = entry['confidence']
        
        if name not in name_counts:
            name_counts[name] = 0
            total_confidence[name] = 0
        
        name_counts[name] += 1
        total_confidence[name] += confidence
    
    # Find the most frequent name with highest average confidence
    if name_counts:
        most_frequent = max(name_counts.items(), key=lambda x: (x[1], total_confidence.get(x[0], 0)))
        name, count = most_frequent
        
        if count >= 2:  # At least 2 consistent detections
            avg_confidence = total_confidence[name] / count
            return name, avg_confidence
    
    return None, 0

def display_face_info(frame, x1, y1, person_name, confidence, status, is_confident=False):
    """Display face recognition information on frame"""
    if is_confident:
        color = (0, 255, 0)  # Green for confident recognition
        if display_config["show_confidence"]:
            display_text = f"{person_name} ({confidence:.2f})"
        else:
            display_text = person_name
    else:
        color = (255, 255, 0)  # Yellow for tentative recognition
        if display_config["tentative_recognition"]:
            display_text = f"{person_name}? ({confidence:.2f})"
        else:
            display_text = person_name
    
    cv2.putText(frame, display_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

print("Starting improved face recognition system...")
print(f"Configuration: Max Distance={face_config['max_distance']}, Min Confidence={face_config['min_confidence']}")
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from webcam")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Face detection
    results = model(frame)
    faces_detected = 0
    faces_recognized = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            
            # Check YOLO confidence
            if confidence < face_config["min_confidence"]:
                continue
            
            # Check face size
            face_width = x2 - x1
            face_height = y2 - y1
            if face_width < face_config["min_face_size"] or face_height < face_config["min_face_size"]:
                continue

            faces_detected += 1
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            try:
                face_id_key = f"{x1}_{y1}_{x2}_{y2}_{frame_count}"
                
                # Perform face recognition
                person_name, recognition_confidence, status = recognize_face_with_confidence(face_crop, face_id_key)
                
                if person_name:
                    # Update face history
                    update_face_history(face_id_key, person_name, recognition_confidence)
                    
                    # Check for consistent recognition
                    consistent_name, consistent_confidence = get_consistent_recognition(face_id_key)
                    
                    # Only close the camera if attendance_marked is True
                    try:
                        attendance_marked = mark_attendance(person_name, face_crop)
                        if attendance_marked:
                            print("Attendance marked, closing camera in 7 seconds.")
                            cv2.putText(frame, "Attendance marked", (x1, y1 - 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.imshow("Improved Face Recognition", frame)
                            cv2.waitKey(7000)  # Show message for 7 seconds
                            cap.release()
                            cv2.destroyAllWindows()
                            exit(0)
                        else:
                            # Show message and close camera if attendance is already marked
                            print("Attendance already marked, closing camera in 7 seconds.")
                            cv2.putText(frame, "Attendance already marked", (x1, y1 - 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            cv2.imshow("Improved Face Recognition", frame)
                            cv2.waitKey(7000)  # Show message for 7 seconds
                            cap.release()
                            cv2.destroyAllWindows()
                            exit(0)
                    except Exception as e:
                        print(f"ERROR in mark_attendance: {e}")
                        cv2.putText(frame, "Error", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # You can comment out the above line for production use
                    if consistent_name and consistent_confidence > face_config["high_confidence_threshold"]:
                        if face_id_key not in face_embeddings_cache:
                            face_embeddings_cache.add(face_id_key)
                        display_face_info(frame, x1, y1, consistent_name, consistent_confidence, status, True)
                    else:
                        display_face_info(frame, x1, y1, person_name, recognition_confidence, status, False)
                else:
                    cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
            except Exception as e:
                logging.error(f"Face Recognition Error: {str(e)}")
                cv2.putText(frame, "Error", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw bounding box
            color = (0, 255, 0) if person_name else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Display ROI if available
    if polygon_roi is not None:
        cv2.polylines(frame, [polygon_roi], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display recognized faces
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

    # Add status information
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Detected: {faces_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Recognized: {faces_recognized}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Max Distance: {face_config['max_distance']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Improved Face Recognition", frame)

    key = cv2.waitKey(int(1000 / fps))
    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Face recognition system stopped.") 