import cv2
import numpy as np
import tensorflow as tf
import joblib
import face_recognition
from deepface import DeepFace

MODEL_DIR = "models/face_recognition/"
model = tf.keras.models.load_model(f"{MODEL_DIR}/face_recognition_model.h5")
label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

attendance_set = set()
ATTENDANCE_FILE = "data/csv/attendance.csv"


CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Real-time face recognition started...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (320, 240))

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)

    for (top, right, bottom, left) in face_locations:
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

        face_img = frame[top:bottom, left:right]
        face_img = cv2.resize(face_img, (160, 160))

        try:
            embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            embedding = np.array(embedding).reshape(1, -1)

            predictions = model.predict(embedding)
            best_match_idx = np.argmax(predictions)
            confidence = predictions[0][best_match_idx]

            student_id = "Unknown"
            if confidence >= 0.90:
                student_id = label_encoder.inverse_transform([best_match_idx])[0]

                if student_id not in attendance_set:
                    attendance_set.add(student_id)
                    with open(ATTENDANCE_FILE, "a") as f:
                        f.write(f"{student_id}, Present\n")

            box_color = (0, 255, 0) if confidence >= 0.90 else (0, 0, 255)

            label_text = f"{student_id} ({confidence:.2f})"
            cv2.putText(frame, label_text, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        except Exception as e:
            print(f"Error detecting face: {e}")

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("CCTV face recognition stopped.")