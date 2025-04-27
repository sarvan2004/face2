import cv2
import os
from mtcnn import MTCNN

def get_next_student_number(base_path):
    existing_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    numbers = sorted([int(folder) for folder in existing_folders if folder.isdigit()])

    return numbers[-1] + 1 if numbers else 1


def create_student_folder(base_path):

    student_number = get_next_student_number(base_path)
    student_folder = os.path.join(base_path, f"{student_number:03d}")
    os.makedirs(student_folder, exist_ok=True)

    return student_folder, student_number


def preprocess_image(image, detector):

    faces = detector.detect_faces(image)

    if len(faces) == 0:
        print("No face detected!")
        return None
    
    x, y, w, h = faces[0]['box']
    face = image[y:y+h, x:x+w]

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(10, 10))
    equalized = clahe.apply(gray)

    return equalized


def capture_faces(student_id, base_path="data/images", num_images=100):
    
    student_folder, student_number = create_student_folder(base_path)

    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    count = 0

    if not cap.isOpened():
        print("Error: Camera not detected!")
        return

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        processed_image = preprocess_image(frame, detector)

        if processed_image is None:
            print("Skipping frame: No face detected.")
            continue 
        
        img_name = os.path.join(student_folder, f"{student_id}_{count+1}.jpg")
        cv2.imwrite(img_name, processed_image)
        print(f"Saved: {img_name}")
        count += 1
        
        cv2.imshow("Capturing Faces", processed_image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Face capture completed for Student ID: {student_id}, Assigned Number: {student_number:03d}")

if __name__ == "__main__":
    student_id = input("Enter Student ID: ")
    capture_faces(student_id)
