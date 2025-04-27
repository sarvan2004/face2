import os
import numpy as np
from deepface import DeepFace

IMAGE_DIR = "data/images/"
EMBEDDING_DIR = "data/embeddings/"

os.makedirs(EMBEDDING_DIR, exist_ok=True)

embeddings = []
labels = []

for student_folder in sorted(os.listdir(IMAGE_DIR)):
    student_path = os.path.join(IMAGE_DIR, student_folder)
    if not os.path.isdir(student_path):
        continue
    
    print(f"Processing images for Student: {student_folder}")

    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)

        try:
            embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            embeddings.append(embedding)
            labels.append(student_folder)
        except Exception as e:
            print(f"⚠ Error processing {img_name}: {e}")

embeddings = np.array(embeddings)
labels = np.array(labels)

np.save(os.path.join(EMBEDDING_DIR, "embeddings.npy"), embeddings)
np.save(os.path.join(EMBEDDING_DIR, "labels.npy"), labels)

print("Face embeddings extracted and saved successfully!")
