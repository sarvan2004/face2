import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import joblib

EMBEDDING_DIR = "data/embeddings/"
MODEL_DIR = "models/face_recognition/"
os.makedirs(MODEL_DIR, exist_ok=True)

embeddings = np.load(os.path.join(EMBEDDING_DIR, "embeddings.npy"))
labels = np.load(os.path.join(EMBEDDING_DIR, "labels.npy"))

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(embeddings.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(32, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(12, activation="relu"),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(num_classes, activation="softmax") 
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Training Neural Network face recognition model...")
history = model.fit(embeddings, encoded_labels, epochs=200, batch_size=10, validation_split=0.3)


model.save(os.path.join(MODEL_DIR, "face_recognition_model.h5"))
print("Face recognition model trained and saved successfully!")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Ensure you have 'history' from model.fit()
# Example: history = model.fit(...)

# Plot Loss vs Epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='dashed')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy vs Epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='dashed')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Simulate Predictions (Replace with actual model predictions)
y_true = np.random.randint(0, 5, size=50)  # Replace with actual validation labels
y_pred = np.random.randint(0, 5, size=50)  # Replace with actual model predictions

# Compute Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix as Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[f"Person {i}" for i in range(5)], 
            yticklabels=[f"Person {i}" for i in range(5)])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Simulated Confidence Scores (Replace with actual model confidence scores)
confidence_scores = np.random.uniform(0.5, 1.0, size=100)  # Replace with actual model confidences

# Plot Prediction Confidence Histogram
plt.figure(figsize=(8, 5))
plt.hist(confidence_scores, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel("Prediction Confidence")
plt.ylabel("Frequency")
plt.title("Prediction Confidence Histogram")
plt.grid(True)
plt.show()
