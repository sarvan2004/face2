# 🎓 Face Recognition Attendance System

A robust, bilingual, real-time face recognition-based attendance system using YOLOv8, DeepFace, TensorFlow, and OpenCV. This project supports both **real-time webcam-based** tracking and **video file-based** pipelines with Region of Interest (ROI), logging, frame skipping, and CSV-based attendance.

![preview](https://github.com/yashbisht077/DeepTrack/blob/main/Image.png?raw=true)

---

## 🚀 Project Structure

### 1. `Deep Track Prototype` (Webcam-based)

**Pipeline:**
- Real-time webcam feed.
- Face detection: `face_recognition`.
- Feature extraction: `DeepFace.represent()`.
- Identity prediction: Pretrained TensorFlow model.
- Attendance CSV generation with name and timestamp.

**Highlights:**
- Real-time accuracy.
- Simple pipeline to test live tracking.
- CSV output: `attendance.csv`

---

### 2. `Core 1` (Video + YOLO + DeepFace)

**Pipeline:**
- Input: Pre-recorded video file.
- ROI (Region of Interest): Defined polygon mask.
- Face detection: `YOLOv8` using `yolov11l-face.pt`.
- Recognition: `DeepFace.find()` using cosine similarity.
- Output: Annotated video + attendance CSV.

**Highlights:**
- Supports masked region-based filtering.
- Face matching from pre-encoded database folder.
- Output CSV: `attendance_log.csv`.

---

### 3. `Core 1 Improved` (Optimized)

**Pipeline Enhancements:**
- Faster detection using `yolov11n-face.pt`.
- Frame skipping for performance boost.
- Embedding caching to avoid repeated comparisons.
- Overlay of last matched face image and label (“Present”).
- Real-time bounding box display with names.
- Attendance duplication prevention.

**Extras:**
- Logging with `logging` module (`logs.txt`).
- Efficient attendance marking.
- Output files:
  - Annotated video: `output.mp4`
  - Attendance CSV: `final_attendance.csv`

---

## 🧠 Technologies Used

| Module         | Purpose                            |
|----------------|------------------------------------|
| `face_recognition` | Real-time face detection       |
| `DeepFace`     | Face embeddings and identity match |
| `YOLOv8`       | High-speed face detection          |
| `TensorFlow`   | Custom model for identity prediction (prototype) |
| `OpenCV`       | Video I/O, drawing, overlays       |
| `Numpy/Pandas` | Data and CSV handling              |
| `Logging`      | Robust debugging                   |

---

## 🗂️ File Structure

```plaintext
├── deep_track_prototype.py       # Webcam-based face tracking
├── core1.py                      # Video + YOLO + DeepFace
├── core1_improved.py             # Optimized video pipeline
├── known_faces/                  # Folder with known images
├── attendance/                   # Output CSV files
├── output/                       # Output annotated video
├── roi_mask.npy                  # ROI polygon mask file
├── logs.txt                      # Runtime logs
├── model.h5                      # Trained TensorFlow model
```

---

## 🛠️ Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yashbisht077/DeepTrack.git
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 weights:**
   - Place `yolov11l-face.pt` and `yolov11n-face.pt` in the root directory.

4. **Run pipelines:**
   - Prototype (webcam):
     ```bash
     python detect_faces.py
     ```
   - Core 1 (video):
     ```bash
     python app.py
     ```
   - Core 1 Improved:
     ```bash
     python app.py
     ```

---

## 📝 Sample Output

- ✅ `final_attendance.csv`:
```
Name,Time
Shankar Singh,2025-07-12 09:43:21
Anurag Singh,2025-07-12 09:44:12
...
```

---

## 📌 Future Improvements

- Web-based dashboard for CSV management.
- Real-time multi-camera support.
- Automatic ROI calibration using perspective transform.
- Facial spoofing detection.

---

## 📧 Contact

**Shankar Singh**  
📍 BTech CSE AIML @ GEHU Bhimtal  
📧 [shankarbisht1224@gmail.com]

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and contribute!