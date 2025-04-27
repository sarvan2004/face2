# 🎯 Face Detection and Dataset Creation from Video Footage

Extract and organize student faces from a classroom video by detecting faces **inside a defined ROI** using **YOLOv11-face** and **DeepFace (ArcFace)**.

---

## ⚙️ Installation

1. Install Python 3.8+.
2. Install required packages:

```bash
pip install -r requirements.txt

🚀 Usage
	1.Place your video inside the data/ directory.
	2.Define your ROI in roi_config.json like:
            {"roi": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}
        3.Run the script:
        python main.py
        4.Press q to quit the live video feed.
