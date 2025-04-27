import cv2
import json
import numpy as np

# === CONFIG ===
VIDEO_PATH = "footage.mp4"     # Your video file
FRAME_NUMBER = 100                      # Frame to grab (adjust this as needed)
OUTPUT_JSON = "roi_config6.json"         # Output JSON file

# === Global variables ===
roi_points = []

def mouse_callback(event, x, y, flags, param):
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
        roi_points.pop()

def draw_roi(frame, points):
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)

def save_roi(points, filename):
    data = {"roi": points}
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"ROI saved to {filename}")

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    print("Left click to add points. Right click to remove. Press 's' to save, 'q' to quit.")

    cv2.namedWindow("Draw ROI")
    cv2.setMouseCallback("Draw ROI", mouse_callback)

    while True:
        display_frame = frame.copy()
        draw_roi(display_frame, roi_points)
        cv2.imshow("Draw ROI", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(roi_points) >= 3:
            save_roi(roi_points, OUTPUT_JSON)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()