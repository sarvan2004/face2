import cv2
import numpy as np

# Load the video
video_path = "data/raw_footage.mp4"  # Update with actual video path
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    cap.release()
    exit()

# List to store points of the polygon
polygon_points = []
drawing = False  # To track drawing state

def draw_polygon(event, x, y, flags, param):
    global polygon_points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # Left click to add point
        polygon_points.append((x, y))
    
    elif event == cv2.EVENT_RBUTTONDOWN and len(polygon_points) > 2:  # Right click to close
        drawing = True  # Stop selecting points

# Display frame and allow user to draw ROI
cv2.namedWindow("Select Polygonal ROI")
cv2.setMouseCallback("Select Polygonal ROI", draw_polygon)

while True:
    temp_frame = frame.copy()
    
    # Draw selected points & lines
    if len(polygon_points) > 1:
        cv2.polylines(temp_frame, [np.array(polygon_points)], isClosed=False, color=(0, 255, 0), thickness=2)
    
    cv2.imshow("Select Polygonal ROI", temp_frame)

    if drawing:  # If polygon is complete
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        exit()

cv2.destroyWindow("Select Polygonal ROI")


# Convert polygon to numpy array
polygon_points = np.array(polygon_points, np.int32)

print("Selected Polygon ROI Coordinates:", polygon_points)


import json

# Save polygon points to a JSON file
def save_roi(polygon_points, filename="roi_config.json"):
    with open(filename, "w") as f:
        json.dump({"roi": polygon_points.tolist()}, f)
    print(f"ROI saved to {filename}")

# Example usage (Call this after selecting ROI)
save_roi(polygon_points)

# Process the video within ROI
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mask = np.zeros_like(frame, dtype=np.uint8)  # Create blank mask
    cv2.fillPoly(mask, [polygon_points], (255, 255, 255))  # Fill ROI
    roi_frame = cv2.bitwise_and(frame, mask)  # Apply mask to frame

    cv2.imshow("Polygonal ROI Video", roi_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()