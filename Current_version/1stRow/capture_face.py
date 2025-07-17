import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import sys

def create_known_faces_dir():
    """Create known_faces directory if it doesn't exist"""
    known_faces_dir = "known_faces"
    os.makedirs(known_faces_dir, exist_ok=True)
    return known_faces_dir

def capture_face():
    """Capture face from webcam and save to known_faces folder"""
    
    # Load YOLO model for face detection
    model = YOLO("yolov11n-face.pt")
    
    # Create known_faces directory
    known_faces_dir = create_known_faces_dir()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: Could not open webcam")
        return
    
    print(" Face Capture Tool")
    print("=" * 40)
    print("Instructions:")
    print("1. Position your face in the center of the frame")
    print("2. Make sure your face is clearly visible and well-lit")
    print("3. Press 'c' to capture your face")
    print("4. Press 'q' to quit without capturing")
    print("=" * 40)
    
    face_captured = False
    captured_face = None
    person_name = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Error reading from webcam")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Detect faces
        results = model(frame)
        
        # Draw face detection boxes and instructions
        face_detected = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                
                if confidence > 0.5:  # Only show high-confidence detections
                    face_detected = True
                    # Draw green box around detected face
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add "Face Detected" text
                    cv2.putText(display_frame, "Face Detected", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add instructions to the frame
        cv2.putText(display_frame, "Press 'c' to capture face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if face_detected:
            cv2.putText(display_frame, "Face detected - Ready to capture!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No face detected", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow("Face Capture Tool", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(" Cancelled face capture")
            break
        elif key == ord('c'):
            if face_detected:
                # Capture the face
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0].item()
                        
                        if confidence > 0.5:
                            # Extract face region with some padding
                            padding = 20
                            y1_padded = max(0, y1 - padding)
                            y2_padded = min(frame.shape[0], y2 + padding)
                            x1_padded = max(0, x1 - padding)
                            x2_padded = min(frame.shape[1], x2 + padding)
                            
                            captured_face = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                            face_captured = True
                            break
                    if face_captured:
                        break
                
                if face_captured:
                    print(" Face captured successfully!")
                    break
            else:
                print(" No face detected. Please position your face in the frame.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if face_captured and captured_face is not None:
        return captured_face
    else:
        return None

def save_face(face_image, person_name):
    """Save the captured face to the known_faces folder"""
    
    if face_image is None:
        print(" No face image to save")
        return False
    
    # Clean the person name (remove spaces, special characters)
    clean_name = person_name.replace(" ", "_").replace("-", "_")
    clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")
    
    # Create filename
    filename = f"{clean_name}.png"
    filepath = os.path.join("known_faces", filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        print(f"  Warning: {filename} already exists")
        response = input("Do you want to overwrite it? (y/n): ").lower()
        if response != 'y':
            print(" Face not saved")
            return False
    
    # Save the face image
    try:
        cv2.imwrite(filepath, face_image)
        print(f" Face saved as: {filename}")
        print(f" Location: {os.path.abspath(filepath)}")
        return True
    except Exception as e:
        print(f"Error saving face: {e}")
        return False

def main():
    """Main function to run the face capture tool"""
    import sys
    # Check if name is provided as a command-line argument
    if len(sys.argv) > 1:
        person_name = sys.argv[1].strip()
        if not person_name:
            print(" Name cannot be empty")
            return
    else:
        person_name = input("Enter the person's name: ").strip()
        if not person_name:
            print(" Name cannot be empty")
            return
    print(f"\n Capturing face for: {person_name}")
    print("Position your face in the camera and press 'c' to capture")
    # Capture face
    face_image = capture_face()
    if face_image is not None:
        # Save the face
        if save_face(face_image, person_name):
            print("\n Face added successfully!")
            print("You can now run the face recognition system to test it.")
            # Show preview of saved face
            print("\n Preview of saved face:")
            cv2.imshow("Captured Face", face_image)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()
        else:
            print(" Failed to save face")
    else:
        print(" No face was captured")

if __name__ == "__main__":
    main() 