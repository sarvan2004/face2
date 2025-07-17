#!/usr/bin/env python3
"""
Test script for face capture functionality
"""

import cv2
import requests
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# Cloud API URL
API_URL = "http://13.201.230.71:5000/recognize"

def test_face_capture():
    """Test face capture with cloud API"""
    print("Testing face capture functionality...")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return False
    
    print("Camera opened successfully")
    print("Press 'c' to capture face, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        # Display frame
        cv2.imshow("Test Capture", frame)
        
        # Send frame to API for detection
        temp_path = "/tmp/test_capture.jpg"
        cv2.imwrite(temp_path, frame)
        
        try:
            with open(temp_path, "rb") as img_file:
                files = {"file": img_file}
                response = requests.post(API_URL, files=files, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                detected_faces = data.get("recognized", [])
                
                if detected_faces:
                    print(f"✓ Face detected: {len(detected_faces)} face(s)")
                else:
                    print("No faces detected")
                    
        except Exception as e:
            print(f"API Error: {e}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Capture key pressed")
            # Test capture logic here
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    test_face_capture() 