#!/usr/bin/env python3
"""
Face Recognition GUI for Raspberry Pi
Optimized for Pi's limited resources - connects to cloud API
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import requests
import cv2
import os
from PIL import Image, ImageTk

# Cloud API URLs
API_URL = "http://13.201.230.71:5000/recognize"
ATTENDANCE_URL = "http://13.201.230.71:5000/attendance"

class PiFaceRecognitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition - Raspberry Pi")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Pi-optimized settings
        self.cap = None
        self.recognition_thread = None
        self.stop_event = threading.Event()
        self.is_recognition_running = False
        self.is_paused = False
        self.last_detection_boxes = []
        
        self.create_widgets()
        self.start_camera()

    def create_widgets(self):
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title = tk.Label(header, text="Face Recognition System", 
                        font=("Arial", 16, "bold"), 
                        bg="#2c3e50", fg="white")
        title.pack(pady=15)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Video frame (smaller for Pi)
        self.frame_panel = tk.Label(main_frame, bg="black", width=640, height=480)
        self.frame_panel.pack(pady=10)
        
        # Control buttons
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(pady=10)
        
        self.recognize_btn = tk.Button(button_frame, text="Recognize", 
                                      command=self.trigger_recognition,
                                      bg="#3498db", fg="white", 
                                      font=("Arial", 12, "bold"),
                                      width=15, height=2)
        self.recognize_btn.pack(side=tk.LEFT, padx=5)
        
        attendance_btn = tk.Button(button_frame, text="View Attendance", 
                                  command=self.view_attendance,
                                  bg="#27ae60", fg="white", 
                                  font=("Arial", 12, "bold"),
                                  width=15, height=2)
        attendance_btn.pack(side=tk.LEFT, padx=5)
        
        exit_btn = tk.Button(button_frame, text="Exit", 
                            command=self.quit_app,
                            bg="#e74c3c", fg="white", 
                            font=("Arial", 12, "bold"),
                            width=15, height=2)
        exit_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Ready", 
                                    font=("Arial", 10), 
                                    bg="#f0f0f0", fg="#2c3e50")
        self.status_label.pack(pady=5)

    def start_camera(self):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                # Try different camera indices for Pi
                for i in range(4):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No camera found")
                return
                
            # Set lower resolution for Pi performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.recognition_thread = threading.Thread(target=self.recognition_loop, daemon=True)
            self.recognition_thread.start()
            self.update_status("Camera started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")

    def recognition_loop(self):
        """Main recognition loop optimized for Pi"""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read() if self.cap else (False, None)
            if not ret or frame is None:
                time.sleep(0.1)
                continue
                
            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()
            
            # Save frame for API call
            temp_path = "/tmp/temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Call cloud API for detection
            detected_faces = []
            try:
                with open(temp_path, "rb") as img_file:
                    files = {"file": img_file}
                    response = requests.post(API_URL, files=files, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    detected_faces = data.get("recognized", [])
                    self.last_detection_boxes = detected_faces
                    
            except Exception as e:
                print(f"API Error: {e}")
                
            # Draw bounding boxes
            for person in self.last_detection_boxes:
                box = person.get("box", None)
                name = person.get("name", "Unknown")
                if box and len(box) == 4:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, name, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            if self.frame_panel is not None:
                self._imgtk = imgtk
                self.frame_panel.config(image=imgtk)
            
            # Handle recognition trigger
            if self.is_recognition_running and not self.is_paused:
                self.is_recognition_running = False
                marked = False
                for person in self.last_detection_boxes:
                    name = person.get("name", "Unknown")
                    attendance = person.get("attendance", "")
                    if attendance in ["in", "out", "already_marked"]:
                        self.is_paused = True
                        self.show_attendance_popup(frame, name, attendance)
                        marked = True
                        break
                        
            time.sleep(0.5)  # Slower loop for Pi

    def trigger_recognition(self):
        """Trigger face recognition"""
        self.is_recognition_running = True
        self.recognize_btn.config(text="Processing...", bg="#f39c12")
        self.update_status("Triggering recognition...")

    def show_attendance_popup(self, frame, name, attendance):
        """Show attendance popup"""
        if attendance == "already_marked":
            msg = f"Attendance already marked for {name}"
            color = (184, 134, 11)
        else:
            msg = f"Attendance marked for {name} ({attendance.upper()})"
            color = (34, 139, 34)
            
        # Create popup
        popup = tk.Toplevel(self.root)
        popup.title("Attendance Status")
        popup.geometry("400x300")
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.grab_set()
        
        # Display message
        label = tk.Label(popup, text=msg, font=("Arial", 14, "bold"), 
                        fg="white", bg="#2c3e50", wraplength=350)
        label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Auto-close after 7 seconds
        def close_popup():
            popup.destroy()
            self.reset_interface()
            self.is_paused = False
            
        popup.after(7000, close_popup)

    def reset_interface(self):
        """Reset interface after popup"""
        self.recognize_btn.config(text="Recognize", bg="#3498db")
        self.update_status("Ready")

    def view_attendance(self):
        """View attendance records"""
        try:
            response = requests.get(ATTENDANCE_URL, timeout=10)
            if response.status_code == 200:
                # Save to temporary file
                with open("/tmp/attendance.csv", "wb") as f:
                    f.write(response.content)
                
                # Show in popup
                self.show_attendance_window()
            else:
                messagebox.showerror("Error", f"Failed to fetch attendance: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch attendance: {e}")

    def show_attendance_window(self):
        """Show attendance in popup window"""
        try:
            import pandas as pd
            df = pd.read_csv("/tmp/attendance.csv")
            
            popup = tk.Toplevel(self.root)
            popup.title("Attendance Records")
            popup.geometry("600x400")
            
            # Create text widget with scrollbar
            text_frame = tk.Frame(popup)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, font=("Courier", 10))
            scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Insert attendance data
            text_widget.insert(tk.END, df.to_string(index=False))
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display attendance: {e}")

    def update_status(self, text):
        """Update status label"""
        if self.status_label:
            self.status_label.config(text=text)

    def quit_app(self):
        """Quit application"""
        self.stop_event.set()
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = PiFaceRecognitionGUI()
    app.run() 