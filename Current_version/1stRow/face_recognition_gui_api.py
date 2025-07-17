#!/usr/bin/env python3
"""
Face Recognition GUI Application (API Version)
A cross-platform GUI for the face recognition attendance system using a cloud API
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import time
import requests
import cv2
import os
import pandas as pd
from datetime import datetime
from PIL import Image, ImageTk

# Add these at the top of the file (after imports)
CLOUD_USER = "ubuntu"  # Cloud server username
CLOUD_IP = "15.206.60.212"  # Elastic IP
CLOUD_KNOWN_FACES_PATH = "/home/ubuntu/DeepTrack/Current_version/1stRow/known_faces/"

API_URL = "http://15.206.60.212:5000/recognize"  # Cloud API URL
ATTENDANCE_URL = "http://15.206.60.212:5000/attendance"  # Cloud attendance CSV endpoint

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)
    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#333", foreground="white",
                         relief=tk.SOLID, borderwidth=1,
                         font=("Arial", 9, "normal"))
        label.pack(ipadx=6, ipady=2)
    def hide_tip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

class FaceRecognitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition System (API)")
        self.root.geometry("900x650")
        self.root.configure(bg="#f4f7fa")
        self.root.resizable(True, False)
        try:
            self.root.state('zoomed')  # Windows full screen
        except Exception:
            try:
                self.root.attributes('-zoomed', True)  # Linux full screen
            except Exception:
                pass
        self.dark_mode = False
        self.setup_styles()
        self.is_recognition_running = False
        # Initialize thread/camera-related attributes BEFORE widgets
        self.cap = None
        self.recognition_thread = None
        self.stop_event = threading.Event()
        self.frame_panel = None
        self.last_frame = None
        self.result_label = None
        self.is_paused = False  # Add pause flag
        self.pause_timer = None
        self.last_recognized_faces = []  # Store last recognized faces for display
        self.last_detection_boxes = []   # Store last detected boxes for display
        self.create_widgets()
        self.update_status()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        if self.dark_mode:
            style.configure('Header.TFrame', background="#23272e")
            style.configure('Header.TLabel', background="#23272e", foreground="white", font=("Arial", 18, "bold"))
            style.configure('SubHeader.TLabel', background="#23272e", foreground="#b0c4de", font=("Arial", 11, "italic"))
            style.configure('Main.TFrame', background="#181a1b")
            style.configure('Card.TFrame', background="#23272e", relief="raised", borderwidth=1)
            style.configure('TButton', font=("Arial", 11, "bold"), padding=8, background="#23272e", foreground="white")
            style.configure('Start.TButton', background="#43d17a", foreground="white")
            style.configure('Stop.TButton', background="#e74c3c", foreground="white")
            style.configure('Action.TButton', background="#3498db", foreground="white")
            style.configure('Exit.TButton', background="#ff9800", foreground="white")
            style.configure('Status.TLabel', background="#23272e", foreground="#b0c4de", font=("Arial", 10))
            style.map('Start.TButton', background=[('active', '#36b86c'), ('!active', '#43d17a')], foreground=[('active', 'white'), ('!active', 'white')])
            style.map('Stop.TButton', background=[('active', '#c0392b'), ('!active', '#e74c3c')], foreground=[('active', 'white'), ('!active', 'white')])
            style.map('Action.TButton', background=[('active', '#217dbb'), ('!active', '#3498db')], foreground=[('active', 'white'), ('!active', 'white')])
            style.map('Exit.TButton', background=[('active', '#e67e22'), ('!active', '#ff9800')], foreground=[('active', 'white'), ('!active', 'white')])
        else:
            style.configure('Header.TFrame', background="#1a2233")
            style.configure('Header.TLabel', background="#1a2233", foreground="white", font=("Arial", 18, "bold"))
            style.configure('SubHeader.TLabel', background="#1a2233", foreground="#b0c4de", font=("Arial", 11, "italic"))
            style.configure('Main.TFrame', background="#f4f7fa")
            style.configure('Card.TFrame', background="white", relief="raised", borderwidth=1)
            style.configure('TButton', font=("Arial", 11, "bold"), padding=8)
            style.configure('Start.TButton', background="#4CAF50", foreground="white")
            style.configure('Stop.TButton', background="#f44336", foreground="white")
            style.configure('Action.TButton', background="#2196F3", foreground="white")
            style.configure('Exit.TButton', background="#ff9800", foreground="white")
            style.configure('Status.TLabel', background="#e9ecef", foreground="#333", font=("Arial", 10))
            style.map('Start.TButton', background=[('active', '#388e3c'), ('!active', '#4CAF50')], foreground=[('active', 'white'), ('!active', 'white')])
            style.map('Stop.TButton', background=[('active', '#b71c1c'), ('!active', '#f44336')], foreground=[('active', 'white'), ('!active', 'white')])
            style.map('Action.TButton', background=[('active', '#1565c0'), ('!active', '#2196F3')], foreground=[('active', 'white'), ('!active', 'white')])
            style.map('Exit.TButton', background=[('active', '#e65100'), ('!active', '#ff9800')], foreground=[('active', 'white'), ('!active', 'white')])

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.setup_styles()
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_widgets()
        self.update_status()

    def create_widgets(self):
        header = ttk.Frame(self.root, style='Header.TFrame', height=70)
        header.pack(fill=tk.X, side=tk.TOP)
        header.grid_propagate(False)
        logo_canvas = tk.Canvas(header, width=200, height=60, bg=('#23272e' if self.dark_mode else '#1a2233'), highlightthickness=0)
        logo_canvas.grid(row=0, column=0, rowspan=2, padx=(20, 10), pady=5)
        logo_canvas.create_text(100, 25, text="SEMICORE LABS", fill="white", font=("Arial", 16, "bold"))
        center_x, center_y = 100, 25
        for i in range(8):
            angle = i * 45
            radius = 8
            x = center_x + radius * (angle % 90 == 0) * (1 if angle < 180 else -1)
            y = center_y + radius * (angle % 90 != 0) * (1 if 90 < angle < 270 else -1)
            logo_canvas.create_oval(x-2, y-2, x+2, y+2, fill="#2196F3", outline="")
        logo_canvas.create_text(100, 45, text="Engineering Your Ideas", fill="#b0c4de", font=("Arial", 9, "normal"))
        title = ttk.Label(header, text="Face Recognition Attendance System", style='Header.TLabel')
        title.grid(row=0, column=1, sticky="w", pady=(10, 0))
        subtitle = ttk.Label(header, text="Modern, Fast, Accurate", style='SubHeader.TLabel')
        subtitle.grid(row=1, column=1, sticky="w", pady=(0, 10))
        header.columnconfigure(1, weight=1)
        dark_btn = ttk.Button(header, text=("🌙 Dark Mode" if not self.dark_mode else "☀️ Light Mode"), width=12, command=self.toggle_dark_mode)
        dark_btn.grid(row=0, column=2, rowspan=2, padx=20, pady=10, sticky="e")
        ToolTip(dark_btn, "Toggle dark/light mode.")
        main_frame = ttk.Frame(self.root, style='Main.TFrame', padding="30 10 30 10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        section_label = ttk.Label(main_frame, text="Main Actions", font=("Arial", 13, "bold"), background=('#181a1b' if self.dark_mode else '#f4f7fa'), foreground=('#b0c4de' if self.dark_mode else '#1a2233'))
        section_label.pack(anchor="w", pady=(0, 5))
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 15))
        card = ttk.Frame(main_frame, style='Card.TFrame', padding="30 25 30 25")
        card.pack(pady=10, ipadx=10, ipady=10)
        self.start_btn = ttk.Button(card, text="Recognize", style='Start.TButton', command=self.trigger_recognition)
        self.start_btn.grid(row=0, column=0, sticky="ew", pady=8)
        ToolTip(self.start_btn, "Trigger face recognition and attendance marking.")
        capture_btn = ttk.Button(card, text="📸 Capture New Face", style='Action.TButton', command=self.capture_face)
        capture_btn.grid(row=1, column=0, sticky="ew", pady=8)
        ToolTip(capture_btn, "Add a new person to the known faces database.")
        attendance_btn = ttk.Button(card, text="📋 View Attendance", style='Action.TButton', command=self.view_attendance)
        attendance_btn.grid(row=2, column=0, sticky="ew", pady=8)
        ToolTip(attendance_btn, "View all attendance records.")
        settings_btn = ttk.Button(card, text="⚙️ Settings", style='Action.TButton', command=self.show_settings)
        settings_btn.grid(row=3, column=0, sticky="ew", pady=8)
        ToolTip(settings_btn, "View system information and file paths.")
        exit_btn = ttk.Button(card, text="❌ Exit", style='Exit.TButton', command=self.quit_app)
        exit_btn.grid(row=4, column=0, sticky="ew", pady=8)
        ToolTip(exit_btn, "Exit the application.")
        card.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(card, mode='indeterminate', length=220)
        self.progress.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        # Video frame panel
        self.frame_panel = tk.Label(main_frame, bg="#222", width=480, height=320)
        self.frame_panel.pack(pady=10)
        self.result_label = tk.Label(main_frame, text="Recognition results will appear here.", font=("Arial", 12), bg="#f4f7fa", fg="#333")
        self.result_label.pack(pady=5)
        # Status bar at the bottom
        status_bar = ttk.Frame(self.root, style='Card.TFrame', height=32)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = ttk.Label(status_bar, text="Ready to start face recognition", style='Status.TLabel', anchor="w")
        self.status_label.pack(fill=tk.X, padx=10, pady=4)
        # Start camera and detection loop (but NOT recognition) automatically
        self.cap = cv2.VideoCapture(0)
        self.stop_event.clear()
        self.recognition_thread = threading.Thread(target=self.recognition_loop, daemon=True)
        self.recognition_thread.start()

    def toggle_recognition(self):
        if not self.is_recognition_running:
            self.start_face_recognition()
        else:
            self.stop_face_recognition()

    def start_face_recognition(self):
        self.update_status_text("Starting face recognition via API...")
        self.is_recognition_running = True
        self.start_btn.configure(text="Stop Recognition", style='Stop.TButton')
        self.progress.start()
        self.stop_event.clear()
        self.cap = cv2.VideoCapture(0)
        self.recognition_thread = threading.Thread(target=self.recognition_loop, daemon=True)
        self.recognition_thread.start()

    def stop_face_recognition(self):
        self.is_recognition_running = False
        self.start_btn.configure(text="Start Face Recognition", style='Start.TButton')
        self.progress.stop()
        self.stop_event.set()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.update_status_text("Face recognition stopped")

    def recognition_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read() if self.cap else (False, None)
            if not ret or frame is None:
                self.update_status_text("Failed to capture frame from camera.")
                break
            display_frame = frame.copy()
            # Always run detection for bounding boxes and names
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            detected_faces = []
            try:
                with open(temp_path, "rb") as img_file:
                    files = {"file": img_file}
                    response = requests.post(API_URL, files=files, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    detected_faces = data.get("recognized", [])
                    self.last_detection_boxes = detected_faces
            except Exception as e:
                print(f"[DEBUG] Exception in detection: {e}")
            # Draw bounding boxes and names for detection
            for person in self.last_detection_boxes:
                box = person.get("box", None)
                name = person.get("name", "Unknown")
                if box and len(box) == 4:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Show annotated frame in GUI
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img.resize((480, 320)))
            if self.frame_panel is not None:
                self._imgtk = imgtk  # Store reference to avoid garbage collection
                self.frame_panel.config(image=imgtk)
            # Only do recognition/attendance marking if triggered
            if self.is_recognition_running and not self.is_paused:
                self.is_recognition_running = False  # Only run once per button press
                marked = False
                for person in self.last_detection_boxes:
                    name = person.get("name", "Unknown")
                    attendance = person.get("attendance", "")
                    if attendance in ["in", "out", "already_marked"]:
                        self.is_paused = True
                        self.show_attendance_popup(frame, name, attendance)
                        marked = True
                        break
                if not marked and self.result_label is not None:
                    self.result_label.config(text="No attendance marked.", fg="#b71c1c")
            time.sleep(1)  # Adjust interval as needed

    def trigger_recognition(self):
        # Allow repeated recognition attempts
        if not self.is_recognition_running and not self.is_paused:
            self.is_recognition_running = True
            self.start_btn.configure(text="Recognize", style='Start.TButton')

    def show_attendance_popup(self, frame, name, attendance):
        # Draw the message on the image using OpenCV
        display_img = frame.copy()
        if attendance == "already_marked":
            msg = f"Attendance already marked for {name}"
            color = (184, 134, 11)  # Dark goldenrod
        else:
            msg = f"Attendance marked for {name} ({attendance.upper()})"
            color = (34, 139, 34)  # Forest green
        cv2.rectangle(display_img, (0, 0), (480, 40), color, -1)
        cv2.putText(display_img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img.resize((480, 320)))
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Attendance Status")
        popup.geometry("500x350")
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.grab_set()
        label_img = tk.Label(popup, image=imgtk)
        # Keep a reference in a local variable to avoid garbage collection
        self._popup_imgtk = imgtk
        label_img.pack(pady=10)
        label_msg = tk.Label(popup, text=msg, font=("Arial", 14, "bold"), fg="#fff", bg="#222")
        label_msg.pack(pady=5, fill=tk.X)
        # Schedule popup to close after 7 seconds and reset interface
        def close_popup():
            popup.destroy()
            self.reset_interface_after_popup()
            self.is_paused = False
        popup.after(7000, close_popup)

    def reset_interface_after_popup(self):
        # Reset recognize button and any other interface elements if needed
        if self.start_btn is not None:
            self.start_btn.configure(text="Recognize", style='Start.TButton')
        if self.result_label is not None:
            self.result_label.config(text="Recognition results will appear here.", fg="#333")
        self.is_paused = False  # Ensure recognition can be triggered again
        self.update_status()

    def capture_face(self):
        """Capture new face with proper face detection and cropping"""
        def start_capture():
            person_name = name_var.get().strip()
            if not person_name:
                messagebox.showerror("Error", "Name cannot be empty.")
                return
            
            # Close the input dialog
            top.destroy()
            
            # Start face capture process
            self.capture_face_with_detection(person_name)
        
        top = tk.Toplevel(self.root)
        top.title("Capture New Face")
        top.geometry("300x150")
        top.resizable(False, False)
        top.transient(self.root)
        top.grab_set()
        
        # Instructions
        tk.Label(top, text="Enter the person's name:", font=("Arial", 12)).pack(pady=10)
        
        name_var = tk.StringVar()
        name_entry = tk.Entry(top, textvariable=name_var, font=("Arial", 12))
        name_entry.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(top)
        button_frame.pack(pady=10)
        
        ok_btn = tk.Button(button_frame, text="Start Capture", command=start_capture, 
                          bg="#4CAF50", fg="white", font=("Arial", 11, "bold"))
        ok_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=top.destroy,
                              bg="#f44336", fg="white", font=("Arial", 11, "bold"))
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        name_entry.focus()
        name_entry.bind('<Return>', lambda e: start_capture())

    def capture_face_with_detection(self, person_name):
        """Capture face using face detection and cropping"""
        # Create capture window
        capture_window = tk.Toplevel(self.root)
        capture_window.title("Face Capture - Press 'C' to capture, 'Q' to quit")
        capture_window.geometry("800x600")
        capture_window.resizable(False, False)
        capture_window.transient(self.root)
        capture_window.grab_set()
        
        # Create video display
        video_label = tk.Label(capture_window, bg="black")
        video_label.pack(pady=10)
        
        # Status label
        status_label = tk.Label(capture_window, text="Position your face in the frame", 
                               font=("Arial", 12), fg="#333")
        status_label.pack(pady=5)
        
        # Instructions
        instructions = tk.Label(capture_window, 
                              text="Instructions:\n• Position your face clearly in the frame\n• Press 'C' to capture when face is detected\n• Press 'Q' to quit", 
                              font=("Arial", 10), fg="#666")
        instructions.pack(pady=5)
        
        # Variables
        captured_face = None
        face_detected = False
        detected_boxes = []
        cap = cv2.VideoCapture(0)
        
        def update_frame():
            nonlocal captured_face, face_detected, detected_boxes
            
            ret, frame = cap.read()
            if not ret:
                status_label.config(text="Error: Cannot read from camera")
                return
            
            # Resize frame for display
            display_frame = cv2.resize(frame, (640, 480))
            
            # Send frame to cloud API for face detection
            temp_path = "temp_capture_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            face_detected = False
            detected_boxes.clear()
            
            try:
                with open(temp_path, "rb") as img_file:
                    files = {"file": img_file}
                    response = requests.post(API_URL, files=files, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    detected_boxes = data.get("recognized", [])
                    
                    # Draw detection boxes
                    for person in detected_boxes:
                        box = person.get("box", None)
                        if box and len(box) == 4:
                            x1, y1, x2, y2 = box
                            # Scale coordinates for display
                            scale_x = 640 / frame.shape[1]
                            scale_y = 480 / frame.shape[0]
                            x1_scaled = int(x1 * scale_x)
                            y1_scaled = int(y1 * scale_y)
                            x2_scaled = int(x2 * scale_x)
                            y2_scaled = int(y2 * scale_y)
                            
                            cv2.rectangle(display_frame, (x1_scaled, y1_scaled), 
                                        (x2_scaled, y2_scaled), (0, 255, 0), 2)
                            cv2.putText(display_frame, "Face Detected", (x1_scaled, y1_scaled - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            face_detected = True
                            
            except Exception as e:
                print(f"API Error during capture: {e}")
            
            # Update status
            if face_detected:
                status_label.config(text="Face detected - Press 'C' to capture", fg="#4CAF50")
            else:
                status_label.config(text="No face detected - Position your face clearly", fg="#f44336")
            
            # Display frame
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.config(image=imgtk)
            # Store reference to prevent garbage collection
            setattr(video_label, 'imgtk', imgtk)
            
            # Schedule next update
            capture_window.after(100, update_frame)
        
        def on_key(event):
            nonlocal captured_face, face_detected, detected_boxes
            
            if event.char.lower() == 'c' and face_detected:
                # Capture the face
                ret, current_frame = cap.read()
                if ret:
                    for person in detected_boxes:
                        box = person.get("box", None)
                        if box and len(box) == 4:
                            x1, y1, x2, y2 = box
                            
                            # Add padding around the face
                            padding = 20
                            y1_padded = max(0, y1 - padding)
                            y2_padded = min(current_frame.shape[0], y2 + padding)
                            x1_padded = max(0, x1 - padding)
                            x2_padded = min(current_frame.shape[1], x2 + padding)
                            
                            captured_face = current_frame[y1_padded:y2_padded, x1_padded:x2_padded]
                            
                            # Save the face
                            self.save_captured_face(captured_face, person_name)
                            cap.release()
                            capture_window.destroy()
                            return
                
                messagebox.showwarning("Warning", "No face detected. Please try again.")
                
            elif event.char.lower() == 'q':
                cap.release()
                capture_window.destroy()
        
        # Bind keyboard events
        capture_window.bind('<Key>', on_key)
        capture_window.focus_set()
        
        # Start frame updates
        update_frame()
        
        # Wait for window to close
        capture_window.wait_window()

    def save_captured_face(self, face_image, person_name):
        """Save the captured face to known_faces directory (automated, no file dialog)"""
        try:
            # Create known_faces directory if it doesn't exist
            known_faces_dir = "known_faces"
            os.makedirs(known_faces_dir, exist_ok=True)
            
            # Clean the person name (remove spaces, special characters)
            clean_name = person_name.replace(" ", "_").replace("-", "_")
            clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")
            
            # Create filename
            filename = f"{clean_name}.png"
            filepath = os.path.join(known_faces_dir, filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                response = messagebox.askyesno("File Exists", 
                                            f"{filename} already exists. Do you want to overwrite it?")
                if not response:
                    return
            
            # Save the face image
            cv2.imwrite(filepath, face_image)
            
            # Show success message
            messagebox.showinfo("Success", 
                              f"Face saved as: {filename}\nLocation: {os.path.abspath(filepath)}")
            
            # Upload to cloud server if needed
            self.upload_face_to_cloud(filepath, filename)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save face: {e}")

    def upload_face_to_cloud(self, filepath, filename):
        """Upload the captured face to the cloud server automatically using scp"""
        import subprocess
        try:
            remote_path = f"{CLOUD_USER}@{CLOUD_IP}:{CLOUD_KNOWN_FACES_PATH}{filename}"
            result = subprocess.run([
                "scp", filepath, remote_path
            ], capture_output=True, text=True)
            if result.returncode == 0:
                messagebox.showinfo("Upload Success", f"Face uploaded to cloud server at:\n{remote_path}")
            else:
                messagebox.showerror("Upload Failed", f"Failed to upload face to cloud.\n\nError: {result.stderr}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload face: {e}")

    def view_attendance(self):
        # Download attendance CSV from server (if available)
        try:
            response = requests.get(ATTENDANCE_URL, timeout=10)
            if response.status_code == 200:
                with open("attendance_downloaded.csv", "wb") as f:
                    f.write(response.content)
                df = pd.read_csv("attendance_downloaded.csv")
                self.show_attendance_window(df)
            else:
                messagebox.showerror("Error", f"Failed to fetch attendance: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch attendance: {e}")

    def show_attendance_window(self, df):
        top = tk.Toplevel(self.root)
        top.title("Attendance Records")
        text = scrolledtext.ScrolledText(top, width=80, height=25)
        text.pack(padx=10, pady=10)
        text.insert(tk.END, df.to_string(index=False))
        text.config(state=tk.DISABLED)

    def show_settings(self):
        top = tk.Toplevel(self.root)
        top.title("Settings & Info")
        info = f"API URL: {API_URL}\nAttendance URL: {ATTENDANCE_URL}\nCamera: 0 (default)\n"
        tk.Label(top, text=info, font=("Arial", 12)).pack(padx=10, pady=10)

    def update_status_text(self, text):
        self.status_label.config(text=text)

    def update_status(self):
        self.status_label.config(text="Ready to start face recognition via API")

    def quit_app(self):
        self.stop_face_recognition()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceRecognitionGUI()
    app.run() 