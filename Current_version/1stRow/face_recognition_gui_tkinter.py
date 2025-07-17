#!/usr/bin/env python3
"""
Face Recognition GUI Application (Tkinter Version)
A cross-platform GUI for the face recognition attendance system
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import os
import sys
import threading
import time
import pandas as pd
from datetime import datetime
import cv2

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
        self.root.title("Face Recognition System")
        self.root.geometry("700x570")
        self.root.configure(bg="#f4f7fa")
        self.root.resizable(True, False)
        self.dark_mode = False
        self.setup_styles()
        self.recognition_process = None
        self.is_recognition_running = False
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
            # Fix blinking by mapping colors for all states
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
            # Fix blinking by mapping colors for all states
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
        # Header with logo and title
        header = ttk.Frame(self.root, style='Header.TFrame', height=70)
        header.pack(fill=tk.X, side=tk.TOP)
        header.grid_propagate(False)
        
        # SEMICORE LABS Logo
        logo_canvas = tk.Canvas(header, width=200, height=60, bg=('#23272e' if self.dark_mode else '#1a2233'), highlightthickness=0)
        logo_canvas.grid(row=0, column=0, rowspan=2, padx=(20, 10), pady=5)
        
        # Draw SEMICORE LABS text
        logo_canvas.create_text(100, 25, text="SEMICORE LABS", fill="white", font=("Arial", 16, "bold"))
        
        # Draw distinctive graphic element in the 'O' (circuit-like pattern)
        # Position dots in a radiating pattern within the 'O' of SEMICORE
        center_x, center_y = 100, 25
        for i in range(8):
            angle = i * 45
            radius = 8
            x = center_x + radius * (angle % 90 == 0) * (1 if angle < 180 else -1)
            y = center_y + radius * (angle % 90 != 0) * (1 if 90 < angle < 270 else -1)
            logo_canvas.create_oval(x-2, y-2, x+2, y+2, fill="#2196F3", outline="")
        
        # Draw tagline
        logo_canvas.create_text(100, 45, text="Engineering Your Ideas", fill="#b0c4de", font=("Arial", 9, "normal"))
        
        # Title and subtitle
        title = ttk.Label(header, text="Face Recognition Attendance System", style='Header.TLabel')
        title.grid(row=0, column=1, sticky="w", pady=(10, 0))
        subtitle = ttk.Label(header, text="Modern, Fast, Accurate", style='SubHeader.TLabel')
        subtitle.grid(row=1, column=1, sticky="w", pady=(0, 10))
        header.columnconfigure(1, weight=1)
        
        # Dark mode toggle
        dark_btn = ttk.Button(header, text=("🌙 Dark Mode" if not self.dark_mode else "☀️ Light Mode"), width=12, command=self.toggle_dark_mode)
        dark_btn.grid(row=0, column=2, rowspan=2, padx=20, pady=10, sticky="e")
        ToolTip(dark_btn, "Toggle dark/light mode.")

        # Main content frame
        main_frame = ttk.Frame(self.root, style='Main.TFrame', padding="30 10 30 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Section heading
        section_label = ttk.Label(main_frame, text="Main Actions", font=("Arial", 13, "bold"), background=('#181a1b' if self.dark_mode else '#f4f7fa'), foreground=('#b0c4de' if self.dark_mode else '#1a2233'))
        section_label.pack(anchor="w", pady=(0, 5))
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 15))

        # Card frame for buttons
        card = ttk.Frame(main_frame, style='Card.TFrame', padding="30 25 30 25")
        card.pack(pady=10, ipadx=10, ipady=10)

        # Buttons with emoji icons
        self.start_btn = ttk.Button(card, text="🟢 Start Face Recognition", style='Start.TButton', command=self.toggle_recognition)
        self.start_btn.grid(row=0, column=0, sticky="ew", pady=8)
        ToolTip(self.start_btn, "Start or stop the face recognition system.")

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

        # Progress bar
        self.progress = ttk.Progressbar(card, mode='indeterminate', length=220)
        self.progress.grid(row=5, column=0, sticky="ew", pady=(12, 0))

        # Status bar at the bottom
        status_bar = ttk.Frame(self.root, style='Card.TFrame', height=32)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = ttk.Label(status_bar, text="Ready to start face recognition", style='Status.TLabel', anchor="w")
        self.status_label.pack(fill=tk.X, padx=10, pady=4)

    def toggle_recognition(self):
        """Toggle face recognition on/off"""
        if not self.is_recognition_running:
            self.start_face_recognition()
        else:
            self.stop_face_recognition()
    
    def start_face_recognition(self):
        """Start the face recognition process"""
        try:
            self.update_status_text("Starting face recognition...")
            self.is_recognition_running = True
            self.start_btn.configure(text="Stop Recognition", style='Stop.TButton')
            self.progress.start()
            
            # Run the recognition in a separate thread
            def run_recognition():
                try:
                    self.recognition_process = subprocess.Popen(
                        [sys.executable, "app.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    self.recognition_process.wait()
                except Exception as e:
                    print(f"Error running recognition: {e}")
                finally:
                    self.root.after(0, self.recognition_finished)
            
            threading.Thread(target=run_recognition, daemon=True).start()
            
        except Exception as e:
            self.update_status_text(f"Error starting recognition: {str(e)}")
            self.is_recognition_running = False
            self.start_btn.configure(text="Start Face Recognition", style='Start.TButton')
            self.progress.stop()
    
    def stop_face_recognition(self):
        """Stop the face recognition process"""
        if self.recognition_process:
            self.recognition_process.terminate()
            self.recognition_process = None
        
        self.is_recognition_running = False
        self.start_btn.configure(text="Start Face Recognition", style='Start.TButton')
        self.progress.stop()
        self.update_status_text("Face recognition stopped")
    
    def recognition_finished(self):
        """Called when recognition process finishes"""
        self.is_recognition_running = False
        self.start_btn.configure(text="Start Face Recognition", style='Start.TButton')
        self.progress.stop()
        self.update_status_text("Face recognition finished")
    
    def capture_face(self):
        """Run the face capture tool with a name prompt"""
        def on_ok():
            person_name = name_var.get().strip()
            if not person_name:
                messagebox.showerror("Error", "Name cannot be empty.")
                return
            name_dialog.destroy()
            self.update_status_text(f"Opening face capture tool for: {person_name}")
            try:
                subprocess.Popen([sys.executable, "capture_face.py", person_name])
                self.update_status_text(f"Face capture tool opened for: {person_name}")
            except Exception as e:
                self.update_status_text(f"Error opening face capture: {str(e)}")
                messagebox.showerror("Error", f"Could not open face capture tool:\n{str(e)}")

        # Create dialog to ask for name
        name_dialog = tk.Toplevel(self.root)
        name_dialog.title("Enter Name")
        name_dialog.geometry("300x120")
        name_dialog.resizable(False, False)
        name_dialog.grab_set()
        
        ttk.Label(name_dialog, text="Enter the person's name:").pack(pady=(15, 5))
        name_var = tk.StringVar()
        name_entry = ttk.Entry(name_dialog, textvariable=name_var, width=30)
        name_entry.pack(pady=5)
        name_entry.focus_set()
        
        button_frame = ttk.Frame(name_dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=name_dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        name_dialog.bind('<Return>', lambda event: on_ok())
        name_dialog.bind('<Escape>', lambda event: name_dialog.destroy())
    
    def view_attendance(self):
        """Show attendance data in a new window"""
        try:
            if os.path.exists("attendance.csv"):
                df = pd.read_csv("attendance.csv")
                if not df.empty:
                    # Create attendance window
                    attendance_window = tk.Toplevel(self.root)
                    attendance_window.title("Attendance Records")
                    attendance_window.geometry("600x400")
                    attendance_window.resizable(True, True)
                    
                    # Create text widget
                    text_widget = scrolledtext.ScrolledText(attendance_window, 
                                                          wrap=tk.WORD,
                                                          font=('Courier', 10))
                    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                    
                    # Format attendance data
                    attendance_text = "Attendance Records:\n\n"
                    attendance_text += "Date\t\tName\t\tTime\tType\n"
                    attendance_text += "-" * 65 + "\n"
                    
                    for _, row in df.iterrows():
                        date = row.get('Date', 'N/A')
                        name = row.get('Name', 'N/A')
                        time = row.get('Time', 'N/A')
                        typ = row.get('Type', 'N/A')
                        attendance_text += f"{date}\t{name}\t\t{time}\t{typ}\n"
                    
                    text_widget.insert(tk.END, attendance_text)
                    text_widget.configure(state=tk.DISABLED)
                    
                else:
                    messagebox.showinfo("No Records", "The attendance file is empty.")
            else:
                messagebox.showinfo("No File", "No attendance.csv file found.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error reading attendance: {str(e)}")
    
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)
        
        # Create settings content
        main_frame = ttk.Frame(settings_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Known faces info
        known_faces_count = self.get_known_faces_count()
        ttk.Label(main_frame, text="System Information", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(main_frame, text=f"Known faces: {known_faces_count}").pack(anchor=tk.W)
        
        # File paths
        attendance_path = os.path.abspath("attendance.csv")
        known_faces_path = os.path.abspath("known_faces")
        
        ttk.Label(main_frame, text="\nFile Paths:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        ttk.Label(main_frame, text=f"Attendance: {attendance_path}").pack(anchor=tk.W)
        ttk.Label(main_frame, text=f"Known faces: {known_faces_path}").pack(anchor=tk.W)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=settings_window.destroy).pack(pady=(20, 0))
    
    def get_known_faces_count(self):
        """Get the number of known faces"""
        try:
            if os.path.exists("known_faces"):
                files = [f for f in os.listdir("known_faces") if f.endswith(('.png', '.jpg', '.jpeg'))]
                return len(files)
            return 0
        except:
            return 0
    
    def update_status_text(self, text):
        """Update the status label text"""
        self.status_label.configure(text=text)
    
    def update_status(self):
        """Periodic status update"""
        if self.is_recognition_running:
            self.update_status_text("Face recognition is running...")
        else:
            if not self.status_label.cget("text").startswith("Error"):
                self.update_status_text("Ready to start face recognition")
        
        # Schedule next update
        self.root.after(1000, self.update_status)
    
    def quit_app(self):
        """Quit the application"""
        if self.is_recognition_running:
            self.stop_face_recognition()
        self.root.quit()
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

def main():
    """Main function"""
    try:
        app = FaceRecognitionGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 