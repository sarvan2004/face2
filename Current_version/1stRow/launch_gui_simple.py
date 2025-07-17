#!/usr/bin/env python3
"""
Simple launcher for the Face Recognition GUI (Tkinter Version)
No GTK3 required - works on all platforms
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    # Check other dependencies
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        missing_deps.append("ultralytics")
    
    try:
        from deepface import DeepFace
    except ImportError:
        missing_deps.append("deepface")
    
    return missing_deps

def install_dependencies():
    """Install missing dependencies"""
    print("Installing missing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "opencv-python", "ultralytics", "deepface", "numpy"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def main():
    """Main launcher function"""
    print("🎯 Face Recognition GUI Launcher (Tkinter)")
    print("=" * 45)
    
    # Check if we're in the right directory
    if not os.path.exists("face_recognition_gui_tkinter.py"):
        print("❌ Error: face_recognition_gui_tkinter.py not found!")
        print("Please run this script from the project directory.")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("\n📦 Installing dependencies...")
        
        if install_dependencies():
            # Check again after installation
            missing_deps = check_dependencies()
            if missing_deps:
                print(f"❌ Still missing: {', '.join(missing_deps)}")
                print("Please install manually:")
                print("pip install pandas opencv-python ultralytics deepface numpy")
                sys.exit(1)
        else:
            print("❌ Failed to install dependencies automatically.")
            print("Please install manually:")
            print("pip install pandas opencv-python ultralytics deepface numpy")
            sys.exit(1)
    else:
        print("✅ All dependencies are available!")
    
    # Launch the GUI
    print("\n🚀 Launching Face Recognition GUI...")
    try:
        import face_recognition_gui_tkinter
        face_recognition_gui_tkinter.main()
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all Python dependencies are installed")
        print("2. Check if app.py and capture_face.py exist")
        print("3. Ensure you have a webcam connected")
        sys.exit(1)

if __name__ == "__main__":
    main() 