#!/bin/bash
# Raspberry Pi Setup Script for Face Recognition System

echo "Setting up Face Recognition System on Raspberry Pi..."

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required system packages
echo "Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-tk python3-pil python3-pil.imagetk
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libatlas-base-dev  # For numpy optimization
sudo apt install -y libhdf5-dev libhdf5-serial-dev
sudo apt install -y libjasper-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5

# Install Python packages
echo "Installing Python packages..."
pip3 install --user opencv-python-headless
pip3 install --user pillow
pip3 install --user requests
pip3 install --user pandas

# Enable camera interface
echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Enable SPI and I2C if needed for additional sensors
echo "Enabling SPI and I2C interfaces..."
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0

# Create desktop shortcut
echo "Creating desktop shortcut..."
cat > ~/Desktop/FaceRecognition.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Face Recognition
Comment=Face Recognition Attendance System
Exec=python3 /home/pi/DeepTrack/Current_version/1stRow/face_recognition_gui_pi.py
Icon=applications-graphics
Terminal=false
Categories=Utility;
EOF

chmod +x ~/Desktop/FaceRecognition.desktop

echo "Setup complete! You can now run the face recognition system."
echo "To start: python3 face_recognition_gui_pi.py" 