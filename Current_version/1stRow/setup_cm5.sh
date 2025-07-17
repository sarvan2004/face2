#!/bin/bash
# Automated Setup Script for Raspberry Pi CM5 Face Recognition System

set -e  # Exit on any error

echo "=========================================="
echo "Raspberry Pi CM5 Face Recognition Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run this script as root"
    exit 1
fi

# Check if we're on a Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    print_warning "This script is designed for Raspberry Pi. Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_status "Starting CM5 setup..."

# Step 1: Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-tk python3-pil python3-pil.imagetk
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libjasper-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install -y curl wget git htop

# Step 3: Install Python packages
print_status "Installing Python packages..."
pip3 install --user opencv-python
pip3 install --user pillow
pip3 install --user requests
pip3 install --user pandas

# Step 4: Enable camera interface
print_status "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Step 5: Enable SPI and I2C
print_status "Enabling SPI and I2C interfaces..."
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0

# Step 6: Performance optimizations
print_status "Applying performance optimizations..."

# Disable unnecessary services
sudo systemctl disable bluetooth 2>/dev/null || true
sudo systemctl disable hciuart 2>/dev/null || true

# Configure GPU memory
if ! grep -q "gpu_mem=128" /boot/config.txt; then
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt
fi

if ! grep -q "dtoverlay=vc4-kms-v3d" /boot/config.txt; then
    echo "dtoverlay=vc4-kms-v3d" | sudo tee -a /boot/config.txt
fi

# Increase swap size
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo systemctl restart dphys-swapfile

# Step 7: Test camera
print_status "Testing camera..."
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✓ Camera working!')
    cap.release()
else:
    print('✗ Camera not found')
"

# Step 8: Create application directory
print_status "Setting up application directory..."
mkdir -p /home/pi/DeepTrack/Current_version/1stRow/

# Step 9: Create startup script
print_status "Creating startup script..."
cat > /home/pi/start_face_recognition.sh << 'EOF'
#!/bin/bash
cd /home/pi/DeepTrack/Current_version/1stRow/
export DISPLAY=:0
python3 face_recognition_gui_cm5.py
EOF

chmod +x /home/pi/start_face_recognition.sh

# Step 10: Configure auto-start
print_status "Configuring auto-start..."
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/face-recognition.desktop << EOF
[Desktop Entry]
Type=Application
Name=Face Recognition
Exec=/home/pi/start_face_recognition.sh
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

# Step 11: Test network connectivity
print_status "Testing network connectivity..."
if ping -c 1 google.com &> /dev/null; then
    print_status "✓ Internet connection working"
else
    print_warning "✗ No internet connection detected"
fi

# Step 12: Create test script
print_status "Creating test script..."
cat > /home/pi/test_face_recognition.py << 'EOF'
#!/usr/bin/env python3
import cv2
import requests
import sys

def test_camera():
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("✓ Camera working")
            return True
        else:
            print("✗ Camera not capturing frames")
            return False
    else:
        print("✗ Camera not found")
        return False

def test_api():
    print("Testing API connection...")
    try:
        response = requests.get("http://13.201.230.71:5000/attendance", timeout=10)
        if response.status_code == 200:
            print("✓ API connection working")
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False

if __name__ == "__main__":
    camera_ok = test_camera()
    api_ok = test_api()
    
    if camera_ok and api_ok:
        print("\n✓ All tests passed! System ready.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Check configuration.")
        sys.exit(1)
EOF

chmod +x /home/pi/test_face_recognition.py

# Step 13: Create desktop shortcut
print_status "Creating desktop shortcut..."
cat > ~/Desktop/FaceRecognition.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Face Recognition
Comment=Face Recognition Attendance System
Exec=python3 /home/pi/DeepTrack/Current_version/1stRow/face_recognition_gui_cm5.py
Icon=applications-graphics
Terminal=false
Categories=Utility;
EOF

chmod +x ~/Desktop/FaceRecognition.desktop

# Step 14: Final instructions
print_status "Setup complete!"
echo ""
echo "=========================================="
echo "NEXT STEPS:"
echo "=========================================="
echo "1. Copy your application files to:"
echo "   /home/pi/DeepTrack/Current_version/1stRow/"
echo ""
echo "2. Test the system:"
echo "   python3 /home/pi/test_face_recognition.py"
echo ""
echo "3. Run the application:"
echo "   python3 /home/pi/DeepTrack/Current_version/1stRow/face_recognition_gui_cm5.py"
echo ""
echo "4. Files to copy from your development machine:"
echo "   - face_recognition_gui_cm5.py"
echo "   - face_recognition_gui_api.py"
echo "   - face_api.py (if running locally)"
echo "   - known_faces/ directory"
echo "   - yolov11n-face.pt"
echo ""
echo "5. Reboot to apply all changes:"
echo "   sudo reboot"
echo "==========================================" 