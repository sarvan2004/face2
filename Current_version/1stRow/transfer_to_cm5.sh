#!/bin/bash
# File Transfer Script for CM5 Setup

echo "=========================================="
echo "File Transfer to Raspberry Pi CM5"
echo "=========================================="

# Get CM5 IP address
read -p "Enter CM5 IP address: " CM5_IP

# Check if IP is provided
if [ -z "$CM5_IP" ]; then
    echo "Error: IP address is required"
    exit 1
fi

# Test connection
echo "Testing connection to $CM5_IP..."
if ! ping -c 1 "$CM5_IP" &> /dev/null; then
    echo "Error: Cannot reach $CM5_IP"
    exit 1
fi

echo "✓ Connection successful"

# Create transfer directory on CM5
echo "Creating directory structure on CM5..."
ssh pi@"$CM5_IP" "mkdir -p /home/pi/DeepTrack/Current_version/1stRow/"

# Transfer files
echo "Transferring files to CM5..."

# Transfer main application files
scp face_recognition_gui_cm5.py pi@"$CM5_IP":/home/pi/DeepTrack/Current_version/1stRow/
scp face_recognition_gui_api.py pi@"$CM5_IP":/home/pi/DeepTrack/Current_version/1stRow/
scp face_api.py pi@"$CM5_IP":/home/pi/DeepTrack/Current_version/1stRow/

# Transfer model file if it exists
if [ -f "yolov11n-face.pt" ]; then
    scp yolov11n-face.pt pi@"$CM5_IP":/home/pi/DeepTrack/Current_version/1stRow/
fi

# Transfer known_faces directory if it exists
if [ -d "known_faces" ]; then
    scp -r known_faces pi@"$CM5_IP":/home/pi/DeepTrack/Current_version/1stRow/
fi

# Transfer configuration files
scp recognition_config.json pi@"$CM5_IP":/home/pi/DeepTrack/Current_version/1stRow/ 2>/dev/null || echo "recognition_config.json not found"
scp roi_config_first_row.json pi@"$CM5_IP":/home/pi/DeepTrack/Current_version/1stRow/ 2>/dev/null || echo "roi_config_first_row.json not found"

# Transfer setup scripts
scp setup_cm5.sh pi@"$CM5_IP":/home/pi/
scp cm5_setup_guide.md pi@"$CM5_IP":/home/pi/

echo "✓ Files transferred successfully"

# Run setup on CM5
echo "Running setup on CM5..."
ssh pi@"$CM5_IP" "cd /home/pi && chmod +x setup_cm5.sh && ./setup_cm5.sh"

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To connect to CM5:"
echo "  ssh pi@$CM5_IP"
echo ""
echo "To test the system:"
echo "  ssh pi@$CM5_IP 'python3 /home/pi/test_face_recognition.py'"
echo ""
echo "To run the application:"
echo "  ssh pi@$CM5_IP 'python3 /home/pi/DeepTrack/Current_version/1stRow/face_recognition_gui_cm5.py'"
echo "==========================================" 