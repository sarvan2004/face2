# Raspberry Pi CM5 Setup Guide for Face Recognition System

## 1. Hardware Requirements

### Essential Components:
- **Raspberry Pi CM5** (Compute Module 5)
- **CM5 Development Kit** or **CM5 Carrier Board**
- **Camera Module** (Pi Camera v3 or USB camera)
- **Power Supply** (5V/3A minimum)
- **MicroSD Card** (32GB+ recommended)
- **Display** (HDMI monitor or touchscreen)
- **Keyboard & Mouse** (for initial setup)

### Optional Components:
- **Case/Enclosure** for protection
- **Cooling fan** for extended operation
- **USB WiFi adapter** (if not using Ethernet)

## 2. Software Setup

### Step 1: Flash Raspberry Pi OS
```bash
# Download Raspberry Pi OS (64-bit recommended for CM5)
# Use Raspberry Pi Imager to flash to microSD card
# Enable SSH and set WiFi during imaging
```

### Step 2: Initial Boot and Configuration
```bash
# Connect to CM5 via SSH or directly
ssh pi@your_cm5_ip

# Update system
sudo apt update && sudo apt upgrade -y

# Enable camera interface
sudo raspi-config nonint do_camera 0

# Enable SPI and I2C (if needed)
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0

# Reboot
sudo reboot
```

### Step 3: Install Python Dependencies
```bash
# Install system packages
sudo apt install -y python3 python3-pip python3-tk python3-pil python3-pil.imagetk
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libjasper-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5

# Install Python packages
pip3 install --user opencv-python
pip3 install --user pillow
pip3 install --user requests
pip3 install --user pandas
```

### Step 4: Test Camera
```bash
# Test camera functionality
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('Camera working!')
    cap.release()
else:
    print('Camera not found')
"
```

## 3. Application Setup

### Step 1: Copy Application Files
```bash
# From your development machine
scp -r DeepTrack/Current_version/1stRow/ pi@your_cm5_ip:/home/pi/

# Or download directly on CM5
cd /home/pi
wget https://your-repo-url/DeepTrack.zip
unzip DeepTrack.zip
```

### Step 2: Configure Application
```bash
cd /home/pi/DeepTrack/Current_version/1stRow/

# Edit API URLs if needed
nano face_recognition_gui_cm5.py
# Change API_URL to your cloud server IP
```

### Step 3: Test Application
```bash
# Test the GUI
python3 face_recognition_gui_cm5.py

# Or test the original version
python3 face_recognition_gui_api.py
```

## 4. Network Configuration

### Step 1: Ensure Network Connectivity
```bash
# Test internet connection
ping google.com

# Test connection to cloud server
curl -I http://13.201.230.71:5000/attendance
```

### Step 2: Configure Firewall (if needed)
```bash
# Allow outbound connections
sudo ufw allow out 5000
sudo ufw allow out 80
sudo ufw allow out 443
```

## 5. Performance Optimization

### Step 1: Enable GPU Memory Split
```bash
# Edit config.txt
sudo nano /boot/config.txt

# Add these lines:
gpu_mem=128
dtoverlay=vc4-kms-v3d
```

### Step 2: Optimize for Performance
```bash
# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable hciuart

# Increase swap if needed
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo systemctl restart dphys-swapfile
```

## 6. Auto-Start Configuration

### Step 1: Create Startup Script
```bash
# Create startup script
cat > /home/pi/start_face_recognition.sh << 'EOF'
#!/bin/bash
cd /home/pi/DeepTrack/Current_version/1stRow/
python3 face_recognition_gui_cm5.py
EOF

chmod +x /home/pi/start_face_recognition.sh
```

### Step 2: Configure Auto-Start
```bash
# Add to autostart
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/face-recognition.desktop << EOF
[Desktop Entry]
Type=Application
Name=Face Recognition
Exec=/home/pi/start_face_recognition.sh
Terminal=false
EOF
```

## 7. Troubleshooting

### Common Issues:

1. **Camera not detected:**
   ```bash
   # Check camera module
   vcgencmd get_camera
   # Should return: supported=1 detected=1
   ```

2. **GUI not starting:**
   ```bash
   # Check display
   echo $DISPLAY
   # Should return: :0
   
   # Start X server if needed
   startx
   ```

3. **Network connection issues:**
   ```bash
   # Test API connection
   curl -v http://13.201.230.71:5000/attendance
   
   # Check DNS
   nslookup google.com
   ```

4. **Performance issues:**
   ```bash
   # Monitor system resources
   htop
   
   # Check temperature
   vcgencmd measure_temp
   ```

## 8. Testing Checklist

- [ ] Camera working
- [ ] GUI starts without errors
- [ ] Can connect to cloud API
- [ ] Face detection working
- [ ] Attendance marking working
- [ ] Auto-start configured
- [ ] Performance acceptable

## 9. Maintenance

### Regular Updates:
```bash
# Update system monthly
sudo apt update && sudo apt upgrade -y

# Update Python packages
pip3 list --outdated
pip3 install --user --upgrade package_name
```

### Log Monitoring:
```bash
# Check application logs
tail -f /var/log/syslog | grep python

# Monitor disk space
df -h
```

## 10. Security Considerations

- Change default password
- Use SSH keys instead of passwords
- Keep system updated
- Monitor network traffic
- Use firewall rules

## 11. Backup Strategy

```bash
# Backup application files
tar -czf face_recognition_backup_$(date +%Y%m%d).tar.gz /home/pi/DeepTrack/

# Backup system image
sudo dd if=/dev/mmcblk0 of=cm5_backup_$(date +%Y%m%d).img bs=4M
``` 