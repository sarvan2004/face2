# Face Recognition GUI

A Tkinter-based graphical user interface for the face recognition attendance system.

## Features

- **Start/Stop Face Recognition**: Launch the main face recognition application (`app_improved.py`)
- **Capture New Face**: Add new faces to the recognition database
- **View Attendance**: Display attendance records in a readable format
- **Settings**: View system information and file paths
- **Real-time Status**: See the current status of the recognition system

## Installation

### Prerequisites

**No GTK3 required!** Tkinter comes built-in with Python.

1. **Install Python Dependencies**:
   ```bash
   pip install pandas opencv-python ultralytics deepface numpy
   ```

### System Requirements

- Python 3.7 or higher
- Webcam access
- Sufficient disk space for face images and attendance records

## Usage

### Starting the GUI

1. Navigate to the project directory:
   ```bash
   cd DeepTrack/Current_version/1stRow
   ```

2. Run the GUI application:
   ```bash
   python launch_gui_simple.py
   ```

   **Or on Windows:** Double-click `launch_gui_simple.bat`

### Using the Interface

#### 1. Start Face Recognition
- Click the "Start Face Recognition" button to begin the attendance system
- The button will change to "Stop Recognition" while running
- Click again to stop the recognition process

#### 2. Capture New Face
- Click "Capture New Face" to add a new person to the system
- This opens the face capture tool in a separate window
- Follow the instructions in the capture tool to add a new face

#### 3. View Attendance
- Click "View Attendance" to see all attendance records
- Records are displayed in a scrollable dialog
- Shows Date, Name, and Time for each attendance entry

#### 4. Settings
- Click "Settings" to view system information
- Shows the number of known faces
- Displays file paths for attendance and known faces directories

#### 5. Exit
- Click "Exit" to close the application
- Any running recognition process will be stopped automatically

## File Structure

```
1stRow/
├── face_recognition_gui_tkinter.py  # Main GUI application
├── launch_gui_simple.py             # Smart launcher
├── launch_gui_simple.bat            # Windows batch file
├── app_improved.py                  # Face recognition engine
├── capture_face.py                  # Face capture tool
├── attendance.csv                   # Attendance records
├── known_faces/                    # Directory for face images
└── GUI_README.md                   # This file
```

## Troubleshooting

### Common Issues

1. **Webcam not working**:
   - Ensure webcam permissions are granted
   - Check if another application is using the webcam

2. **Face recognition not starting**:
   - Check if `app_improved.py` exists and is executable
   - Verify all dependencies are installed
   - Check the console for error messages

3. **GUI not displaying properly**:
   - Update Python to the latest version
   - Check if your system supports Tkinter

### Error Messages

- **"Error starting recognition"**: Check if `app_improved.py` exists and all dependencies are installed
- **"Error opening face capture"**: Verify `capture_face.py` exists
- **"No attendance file"**: The attendance.csv file hasn't been created yet

## Development

### Modifying the GUI

1. **Edit the Python file**:
   - Modify `face_recognition_gui_tkinter.py` to add new functionality
   - Add new buttons and handlers as needed

2. **Customize appearance**:
   - Edit the styling in the `setup_styles()` method
   - Change colors, fonts, and layout

3. **Add new features**:
   - Create new button handlers
   - Add new windows or dialogs
   - Integrate with additional scripts

### Adding New Features

1. Add new buttons to the GUI
2. Create corresponding handler methods
3. Connect the buttons to the handlers
4. Test the new functionality

## Security Notes

- Face images are stored locally in the `known_faces` directory
- Attendance records are stored in `attendance.csv`
- No data is transmitted over the network
- Ensure proper file permissions for sensitive data

## Advantages of Tkinter Version

- ✅ **No external dependencies** (GTK3 not needed)
- ✅ **Works on all platforms** (Windows, Linux, Mac)
- ✅ **Built into Python**
- ✅ **Modern interface** with progress bars and status updates
- ✅ **Error handling** with user-friendly dialogs
- ✅ **Cross-platform compatibility**

## License

This GUI is part of the face recognition attendance system project. 