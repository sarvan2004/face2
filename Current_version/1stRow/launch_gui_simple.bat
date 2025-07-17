echo off
echo Face Recognition GUI Launcher (Tkinter)
echo ======================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

REM Launch the simple GUI
echo Starting Face Recognition GUI (Tkinter)...
python launch_gui_simple.py

REM If there was an error, pause to show the message
if errorlevel 1 (
    echo.
    echo Press any key to exit...
    pause >nul
) 