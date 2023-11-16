@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Create the directory if it does not exist
IF NOT EXIST "external" (
    MKDIR "external"
)

REM Download OpenCV
PowerShell -Command "& { Invoke-WebRequest -Uri 'https://github.com/opencv/opencv/releases/download/4.8.1/opencv-4.8.1-windows.exe' -OutFile 'external\opencv-4.8.1-windows.exe' }"

REM Run the OpenCV executable for extraction
START /WAIT external\opencv-4.8.1-windows.exe

REM Download and extract Eigen
PowerShell -Command "& { Invoke-WebRequest -Uri 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip' -OutFile 'external\eigen-3.4.0.zip' }"
PowerShell -Command "& { Expand-Archive -LiteralPath 'external\eigen-3.4.0.zip' -DestinationPath 'external\' -Force }"

REM Download and extract ByteTrack-Eigen
PowerShell -Command "& { Invoke-WebRequest -Uri 'https://github.com/cj-mills/byte-track-eigen/releases/download/1.0.0/bytetrack-eigen-1.0.0-windows.zip' -OutFile 'external\bytetrack-eigen-1.0.0-windows.zip' }"
PowerShell -Command "& { Expand-Archive -LiteralPath 'external\bytetrack-eigen-1.0.0-windows.zip' -DestinationPath 'external\' -Force }"

echo Done