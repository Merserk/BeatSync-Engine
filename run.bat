@echo off
chcp 65001 >nul
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON=%SCRIPT_DIR%bin\python-3.13.13-embed-amd64\python.exe"
set "CUDA=%SCRIPT_DIR%bin\CUDA\v13.0"
set "FFMPEG=%SCRIPT_DIR%bin\ffmpeg"

if not exist "%PYTHON%" (
    echo ERROR: Portable Python was not found.
    echo Run install.bat first.
    pause
    exit /b 1
)

set "CUDA_PATH=%CUDA%"
set "CUDA_HOME=%CUDA%"
set "CUDA_ROOT=%CUDA%"
set "PATH=%CUDA%\bin\x64;%CUDA%\bin;%CUDA%\lib\x64;%FFMPEG%;%PYTHON%;%PYTHON%\Scripts;%PATH%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%SCRIPT_DIR%src"

"%PYTHON%" -X utf8 src\gui.py

pause
