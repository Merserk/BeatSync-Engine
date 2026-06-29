@echo off
chcp 65001 >nul
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON=%SCRIPT_DIR%bin\python-3.13.14-embed-amd64\python.exe"
set "CUDA=%SCRIPT_DIR%bin\CUDA\v13.3"
set "PY_SITE=%SCRIPT_DIR%bin\python-3.13.14-embed-amd64\Lib\site-packages"
set "FFMPEG=%SCRIPT_DIR%bin\ffmpeg"

if not exist "%PYTHON%" (
    echo ERROR: Portable Python was not found.
    echo Run install.bat first.
    pause
    exit /b 1
)

set "PATH=%FFMPEG%;%PYTHON%;%PYTHON%\Scripts;%PATH%"
if /I "%BEATSYNC_FORCE_PORTABLE_CUDA%"=="1" if exist "%CUDA%" goto use_portable_cuda
if exist "%PY_SITE%\cuda_toolkit-*.dist-info" goto use_cupy_ctk
if exist "%CUDA%" goto use_portable_cuda
goto after_cuda_setup

:use_cupy_ctk
set "CUDA_PATH="
set "CUDA_HOME="
set "CUDA_ROOT="
goto after_cuda_setup

:use_portable_cuda
set "CUDA_PATH=%CUDA%"
set "CUDA_HOME=%CUDA%"
set "CUDA_ROOT=%CUDA%"
set "PATH=%CUDA%\bin\x64;%CUDA%\bin;%CUDA%\lib\x64;%PATH%"

:after_cuda_setup
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%SCRIPT_DIR%src"

"%PYTHON%" -X utf8 src\gui.py

pause
