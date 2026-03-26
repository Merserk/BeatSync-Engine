@echo off
chcp 65001 >nul
setlocal

:: IMPORTANT: Save this file as UTF-8 with BOM if you edit it with Chinese text.

:: Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Set paths
set "PYTHON=%SCRIPT_DIR%bin\python-3.13.9-embed-amd64\python.exe"

if not exist "%PYTHON%" (
  echo [Error] Portable Python not found: %PYTHON%
set "CUDA=%SCRIPT_DIR%bin\CUDA\v13.0"

if not exist "%PYTHON%" (
  echo [Error] Portable Python not found: %PYTHON%
  echo Please check whether the package is fully extracted.
  pause
  exit /b 1
)

:: Environment
set "CUDA_PATH=%CUDA%"
set "CUDA_HOME=%CUDA%"
set "PATH=%CUDA%\bin\x64;%CUDA%\lib\x64;%PATH%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"

:: Auto-check and repair NumPy/Numba compatibility before launch
:: Auto-check and repair NumPy/Numba compatibility
"%PYTHON%" -X utf8 dependency_guard.py --check
if %errorlevel% neq 0 (
  echo [Warning] Detected NumPy/Numba compatibility issue. Trying auto-repair...
  "%PYTHON%" -X utf8 dependency_guard.py --fix
  if %errorlevel% neq 0 (
    echo [Error] Auto-repair failed. Please run repair_env.bat manually.
    echo [Error] Auto-repair failed.
    echo Please run repair_env.bat manually after checking proxy/network settings.
    pause
    exit /b 1
  )
)

:: Run
echo Starting Music Video Cutter...
echo Starting BeatSync Engine...
"%PYTHON%" -X utf8 gui.py

pause
