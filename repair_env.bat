@echo off
chcp 65001 >nul
setlocal

:: IMPORTANT:
:: 1) Please save this .bat file as UTF-8 with BOM for proper Chinese display in CMD.
:: 2) This script fixes NumPy/Numba compatibility issues in portable Python.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON=%SCRIPT_DIR%bin\python-3.13.9-embed-amd64\python.exe"

if not exist "%PYTHON%" (
  echo [错误] 未找到便携 Python：%PYTHON%
  pause
  exit /b 1
)

echo [信息] 正在修复依赖版本（NumPy/Numba）...
"%PYTHON%" -X utf8 dependency_guard.py --fix

if %errorlevel% neq 0 (
  echo [错误] 依赖修复失败，请检查网络代理或 pip 源设置。
  pause
  exit /b 1
)

echo [完成] 依赖修复成功。你现在可以运行 run.bat。
pause
