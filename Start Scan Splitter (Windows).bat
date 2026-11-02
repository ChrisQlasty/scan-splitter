@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Please install it from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" during installation.
  pause
  exit /b 1
)
if not exist ".venv" (
  echo First start: installing ^(takes 1-2 minutes^)...
  python -m venv .venv
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\pip install -q -r requirements.txt
)
.venv\Scripts\python app.py
pause
