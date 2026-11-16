@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup_data.py
python train_model.py
echo.
echo Setup complete. Double-click run.bat to start MoodFlix.
pause
