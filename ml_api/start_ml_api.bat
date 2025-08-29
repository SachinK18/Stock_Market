@echo off
echo Installing Python dependencies...
pip install -r requirements.txt

echo Starting ML API server...
python app.py

pause
