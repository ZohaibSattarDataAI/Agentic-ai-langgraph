@echo off
echo === Step 1: Python check ===
python --version || (echo Python install nahi hai. python.org se install karein. & pause & exit /b 1)

echo === Step 2: Virtual environment ===
if not exist venv python -m venv venv
call venv\Scripts\activate

echo === Step 3: Packages ===
python -m pip install --upgrade pip
pip install -r requirements.txt

echo === Step 4: Ollama model check ===
ollama list | findstr /C:"qwen2.5:1.5b" >nul || ollama pull qwen2.5:1.5b

echo.
echo Setup mukammal. Ab .env file kholein aur ALLOWED_CHATS mein apna test contact likhein.
echo Phir run.bat chalayen.
pause
