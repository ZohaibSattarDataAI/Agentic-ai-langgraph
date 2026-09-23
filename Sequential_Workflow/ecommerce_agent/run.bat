@echo off
call venv\Scripts\activate
echo Ollama check...
curl -s http://localhost:11434 >nul || (echo Ollama chal nahi raha. Naye CMD mein "ollama serve" chalayen. & pause & exit /b 1)
python whatsapp_bot.py
pause
