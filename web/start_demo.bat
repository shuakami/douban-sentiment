@echo off
chcp 65001 >nul
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
if not defined HF_HOME set HF_HOME=%USERPROFILE%\.cache\huggingface
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
