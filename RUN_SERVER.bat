@echo off
title CCTV Fall Detection Server
echo ===========================================
echo    Memulai Server dengan Virtual Environment
echo ===========================================

REM Selalu gunakan Python dari venv, bukan Python sistem
set PYTHON="%~dp0.venv\Scripts\python.exe"

REM Cek apakah venv tersedia
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment tidak ditemukan!
    echo Jalankan: python -m venv .venv
    echo Lalu:     .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [OK] Menggunakan Python: %PYTHON%
echo [OK] Membuka browser di http://localhost:8000 ...
start "" "http://localhost:8000"

cd /d "%~dp0"
%PYTHON% 05_api_server.py

pause
