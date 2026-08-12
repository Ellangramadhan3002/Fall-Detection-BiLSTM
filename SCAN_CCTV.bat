@echo off
title CCTV IP Scanner - Sistem Deteksi Jatuh
color 0A
echo ============================================================
echo    CCTV IP SCANNER - Sistem Deteksi Jatuh BiLSTM
echo    Ellang Ramadhan - Politeknik Negeri Malang 2026
echo ============================================================
echo.
echo Mengaktifkan Virtual Environment...
call env\Scripts\activate.bat
echo.
echo Menjalankan Scanner CCTV...
python scan_cctv.py
echo.
pause
