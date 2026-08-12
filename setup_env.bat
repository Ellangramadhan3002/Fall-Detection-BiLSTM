@echo off
echo ========================================================
echo   SETUP PYTHON VIRTUAL ENVIRONMENT (VENV)
echo ========================================================
echo.

echo [1/3] Membuat virtual environment (.venv)...
python -m venv .venv
if %errorlevel% neq 0 echo [ERROR] Gagal membuat virtual environment. Pastikan Python terinstall! && pause && exit /b

echo [2/3] Mengaktifkan virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 echo [ERROR] Gagal mengaktifkan virtual environment! && pause && exit /b

echo [3/3] Menginstall dependencies (TensorFlow, OpenCV, Mediapipe, dll)...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================================
echo   SELESAI!
echo ========================================================
echo Anda sekarang bisa membuka folder ini di VS Code.
echo Pastikan VS Code menggunakan interpreter Python dari folder '.venv'.
echo.
pause
