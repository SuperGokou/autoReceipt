@echo off
REM Run Survey Bot with conda tf environment
echo Activating conda tf environment...
call D:\Study\anaconda3\Scripts\activate.bat tf

echo Starting Survey Bot Web UI...
cd /d "J:\Project Files\MyPython\autoReceipt"
python run_web.py

pause
