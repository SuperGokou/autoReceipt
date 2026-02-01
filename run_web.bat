@echo off
REM Survey Bot Web UI Launcher
REM ==========================
REM
REM This script starts the Survey Bot web interface.
REM Open http://localhost:5000 in your browser after starting.

echo.
echo ========================================
echo    Survey Bot Web UI
echo ========================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Check if Flask is installed
python -c "import flask" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing Flask...
    pip install flask
)

REM Set environment variables (optional)
REM set FLASK_DEBUG=true
REM set PORT=5000

REM Start the web server
echo Starting web server on http://localhost:5000
echo Press Ctrl+C to stop.
echo.

python -m survey_bot.web.app

pause
