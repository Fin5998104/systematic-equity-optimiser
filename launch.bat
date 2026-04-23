@echo off
REM Portfolio Optimiser Launcher
REM Double-click this file to start the app

echo Starting Portfolio Optimiser...
echo.

REM Try to find Anaconda Python
set ANACONDA_PYTHON=%USERPROFILE%\anaconda3\python.exe
if exist "%ANACONDA_PYTHON%" goto :found

set ANACONDA_PYTHON=%USERPROFILE%\Anaconda3\python.exe
if exist "%ANACONDA_PYTHON%" goto :found

set ANACONDA_PYTHON=C:\ProgramData\Anaconda3\python.exe
if exist "%ANACONDA_PYTHON%" goto :found

REM Fall back to system Python
set ANACONDA_PYTHON=python

:found
echo Using Python: %ANACONDA_PYTHON%
echo.

REM Install dependencies (only first run)
echo Checking dependencies...
"%ANACONDA_PYTHON%" -m pip install --quiet flask yfinance scipy scikit-learn pandas pandas-datareader requests

REM Open browser after a short delay
start "" cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:5000"

REM Run the app
echo.
echo Opening browser at http://localhost:5000
echo.
echo *** Keep this window open while using the app ***
echo *** Close this window to stop the app ***
echo.
"%ANACONDA_PYTHON%" "%~dp0app.py"

pause
