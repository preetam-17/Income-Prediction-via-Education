@echo off
echo Testing Flask Installation...
echo.

echo Installing Flask...
pip install flask
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install Flask. Trying with python -m pip...
    python -m pip install flask
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install Flask. Please install manually.
        pause
        exit /b 1
    )
)

echo.
echo Starting test Flask server...
echo If successful, you should be able to access http://127.0.0.1:5000 in your browser
echo.

python test_flask.py
pause 