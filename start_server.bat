@echo off
echo Starting Flask server...
echo.
echo If Python is not found, please install Python from the Microsoft Store or from python.org
echo.
python -m flask run
echo.
echo If the above command failed, try:
echo.
py -m flask run
echo.
echo Or if you have the Flask command available:
echo.
flask run
echo.
pause 