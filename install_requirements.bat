@echo off
echo Installing Smart Resume Screener CLI Requirements...
echo.
echo This will install all necessary Python packages.
echo.
pause

echo Installing packages from src/requirements.txt...
cd src
pip install -r requirements.txt

echo.
echo Installation complete!
echo.
echo To run the program:
echo 1. Create a 'resumelist' folder and add your resume PDFs
echo 2. Double-click 'run_screener.bat' or run 'python src/cli_resume_screener.py'
echo.
pause 