@echo off

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies
pip install -r requirements.txt

:: Create necessary directories
mkdir models 2>nul
mkdir benchmark_results 2>nul
mkdir test_results 2>nul

echo Installation completed! Use the following command to activate the virtual environment:
echo call venv\Scripts\activate.bat 