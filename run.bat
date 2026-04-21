@echo off
REM ============================================================
REM  Driving Behavior Pattern Analysis — Run Script
REM ============================================================
echo.
echo  ======================================================
echo   Driving Behavior Pattern Analysis
echo  ======================================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    echo      Done.
) else (
    echo [1/4] Virtual environment already exists.
)

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [3/4] Installing dependencies...
pip install -r requirements.txt -q

REM Check if pipeline has been run
if not exist "models\best_model.joblib" (
    echo.
    echo [!] Model not found. Running ML pipeline first...
    echo     This may take 1-2 minutes...
    echo.
    python src\run_pipeline.py
    echo.
)

REM Launch dashboard
echo [4/4] Launching Streamlit Dashboard...
echo.
echo  Open your browser at: http://localhost:8501
echo  Press Ctrl+C to stop the server.
echo.
streamlit run dashboard\app.py --server.headless true
