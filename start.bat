@echo off
echo Starting Federated Fraud Detection Server...
echo.
echo Make sure you have installed dependencies:
echo pip install -r requirements.txt
echo.
echo Starting server on http://localhost:8000
echo Admin Console: http://localhost:8000/admin
echo Client Dashboard: http://localhost:8000/client
echo.
echo Press Ctrl+C to stop the server
echo.
python run.py
pause
