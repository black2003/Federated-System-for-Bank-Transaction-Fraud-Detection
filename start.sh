#!/bin/bash

echo "Starting Federated Fraud Detection Server..."
echo ""
echo "Make sure you have installed dependencies:"
echo "pip install -r requirements.txt"
echo ""
echo "Starting server on http://localhost:8000"
echo "Admin Console: http://localhost:8000/admin"
echo "Client Dashboard: http://localhost:8000/client"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

python run.py
