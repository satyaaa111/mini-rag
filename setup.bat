#!/bin/bash

# --- BACKEND SETUP ---
echo "Setting up Python Virtual Environment..."
cd backend
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo "Installing Backend Dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install Python dependencies."
    exit 1
fi

# --- FRONTEND SETUP ---
echo "Installing Frontend Dependencies..."
cd ../frontend
npm install
if [ $? -ne 0 ]; then
    echo "npm not found. Please install Node.js."
    exit 1
fi

# --- OLLAMA CHECK ---
echo "Checking Ollama for phi3 model..."
ollama pull phi3

echo "--------------------------------------------------------"
echo "SETUP COMPLETE"
echo "--------------------------------------------------------"
echo "To start the app:"
echo "1. Terminal 1: cd backend && source venv/bin/activate && python3 -m app.main"
echo "2. Terminal 2: cd frontend && npm run dev"
echo "--------------------------------------------------------"