#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

if [ ! -f ".env" ]; then
  echo "Creating .env from .env.example..."
  cp .env.example .env
  echo "You can set GEMINI_API_KEY later in the Web UI settings."
fi

echo "Installing/updating dependencies..."
.venv/bin/python -m pip install -r requirements.txt

echo ""
echo "Starting Music Classifier..."
echo "Open http://localhost:8080 if the browser does not open automatically."
echo ""

.venv/bin/python app.py
