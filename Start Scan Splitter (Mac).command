#!/bin/bash
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python is not installed. Please install it from https://www.python.org/downloads/ and try again."
  read -p "Press Enter to close..."; exit 1
fi
if [ ! -d ".venv" ]; then
  echo "First start: installing (takes 1-2 minutes)..."
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python app.py
