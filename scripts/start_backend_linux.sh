#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (parent of scripts directory)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PORT="${PORT:-8001}"

echo "Project root: $ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python3 is not available in PATH. Please install Python 3.10+ and retry." >&2
  exit 1
fi

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# Activate venv
echo "Activating virtual environment"
source "$VENV_DIR/bin/activate"

# Install dependencies
REQS_FILE="$ROOT_DIR/requirements.txt"
if [[ ! -f "$REQS_FILE" ]]; then
  echo "requirements.txt not found at $REQS_FILE" >&2
  exit 1
fi
echo "Installing dependencies from $REQS_FILE"
pip install -r "$REQS_FILE"

# Start FastAPI via uvicorn (Yahoo backend)
echo "Starting Yahoo backend: uvicorn app.yahoo_server:app --host 0.0.0.0 --port $PORT"
exec uvicorn app.yahoo_server:app --host 0.0.0.0 --port "$PORT"