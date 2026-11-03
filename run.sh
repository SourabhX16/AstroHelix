#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Prefer the project environment when it exists, but work with system Python too.
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
PYTHON_BIN=python
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

# Load env vars
if [ -f .env ]; then
  export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
fi

# Start Python agent backend
echo "Starting agent backend on :8000..."
"$PYTHON_BIN" -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend on :3000..."
cd frontend && npm run dev &
FRONTEND_PID=$!

cd ..

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
