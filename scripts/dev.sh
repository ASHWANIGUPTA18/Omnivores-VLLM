#!/usr/bin/env bash
# Start the backend and frontend for local development.
#
# Creates the virtualenv and installs dependencies on first run, so a fresh
# clone needs one command. Both processes stop together on Ctrl-C.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

# --- backend ---------------------------------------------------------------

if [[ ! -d backend/.venv ]]; then
  echo "==> creating backend virtualenv"
  "$PYTHON" -m venv backend/.venv
fi

VENV_BIN="backend/.venv/bin"
[[ -d "$VENV_BIN" ]] || VENV_BIN="backend/.venv/Scripts"   # Windows layout

echo "==> installing backend dependencies"
"$VENV_BIN/python" -m pip install --quiet --upgrade pip
"$VENV_BIN/python" -m pip install --quiet -r backend/requirements-dev.txt

if [[ ! -f backend/.env ]]; then
  echo "==> creating backend/.env from the example (mock mode, no GPU needed)"
  cp backend/.env.example backend/.env
fi

# --- frontend --------------------------------------------------------------

if [[ ! -d frontend/node_modules ]]; then
  echo "==> installing frontend dependencies"
  (cd frontend && npm install)
fi

# --- run -------------------------------------------------------------------

cleanup() {
  echo
  echo "==> stopping"
  # Kill the whole process group so uvicorn's reloader children go too.
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> backend  : http://localhost:8000  (API docs at /docs)"
(cd backend && "../$VENV_BIN/python" -m uvicorn app.main:app --reload --port 8000) &

echo "==> frontend : http://localhost:5173"
(cd frontend && npm run dev) &

wait
