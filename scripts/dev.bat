@echo off
REM Start the backend and frontend for local development on Windows.
REM
REM Opens each server in its own window so their logs stay readable.
REM Close either window to stop that server.

setlocal

set ROOT=%~dp0..
cd /d "%ROOT%"

if not exist backend\.venv (
  echo ==^> creating backend virtualenv
  python -m venv backend\.venv
)

echo ==^> installing backend dependencies
backend\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
backend\.venv\Scripts\python.exe -m pip install --quiet -r backend\requirements-dev.txt

if not exist backend\.env (
  echo ==^> creating backend\.env from the example ^(mock mode, no GPU needed^)
  copy backend\.env.example backend\.env >nul
)

if not exist frontend\node_modules (
  echo ==^> installing frontend dependencies
  pushd frontend
  call npm install
  popd
)

echo ==^> backend  : http://localhost:8000  ^(API docs at /docs^)
start "Omnivores backend" cmd /k "cd /d %ROOT%\backend && ..\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"

echo ==^> frontend : http://localhost:5173
start "Omnivores frontend" cmd /k "cd /d %ROOT%\frontend && npm run dev"

echo.
echo Both servers are starting in separate windows.
echo Open http://localhost:5173 once the frontend is ready.

endlocal
