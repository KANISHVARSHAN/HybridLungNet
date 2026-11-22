@echo off
rem Builds the UI (first run only) and starts HybridLungNet v2 on http://127.0.0.1:8000
rem Uses a .venv next to the repo root if present, otherwise "python" from PATH.
set "ROOT=%~dp0..\.."
set "PY=python"
if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
cd /d "%~dp0frontend"
if not exist dist\index.html (
  call npm install
  call npx vite build
)
cd /d "%~dp0backend"
start "" http://127.0.0.1:8000
"%PY%" -m uvicorn main:app --host 127.0.0.1 --port 8000
