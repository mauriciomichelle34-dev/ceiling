@echo off
title Mycelium Local App
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0local_app.py" --open
) else (
    python "%~dp0local_app.py" --open
)
if errorlevel 1 pause
