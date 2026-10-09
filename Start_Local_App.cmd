@echo off
title Mycelium Local App
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0start.py"
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        python "%~dp0start.py"
    ) else (
        py -3 "%~dp0start.py"
    )
)
if errorlevel 1 pause
