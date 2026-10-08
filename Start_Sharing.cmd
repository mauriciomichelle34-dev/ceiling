@echo off
chcp 65001 >nul
title Mycelium Temporary Sharing
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0share.py"
) else (
    python "%~dp0share.py"
)
pause
