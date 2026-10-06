@echo off
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title Wake-Word Spotter "BOT" (Alice Mode)

set "PYTHON_EXE=%~dp0.runtime\voice\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%~dp0word_training\app.py" %*
if errorlevel 1 (
    echo.
    echo Application stopped with an error.
    pause
    exit /b 1
)

exit /b 0