@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bot-control.ps1" -Action Stop
set "bot_exit_code=%errorlevel%"
echo.
pause
exit /b %bot_exit_code%
