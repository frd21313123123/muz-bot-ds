@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bot.ps1" -Action Start
if errorlevel 1 (
  echo.
  echo Bot start failed. See the message above and .runtime logs.
  pause
  exit /b 1
)
exit /b 0
