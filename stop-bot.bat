@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bot.ps1" -Action Stop
if errorlevel 1 (
  echo.
  echo Bot stop failed. See the message above.
  pause
  exit /b 1
)
exit /b 0
