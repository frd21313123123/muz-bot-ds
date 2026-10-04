@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-bot.ps1" %*
set "setup_result=%errorlevel%"
if not "%setup_result%"=="0" echo Setup failed. See the message above.
pause
exit /b %setup_result%
