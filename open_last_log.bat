@echo off
setlocal enabledelayedexpansion

REM ---------- determine script dir and logs dir ----------
REM %~dp0 = directory of this script (guaranteed to include trailing backslash)
set "SCRIPT_DIR=%~dp0"
REM remove trailing backslash if present
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "LOGDIR=%SCRIPT_DIR%\logs"

echo Script directory: "%SCRIPT_DIR%"
echo Logs directory:   "%LOGDIR%"
echo.

REM ---------- make sure logs folder exists ----------
if not exist "%LOGDIR%" (
  echo Logs directory not found: "%LOGDIR%"
  set /p ans=Create it now? (Y/N) :
  if /I "%ans%"=="Y" (
    mkdir "%LOGDIR%"
    if errorlevel 1 (
      echo Failed to create "%LOGDIR%". Aborting.
      goto :eof
    ) else (
      echo Created "%LOGDIR%".
    )
  ) else (
    echo Aborting — logs directory missing.
    goto :eof
  )
)

REM ---------- open newest .log file ----------
for /f "delims=" %%i in ('dir "%LOGDIR%\*.log" /b /a:-d /o:-d 2^>nul') do (
  echo Opening "%LOGDIR%\%%i"
  REM use start "" to avoid issues with quoted paths being interpreted as title
  start "" notepad "%LOGDIR%\%%i"
  goto :done
)

echo No .log files found in "%LOGDIR%".
:done
endlocal
