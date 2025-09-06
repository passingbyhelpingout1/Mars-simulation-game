@echo off
setlocal
REM Run Mars Colony (Windows). Put this file next to mars_colony.py and double-click.

REM Change to this script's directory
cd /d "%~dp0"

REM Prefer the Python launcher if available
where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PYTHON=python"
  ) else (
    echo Python 3 not found. Please install it from https://www.python.org/downloads/ and try again.
    pause
    exit /b 1
  )
)

set "ARGS="
if exist savegame.json set "ARGS=--load"

%PYTHON% mars_colony.py %ARGS%

echo.
pause
