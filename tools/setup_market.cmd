@echo off
REM One-time Market setup on the intern PC (tools/README.md step 5). From the repo root, after git pull:
REM   tools\setup_market.cmd C:\path\to\intern-market.env
REM Adds the file's two MARKET_PUBLISHER_* lines to .env, deletes the file, installs PyMuPDF, runs the selftest.
setlocal ENABLEEXTENSIONS
cd /d "%~dp0.."
if "%~1"=="" (echo Give the path of intern-market.env & exit /b 1)
if not exist "%~1" (echo Not found: %~1 & exit /b 1)
findstr /b /c:"MARKET_PUBLISHER_EMAIL=" ".env" >nul 2>&1
if errorlevel 1 (
    (echo.)>> ".env"
    type "%~1" >> ".env"
    echo Market login added to .env.
) else (
    echo .env already has a Market login: kept.
)
del "%~1"
echo intern-market.env deleted.
python -m pip install -q pymupdf
for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" set "%%A=%%B"
)
python tools\market_publish.py selftest
endlocal
