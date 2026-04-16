@echo off
REM IC2 Market Compass — intern PC launcher.
REM Invoked by Windows Task Scheduler every Wednesday 07:15 Swiss.
REM Calls Claude Code CLI in headless mode; Claude executes the
REM ic2 skill (SKILL.md) end-to-end using the Max-plan subscription.
setlocal ENABLEEXTENSIONS

cd /d "%~dp0.."

REM Force UTF-8 for Python stdout so emoji in print() don't crash on CP1252.
set PYTHONIOENCODING=utf-8

REM Load environment variables from .env (SMTP, Dropbox, Bloomberg).
REM Expected format: KEY=VALUE per line. No comments, no blank lines.
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

REM Hand off to Claude Code. The prompt tells Claude to follow the
REM "Scheduled Task" section of SKILL.md. --dangerously-skip-permissions
REM auto-approves tool calls (required for unattended runs).
claude -p --dangerously-skip-permissions "Run the IC2 weekly pipeline per SKILL.md. This is the fully-automated weekly run on the intern PC. Do not ask questions. Follow the Scheduled Task section of SKILL.md exactly: resolve dates, pre-flight Bloomberg at localhost:8194 (retry via HTEI bloomberg_login.py if down), run prepare, write ALL subtitles into draft_state.json (20 instrument subtitles + 3 YTD overview subtitles under ytd_subtitles with keys equity/commodity/crypto — MUST save JSON as UTF-8), run assemble with --committee-date = today, verify PPTX, email jcourtial via tools.ic_email.notify_success. On any failure call tools.ic_email.notify_failure."

set "RC=%errorlevel%"
endlocal & exit /b %RC%
