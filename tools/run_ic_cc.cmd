@echo off
REM IC2 Market Compass — intern PC launcher.
REM Invoked by Windows Task Scheduler every Wednesday 07:15 Swiss.
REM Calls Claude Code CLI in headless mode; Claude executes the
REM ic2 skill (SKILL.md) end-to-end using the Max-plan subscription.

cd /d C:\Users\User3\github\ic_technical

REM Load environment variables from .env (SMTP, Dropbox, Bloomberg).
for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0..\.env") do (
    if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"
)

REM Hand off to Claude Code. The prompt tells Claude to follow the
REM "Scheduled Task" section of SKILL.md. --dangerously-skip-permissions
REM auto-approves tool calls (required for unattended runs).
claude -p --dangerously-skip-permissions "Run the IC2 weekly pipeline per SKILL.md. Today is Wednesday 07:15 on the intern PC — the fully-automated weekly run. Do not ask questions. Follow the 'Scheduled Task' section of SKILL.md exactly: resolve dates, pre-flight Bloomberg at localhost:8194 (retry via HTEI's bloomberg_login.py if down), run prepare, write subtitles into draft_state.json per config/subtitle_directive.md, run assemble with --committee-date = today, verify the PPTX, email jcourtial via tools.ic_email.notify_success. On any failure call tools.ic_email.notify_failure."

exit /b %errorlevel%
