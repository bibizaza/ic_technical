# IC2 intern-PC entry points

Files that run on the intern PC under Windows Task Scheduler. Not used on the Mac.

## Files

- `run_ic_cc.cmd` — batch launcher. WTS calls this at Wed 07:15. It loads
  `.env` and invokes `claude -p --dangerously-skip-permissions "<prompt>"`.
  Claude (Max plan) executes the full IC2 skill: Bloomberg check + retry,
  prepare, subtitles, assemble, email.
- `ic_wtscheduler_task.xml` — importable Windows Task Scheduler definition
  (weekly Wed 07:15, runs `run_ic_cc.cmd` under User3).
- `ic_email.py` — SMTP helper. Claude imports `notify_success` /
  `notify_failure` from here during the run.

## Required environment (`.env` in repo root, not committed)

```
IC_DROPBOX_PATH=C:\Users\User3\Dropbox\Tools_In_Construction\ic
BLOOMBERG_HOST=localhost
BLOOMBERG_PORT=8194
HTEI_BLOOMBERG_LOGIN=C:\Users\User3\github\tactical_equity_spx\tools\bloomberg_login.py
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=financialquant86@gmail.com
SMTP_PASS=<same Gmail app password as HTEI>
RECIPIENT=jcourtial@herculis.ch
```

## One-time setup

1. Install Claude Code CLI (requires Node 20+):
   ```powershell
   npm install -g @anthropic-ai/claude-code
   ```
2. Authenticate once interactively so the Max-plan login is cached:
   ```powershell
   claude
   ```
   Log in via the browser, then `/exit`.
3. Verify headless mode:
   ```powershell
   claude -p --dangerously-skip-permissions "reply OK"
   ```
4. Import the WTS task:
   ```powershell
   schtasks /Create /XML tools\ic_wtscheduler_task.xml /TN "IC2 Market Compass" /RU User3 /RP <password>
   ```

## Thursday shift

When the IC moves to Thursday, drop a file named `.committee_override`
in `IC_DROPBOX_PATH` containing one line `YYYY-MM-DD` (the Thursday date).
Claude reads it, uses it as the committee date, and deletes it on success.

## Dry run

From the intern PC repo root:

```powershell
tools\run_ic_cc.cmd
```

Should generate the PPTX in Dropbox and email jcourtial. Inspect the Claude
transcript printed to stdout to debug any step that misbehaved.
