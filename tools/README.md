# IC2 intern-PC entry points

Scripts that run on the intern PC under Windows Task Scheduler. They are
not used on the Mac.

## Files

- `run_ic_pc.py` — full weekly run. Checks Bloomberg, retries once via
  HTEI's `bloomberg_login.py` if down, runs `run_ic.py --stage full`,
  emails jcourtial on success or any failure.
- `ic_email.py` — SMTP helper. Reads `SMTP_*` + `RECIPIENT` from env.

## Required environment (`.env` on the intern PC, not committed)

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

## Windows Task Scheduler

Create one task that fires every Wednesday at 07:15 (Swiss time), with:

- **Program/script:** `python.exe` from the `ptf_opt` environment
- **Arguments:** `C:\Users\User3\github\ic_technical\tools\run_ic_pc.py`
- **Start in:** `C:\Users\User3\github\ic_technical`
- **Run whether user is logged on or not** (so it runs even if no one is at the console)
- Import `ic_wtscheduler_task.xml` if present for a ready-made definition.

## Thursday shift

When the IC moves to Thursday, drop a file named `.committee_override`
into the IC Dropbox folder (`IC_DROPBOX_PATH`) containing one line with
the Thursday date in `YYYY-MM-DD` format. The wrapper reads it, uses
that as the committee date, and deletes the file on success.
