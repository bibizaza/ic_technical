---
name: ic2
description: "IC2 (Market Compass) — automated weekly technical analysis presentation for Herculis Partners IC committee. Use this skill whenever the user mentions 'ic', 'market compass', 'run the ic', 'ic2', 'tuesday run', 'weekly presentation', 'technical update', or asks to generate/run the Investment Committee presentation. Also trigger for requests about DMAS scores, breadth analysis, or subtitle generation for the Market Compass. Requires Bloomberg running on the Parallels VM."
---

# IC2 — Market Compass Pipeline

## Overview

IC2 is the automated pipeline that generates the weekly Market Compass presentation for the Herculis Partners Investment Committee. It pulls data from Bloomberg, computes technical scores (DMAS, breadth, fundamentals) for 20 instruments across equity, commodity, and crypto, generates AI-powered subtitles, renders charts, and assembles a polished PowerPoint presentation.

**You are the pipeline operator.** You handle Bloomberg connectivity, data validation, subtitle generation, and presentation assembly. You verify every output before delivering.

## Data Locations

- **Repository:** `~/Desktop/GitHub/Projects/ic_technical/` (branch: `feature/pipeline-v2`)
- **Dropbox IC folder:** `~/Library/CloudStorage/Dropbox/Tools_In_Construction/ic/`
- **Output PPTX:** `{Dropbox IC folder}/Market_Compass_YYYYMMDD.pptx`
- **History:** `~/Desktop/GitHub/Projects/ic_technical/market_compass/data/history.json`
- **Breadth cache:** `~/Desktop/GitHub/Projects/ic_technical/breadth_cache.json`
- **Draft state:** `~/Desktop/GitHub/Projects/ic_technical/draft_state.json`
- **Conda env:** `ptf_opt`

## Bloomberg Connection

Bloomberg runs on the Parallels Windows VM. Connection via raw `blpapi` at `10.211.55.3:8194`.

**Pre-flight check — ALWAYS run this before any pipeline step:**

```python
conda run -n ptf_opt python -c "
import blpapi
opts = blpapi.SessionOptions()
opts.setServerHost('10.211.55.3')
opts.setServerPort(8194)
session = blpapi.Session(opts)
if session.start():
    print('✓ Bloomberg connected')
    session.stop()
else:
    print('✗ Bloomberg NOT reachable')
"
```

If Bloomberg is not reachable:
1. Send Telegram notification: "Bloomberg not reachable at 10.211.55.3:8194. Please check: (1) Parallels VM is running, (2) Bloomberg Terminal is logged in, (3) bbcomm.exe is running. Reply 'ready' when fixed."
2. Wait for user confirmation before proceeding.
3. Re-run the pre-flight check after confirmation.

## Date Logic — CRITICAL

The IC runs Tuesday night for the Wednesday committee. Date handling depends on WHEN the pipeline is triggered.

### Decision Tree

```
Step 1: What day/time is it?
│
├── Tuesday after 10 PM?
│   → NORMAL RUN (no questions needed)
│   → Prices: latest available (Tuesday close)
│   → Committee date: tomorrow (Wednesday)
│   → File: Market_Compass_YYYYMMDD.pptx (Wednesday date)
│   → Source lines: "Data as of DD/MM/YYYY" (Wednesday date)
│   → Proceed automatically.
│
└── Any other day/time?
    → EDGE CASE — ask questions before proceeding.
    │
    ├── Step 2: Ask price reference
    │   "It's currently [day, time]. Some markets may be open.
    │    Which prices should I use?"
    │
    │   Option A: Last close as of [previous trading day]
    │   Option B: Current last price (live snapshot — some markets may still be trading)
    │   Option C: Close as of a specific date → ask "Which date?"
    │
    ├── Step 3: Ask committee date
    │   "What date should appear on the presentation and filename?"
    │
    │   Option A: Today ([current date])
    │   Option B: Specific date → ask "Which date?"
    │
    └── Step 4: Confirm before running
        "Running IC2 pipeline:
         • Prices: [selection] ([date])
         • Committee date: [date]
         • File: Market_Compass_YYYYMMDD.pptx
         Proceed?"
```

### Date implementation

```python
from datetime import date, timedelta

def resolve_dates(price_mode, price_date, committee_date):
    """
    Returns (price_reference_date, committee_display_date)

    price_mode: 'close' | 'last'
    price_date: date object (used only if price_mode='close')
    committee_date: date object for slide title + filename
    """
    pass
```

The committee date affects:
- Slide 1 title: "March 21, 2026"
- Filename: `Market_Compass_20260321.pptx`
- All source lines: "Data as of 21/03/2026"

The price date affects:
- Bloomberg data pull reference date
- Does NOT appear on the slides (prices are always "latest available")

### Important: committee_date ≠ price_date

Normal Tuesday 11 PM run: prices = Tuesday close, committee_date = Wednesday.
The slides never say "prices as of Tuesday" — they say the committee date.

## Pipeline Stages

### Stage 1: Prepare (`--stage prepare`)

```bash
conda run -n ptf_opt python run_ic.py --stage prepare --date YYYY-MM-DD
```

What it does:
1. Bloomberg pull: 103 tickers (close, low, high) → appends to `master_prices.csv`
2. Bloomberg pull: 8 breadth fields × 9 equity indices
3. Bloomberg pull: 9 fundamental fields × 9 equity indices
4. Compute DMAS scores (Technical + Momentum) for 20 instruments
5. Compute breadth composite (Trend 40%, Conviction 35%, Sentiment 25%)
6. Compute fundamental ranks (cross-sectional, 6 pillars)
7. Render 20 technical charts (HTML → Playwright → PNG at 4x)
8. Compute streak counts from history.json (ISO week grouping)
9. Write `draft_state.json` with all scores, ranks, chart paths, streak counts

The `--date` flag is the price reference date, NOT the committee date.

### Stage 2: Assemble (`--stage assemble`)

```bash
conda run -n ptf_opt python run_ic.py --stage assemble
```

What it does:
1. Read `draft_state.json`
2. Load previous week data from `history.json` for WoW deltas
3. Generate subtitles via Claude API (20 instruments + 3 overviews)
4. Render performance charts (equity, commodity, crypto, FX, rates, credit)
5. Render breadth table, fundamental table, quadrant chart
6. Assemble PPTX from template
7. Copy to Dropbox IC folder
8. Update `history.json` with current week data
9. Send Telegram notification with summary

### Stage 3: Full (`--stage full`)

Runs prepare → assemble in sequence. This is what the scheduled task uses.

## Subtitle Rules

The subtitle prompt has strict rules. If editing the prompt, preserve ALL of these:

1. **No DMAS score in subtitle text** — the number is on the gauge chart already
2. **Max 2 numbers per subtitle line** — prioritize MA distances and RSI over raw scores
3. **No promotional adjectives** — no "exceptional", "remarkable", "outstanding", "perfect", "impressive"
4. **Overview subtitles: punchy, narrative** — contrast best vs worst performer
5. **Streak counts: use `streak_weeks` from draft_state.json** — NEVER count history.json entries manually
6. **Line 1 must end with period** — post-processing enforced
7. **No instrument name as first word**
8. **Numbers must be verifiable from the data**
9. **MA position language must match actual vs_50d/vs_100d/vs_200d values**
10. **No investment recommendations**

## Scheduled Task — Tuesday 11:00 PM

The scheduled task prompt:

```
Run the IC2 Market Compass pipeline. Today is Tuesday after US close — this is
the normal weekly run.

1. Run Bloomberg pre-flight check. If it fails, send Telegram notification and
   wait for my reply.
2. If Bloomberg OK, run: python run_ic.py --stage full
3. Use latest prices (Tuesday close).
4. Committee date = tomorrow (Wednesday). Apply to slide 1 title, filename,
   and all source lines.
5. After completion, verify the PPTX:
   - Confirm 20 instrument slides have subtitles
   - Confirm nutshell table scores match individual slides
   - Confirm overview subtitles are single-line
6. Copy PPTX to Dropbox IC folder.
7. Send Telegram summary:
   - "IC2 complete. [N] instruments scored."
   - List any rating changes vs last week
   - List any instruments with DMAS > 90 or < 10
   - Attach PNG of slide 1 and nutshell table
8. Do not ask questions. This is fully automated.
```

### Mac sleep prevention

The scheduled task should run this before the pipeline:

```bash
# Keep Mac awake until midnight
caffeinate -u -t 7200 &
CAFFEINATE_PID=$!
```

And after completion:

```bash
# Release caffeinate
kill $CAFFEINATE_PID 2>/dev/null
```

## Telegram Integration

### Setup (one-time)

1. Create bot via @BotFather → save token
2. Get chat_id by messaging the bot and checking `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Set environment variables:
   ```bash
   export TELEGRAM_BOT_TOKEN="your-token"
   export TELEGRAM_CHAT_ID="your-chat-id"
   ```

### Send function

```python
import requests, os

def send_telegram(message: str, image_path: str = None):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    if image_path:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        with open(image_path, "rb") as f:
            requests.post(url, data={"chat_id": chat_id, "caption": message},
                         files={"photo": f})
    else:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message,
                                  "parse_mode": "Markdown"})
```

### Notification triggers

| Event | Message |
|---|---|
| Bloomberg check failed | "⚠️ Bloomberg not reachable. Check Parallels + Terminal. Reply 'ready'." |
| Pipeline started | "🔄 IC2 pipeline started. Pulling Bloomberg data..." |
| Pipeline complete | "✅ IC2 complete. [summary]. PPTX in Dropbox." |
| Pipeline error | "❌ IC2 failed at [stage]: [error message]" |
| Rating changes | "📊 Rating changes: MEXBOL ↓ Bullish→Constructive, TASI ↑ Bearish→Neutral" |

## Verification Checklist

After every run, before sending the "complete" notification:

- [ ] All 20 instrument slides have subtitles (not empty, not placeholder)
- [ ] Subtitle line 1 ends with period for all instruments
- [ ] Overview subtitles are exactly 1 line each
- [ ] DMAS scores on nutshell table match individual slide gauges
- [ ] Rating on slide title matches nutshell table outlook
- [ ] WoW deltas on gauges are non-zero where expected
- [ ] Sub-score arrows show ▲/▼ not dashes (except unchanged)
- [ ] Breadth table shows 9 indices ranked 1–9
- [ ] Fundamental table shows 9 indices ranked 1–9
- [ ] Quadrant chart shows 9 dots with correct positions
- [ ] Committee date appears correctly on slide 1 and source lines
- [ ] Filename uses committee date, not run date
- [ ] PPTX file exists in Dropbox IC folder

## Manual Run Workflow

When the user says "run the ic" or similar outside of the scheduled task:

1. Run Bloomberg pre-flight check
2. Detect current day/time
3. Follow the Decision Tree (see Date Logic section above)
4. Ask necessary questions and confirm
5. Run the pipeline with resolved parameters
6. Verify and notify

## History & Streak Computation

`history.json` stores weekly snapshots. Streaks are computed in Python using ISO week grouping:

1. Group entries by ISO week (isocalendar)
2. Take latest entry per week
3. Get current rating
4. Walk backwards counting consecutive weeks with same rating
5. Store as `streak_weeks` in draft_state.json

NEVER let the LLM count entries in history.json — it will hallucinate.

## Key Principles

- **Score alignment is prerequisite to everything** — if scores don't match production, stop and fix
- **Verify by opening the actual PPTX** — never self-report success without checking
- **Fix one issue at a time** — re-run pipeline after each fix
- **Read actual code before editing** — don't guess what a function does
- **xbbg has bugs with remote connections** — always use raw `blpapi` via `10.211.55.3:8194`
- **Context window fills fast** — start new chats for long Claude Code sessions
