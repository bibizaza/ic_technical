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

## Subtitle Generation Directive

### Context

You are generating 2-line subtitles for each instrument slide in the Herculis Market Compass weekly IC presentation. Each slide shows a 5-month price chart with 50d/100d/200d moving averages, RSI(14), and a DMAS scorecard (Technical, Momentum, RSI sub-scores). Your subtitle sits directly below the instrument name and rating — it is the analyst's voice.

### Core Philosophy

Write like a technical analyst briefing a committee, not like a data feed summarizing scores. Every subtitle should answer two questions:
1. **Line 1:** What is the chart structure telling us right now?
2. **Line 2:** What is the next technical level to watch, and what happens there?

### Hard Rules (violations = automatic reject)

**R1 — No scores in text:** Never mention DMAS, Technical score, Momentum score, RSI score value as a score, breadth rank, or fundamental rank by name. These are already displayed on the scorecard. Exception: RSI level can be referenced as a technical indicator (e.g., "RSI nearing oversold" or "RSI at 30"), but NOT as "RSI score at 30."

**R2 — Max 2 numbers per line:** Each subtitle line may contain at most 2 numerical values. Prioritize MA distance percentages and RSI over raw prices. If a line needs 3+ numbers, rephrase to drop the least important one.

**R3 — No promotional adjectives:** Banned: "extraordinary," "exceptional," "remarkable," "outstanding," "perfect," "relentless," "impressive," "stunning," "incredible," "powerful." Replace with factual technical language.

**R4 — No instrument name as first words:** The instrument name is already in the title. Don't start the subtitle with it.

**R5 — All numbers must be verifiable:** Every number in the subtitle must come from the data in `draft_state.json`. Never invent or approximate. If `vs_50d` is -4.3%, write -4.3%, not "about 4%."

**R6 — Streak counts from Python only:** Use the pre-computed `streak_weeks` field from `draft_state.json`. Never count entries in `history.json` yourself — you will hallucinate.

**R7 — No investment recommendations:** Never write "buy," "sell," "add exposure," "reduce position," "take profits," or similar advisory language.

**R8 — Both lines end with period:** Both lines are complete sentences ending with a period.

### Technical Narrative Framework

**The MA Stack — your primary structural tool.** Every subtitle must be aware of where price sits relative to the 50d, 100d, and 200d MAs:

| Price position | Structure | Language tone |
|---|---|---|
| Above all 3 MAs, MAs fanning up | Textbook bullish | "All MAs rising and well-spaced" |
| Above 200d, below 50d | Pullback within uptrend | "Corrected to the 100d zone but 200d underpins" |
| Below all 3 MAs, MAs curling down | Bearish breakdown | "Sliced through all MAs with the 50d curling lower" |
| Testing a specific MA | Inflection point | "Clinging to the 200d as last support" |
| Between MAs (trapped) | Indecision / range | "Trapped between the 100d and 200d" |

**Forward-looking language (Line 2):** Almost always include a forward scenario:
- "Needs to reclaim the 50d near X to restore bullish structure."
- "A break below the 200d would open a new leg lower."
- "The rising 200d near X is the first real support for a bounce."

**RSI as urgency/patience signal:** RSI contextualizes timing, not structure:
- RSI < 30: "Deeply oversold — bounce conditions forming"
- RSI 30-40: "RSI nearing oversold" or "limited downside cushion"
- RSI 40-60: No need to mention unless it adds context
- RSI 60-70: "RSI resetting from overbought" or "room to run"
- RSI > 70: "Overbought conditions building — pullback risk elevated"

### DMAS-Driven Tone Calibration

The DMAS score sets the emotional temperature WITHOUT being named:

| DMAS Range | Tone | Vocabulary palette |
|---|---|---|
| 80-100 | Confident, forward-looking | "textbook structure," "next leg," "well-spaced MAs," "resets for continuation" |
| 60-79 | Balanced, constructive | "broader uptrend intact," "underpins," "needs to reclaim," "would restore" |
| 40-59 | Cautious, watchful | "approaching support," "limited cushion," "line in the sand," "pivotal zone" |
| 20-39 | Urgent, defensive | "rapid succession," "losing levels," "no oversold cushion yet," "last support" |
| 0-19 | Structural damage | "freefall," "no base forming," "all MAs sloping down," "stabilization needed" |

**WoW delta modulates intensity:**
- Large negative (↓15+): "sharp reversal," "violent correction," "in rapid succession"
- Small negative (↓1-5): "continues to drift," "edging lower"
- Positive: "beginning to stabilize," "first uptick in X weeks"
- Unchanged: "consolidating," "range-bound," "holding its ground"

**Technical vs Momentum divergence (>30 points):**
- High Momentum + Low Technical: "The longer trend still favors buyers, but the recent breakdown is severe"
- Low Momentum + High Technical: "Near-term positioning has improved but the broader trend remains weak"

### Overview Subtitles (Equity / Commodity / Crypto)

Overview slides get exactly 1 line (not 2). Use a regional or thematic narrative:

1. Contrast strongest vs weakest with narrative tension, not just numbers
2. Name actual indices/assets (e.g., "IBOV," "Sensex," not "Brazil," "India")
3. Use a unifying theme if one exists (e.g., "EM divergence widens" or "Metals correct in unison")
4. Max 1 number (typically the spread between best and worst YTD)
5. No score references whatsoever
6. Frame the group story, not individual instrument stories
7. Do NOT name more than one individual instrument — rest should be categories
8. Numbers must match data within 1 percentage point, no approximate language

### Banned Phrases (hard ban, never use)

"recovery hinges on", "recovery depends on", "outlook persists", "dynamics continue", "exceptional strength remains", "momentum supports further gains", "bullish streak remains unbroken", "maintains bullish trend", "resilience despite DMAS decline", "momentum strength key to", "hinges on reclaiming 50d MA", "momentum yet to confirm technical alignment", "remains trapped below all MAs", "as DMAS drops", "downgraded from X to Y", "score floored at zero", "breadth ranking worst at X/Y", "momentum score at X", "technical score at X"

### Before/After Examples

**Bearish with structural damage (DMAS 6):**
- ❌ "Bearish for 5 consecutive weeks with technical score floored at zero. Down 10.2% below 200d MA and RSI at 34—no reversal signal yet"
- ✅ "In freefall with all MAs now sloping down and price 10.2% below the 200d. RSI nearing oversold at 34 but no base forming — stabilization needed before any recovery attempt."

**Strong bullish (DMAS 91):**
- ❌ "Bullish for 14 consecutive weeks, trading 17.9% above 200d MA. Momentum score at 95 with breadth ranked 1/9"
- ✅ "All three MAs rising and well-spaced — textbook bullish structure. Pulled back to the 50d after 14 weeks of gains; RSI at 41 resets overbought conditions for the next leg."

**Cautious with oversold bounce potential (DMAS 37):**
- ❌ "Downgraded to cautious as price slips 4.3% below 50d MA. RSI at 30 signals oversold—watch for a bounce near 200d MA"
- ✅ "Trading below all short-term MAs as the 50d begins curling lower. Oversold RSI near 30 favors a relief bounce — the 200d at 6,337 is the line in the sand."

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
