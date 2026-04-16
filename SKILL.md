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

**R8 — Both lines end with period:** Both lines are complete sentences ending with a period. Each line must be under 120 characters so the subtitle fits within 2 visual lines in the PowerPoint text box. If a sentence exceeds 120 characters, shorten it — never let a 2-line subtitle wrap to 3 lines.

**R9 — No false regional generalizations:** Never say "EM rallies" if only one EM index is up and another is down. Never say "DM sells off" if one DM market is flat. Any group claim (EM, DM, metals, majors) must be supported by ALL members of that group. When performance diverges within a group, name specific indices instead of using regional labels.

**R10 — MA position verification (CRITICAL):** When `vs_50d`, `vs_100d`, or `vs_200d` is **negative**, price is **BELOW** that MA — the MA is **overhead resistance**, not support. When **positive**, price is **ABOVE** — the MA is **support below**. Before writing any sentence about an MA, verify the sign of the corresponding `vs_Xd` field. A sentence like "the 100d just below current price" when `vs_100d` is negative is a sign inversion error and must be caught.

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

### MA Slope Awareness

When visible on the chart, mention whether MAs are:
- **Rising:** supportive, adds conviction to holds
- **Flattening:** trend losing steam
- **Curling down/rolling over:** trend reversing
- **Converging:** squeeze forming, breakout imminent

### Overview Subtitles (Equity / Commodity / Crypto)

Overview slides get **exactly 1 line, max 100 characters**. Write a broad macro narrative about the market environment — NOT individual instrument analysis.

**What to write:** Describe the regime, the trend, or the divergence:
- "Markets entered a risk-off environment as the two-week correction deepens across DM."
- "The April correction continues with most indices still below their 50d."
- "Broad commodity strength persists — energy and metals aligned for a rare unified rally."
- "Crypto remains in structural damage territory with no signs of stabilization."

**Hard rules:**
1. **ONE line only, max 100 characters.** If it wraps in PowerPoint, it's too long.
2. **Never mention DMAS, ratings (bullish/bearish/constructive), or scores.** These are individual-instrument concepts, not overview language.
3. **Never name more than one individual instrument.** Use categories instead (DM, EM, LatAm, energy, metals, altcoins).
4. **Max 1 number** (e.g., the YTD spread between best and worst).
5. **Frame the environment, not individual stories.** "Risk-off deepens" not "IBOV leads while Sensex lags."
6. **Never generalize a region when performance diverges within it.** If Brazil is +13% and India is -13%, say "LatAm defies a fractured equity market" not "EM rallies."

### Validation Clarifications (avoiding false positives)

- **R2 (max 2 numbers per line):** MA identifiers like "50d", "100d", "200d" are technical labels, NOT data values. Do not count them toward the 2-number limit.
- **R3 (no promotional adjectives):** Match whole words only, not substrings. "unremarkable" does not violate R3 just because it contains "remarkable."

### Banned Phrases (hard ban, never use)

"recovery hinges on", "recovery depends on", "outlook persists", "dynamics continue", "exceptional strength remains", "momentum supports further gains", "bullish streak remains unbroken", "maintains bullish trend", "resilience despite DMAS decline", "momentum strength key to", "hinges on reclaiming 50d MA", "momentum yet to confirm technical alignment", "remains trapped below all MAs", "as DMAS drops", "downgraded from X to Y", "score floored at zero", "breadth ranking worst at X/Y", "momentum score at X", "technical score at X"

### Before/After Examples

**Bearish with structural damage (DMAS 6):**
- ❌ "Bearish for 5 consecutive weeks with technical score floored at zero. Down 10.2% below 200d MA and RSI at 34—no reversal signal yet"
- ✅ "In freefall with all MAs now sloping down and price 10.2% below the 200d. RSI nearing oversold at 34 but no base forming — stabilization needed before any recovery attempt."

**Strong bullish (DMAS 91):**
- ❌ "Bullish for 14 consecutive weeks, trading 17.9% above 200d MA. Momentum score at 95 with breadth ranked 1/9"
- ✅ "All three MAs rising and well-spaced — textbook bullish structure. Pulled back to the 50d after 14 weeks of gains; RSI at 41 resets overbought conditions for the next leg."

**Constructive pullback (DMAS 67, Momentum 82):**
- ❌ "Downgraded from bullish to constructive despite trading 8.7% above 200d MA. RSI at 37 with price 5.0% under 50d MA—near-term rebound is key"
- ✅ "Pulled back below both the 50d and 100d, but the 200d remains well below near 48,500 preserving the longer-term uptrend. Needs to reclaim the 100d to restore bullish structure."

**Cautious with oversold bounce potential (DMAS 37):**
- ❌ "Downgraded to cautious as price slips 4.3% below 50d MA. RSI at 30 signals oversold—watch for a bounce near 200d MA"
- ✅ "Trading below all short-term MAs as the 50d begins curling lower. Oversold RSI near 30 favors a relief bounce — the 200d at 6,337 is the line in the sand."

**Violent correction from strength (DMAS 42, ↓38 WoW):**
- ❌ "Plunged 11.0% below 50d MA with RSI at oversold 25. Downgraded sharply from bullish—still 6.2% above 200d MA for now"
- ✅ "Violent correction through the 50d and 100d after a parabolic run. RSI at 25 is deeply oversold — the rising 200d near 4,200 is the first real support for a bounce attempt."

**Neutral consolidation (DMAS 45):**
- ❌ "Neutral for 3 consecutive weeks, hovering just 0.4% below 50d MA. Top fundamental rank at 1/9 but RSI at 65 suggests limited upside"
- ✅ "Consolidating just below the 50d with the 100d and 200d rising underneath as support. A close above the 50d would confirm the base — RSI at 65 caps near-term upside."

**MA position sign inversion (R10 violation):**
- Data: vs_50d = -5.0%, vs_100d = -3.2%, vs_200d = +8.7%
- ❌ "The 100d just below current price is the near-term pivot." (WRONG: vs_100d is negative → 100d is ABOVE price)
- ✅ "Corrected 5.0% below the 50d with the 100d also overhead — the 200d 8.7% below remains the structural floor."

**Overview (Equity):**
- ❌ "Brazil leads at +12% while India lags at -10.7% YTD"
- ✅ "Broad risk-off sweeps most equity markets — only IBOV holds ground as Sensex extends its decline."

### Generation Workflow

When reading `draft_state.json` to generate subtitles:

1. **Read all 20 instruments first** before writing any subtitle — batch awareness prevents repetition
2. For each instrument, extract: `rating`, `dmas`, `technical`, `momentum`, `rsi`, `vs_50d`, `vs_100d`, `vs_200d`, `streak_weeks`, `price`
3. **R10 CHECK:** For each MA field, note the sign: negative = price BELOW that MA (overhead resistance); positive = price ABOVE (support below). Write this down before drafting.
4. Determine the MA stack position (above/below each MA)
5. Select tone from DMAS calibration table
6. Identify the key MA level (the one price is testing or needs to reclaim)
7. Write Line 1 (current structure) and Line 2 (forward scenario)
8. **R10 CHECK:** Re-read every sentence that mentions an MA. Verify the direction word ("above"/"below"/"overhead"/"support"/"underpins") matches the sign from step 3. Fix any inversions.
9. Verify: max 2 data values per line, no banned phrases, no scores named, no instrument name first
10. After all 20 are written, scan for repetitive phrasing across the batch — rephrase any duplicates
11. Write the 3 overview subtitles last, using the full set of individual results for context

## Scheduled Task — Intern PC, Wednesday 07:15 Swiss

The IC2 weekly run is a Windows Task Scheduler entry on the intern PC
(`User3`) that fires every Wednesday 07:15 Swiss time. It invokes
Claude Code CLI in headless mode, which then executes the prompt below
using the Max-plan subscription. HTEI's 06:50 task has already opened
Bloomberg on the same PC, so `localhost:8194` is live.

- **WTS definition:** `tools/ic_wtscheduler_task.xml`
- **Batch launcher:** `tools/run_ic_cc.cmd` (what WTS actually calls)
- **Env file:** `C:\Users\User3\github\ic_technical\.env` (SMTP, paths)

### Prompt executed by the scheduler

```
Run the IC2 Market Compass pipeline — this is the fully-automated
Wednesday 07:15 run on the intern PC. Do not ask questions.

Context:
- Bloomberg is already logged in locally at localhost:8194 (HTEI 06:50 task).
- Price date = prior business day (Tuesday by default).
- Committee date = today (Wednesday) — UNLESS a file
  $IC_DROPBOX_PATH/.committee_override exists containing a YYYY-MM-DD
  date, in which case use that instead (typically a Thursday shift).
- This skill's SKILL.md describes the full pipeline.

Steps:
1. Resolve dates. Read .committee_override if present.
2. Run a Bloomberg pre-flight (start a blpapi session at
   localhost:8194). If it fails, invoke HTEI's login script at
   $HTEI_BLOOMBERG_LOGIN, wait up to 30 s, retry. If still down,
   call tools.ic_email.notify_failure("bloomberg", "...") and stop.
3. Run: python run_ic.py --stage prepare --date <price_date>
4. Read draft_state.json. Following config/subtitle_directive.md and
   the "Subtitle Generation Directive" section of this SKILL.md:
   a. Write 20 instrument subtitles into
      draft_state.json["instruments"][name]["subtitle"].
   b. Write 3 overview subtitles into draft_state.json["ytd_subtitles"]
      as a dict with keys "equity", "commodity", "crypto" — each a
      single-line string (these appear on the YTD performance slides).
   c. Save draft_state.json with encoding='utf-8' to avoid mojibake
      on Windows (em dashes etc.).
5. Run: python run_ic.py --stage assemble --date <price_date>
   --committee-date <committee_date>
6. Verify the PPTX exists at
   $IC_DROPBOX_PATH/Market_Compass_YYYYMMDD.pptx (committee date)
   and the first slide title, filename, and source lines use the
   correct dates (committee on title/filename, price date on source).
7. Call tools.ic_email.notify_success(pptx_path, committee_date,
   price_date). This sends the "IC generated" email to jcourtial.
8. If .committee_override was used, delete it.

Any failure at any step: call tools.ic_email.notify_failure(stage, err).
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
