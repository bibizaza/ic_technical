# IC Pipeline — Market Compass Generation

## When user says "run ic [date]"

This skill orchestrates the full IC pipeline for a given date.
The working directory must be the ic repo root.

---

## Step 1: Prepare (Bloomberg → Scores → Charts → draft_state.json)

```bash
cd /Users/larazanella/Desktop/GitHub/Projects/ic_technical
python run_ic.py --stage prepare --date [DATE]
```

This will:
- Pull new prices from Bloomberg (10.211.55.3:8194) for all 103 tickers
- Append new rows to master_prices.csv
- Compute DMAS, technical, momentum, RSI scores for 20 instruments
- Compute composite breadth scores for 9 indices
- Render 20 technical charts to charts_cache/
- Save draft_state.json with all data (subtitles = null)

**Prerequisites:** Bloomberg Terminal running in Parallels VM.

---

## Step 2: Generate Subtitles (Claude Code reads draft_state.json)

Read draft_state.json. For each of the 20 instruments:
1. Read the `enriched_context` field
2. Generate a 2-line subtitle

Write all subtitles back to draft_state.json: set `instruments.[name].subtitle = "Line 1\nLine 2"`.

### Subtitle Rules
- **2 lines**, max 12 words each
- **Line 1 ends with a period.** Line 2 has no period at end
- **Never** start with the instrument name
- Each subtitle must be **uniquely different** across the full 20-instrument batch
- All numbers from the data only — no invented numbers
- Directional claims must match scores (bullish language only when DMAS >= 60)
- **Line 1:** most important fact RIGHT NOW
- **Line 2:** what to watch next / key level or catalyst

### MA Position Rule: Use Exact Data, Never Approximate
When describing price position relative to a moving average, use the exact vs_50d, vs_100d, vs_200d percentages from draft_state.json. Negative = below, positive = above.

- **Never** say "hugging", "near", or "testing" unless the value is between -0.5% and +0.5%
- A stock at -1.3% below the 50d MA is **clearly below it**, not near it — say "X% below the 50d MA"
- A stock at -0.3% is legitimately "right at" or "testing" the 50d MA
- A stock at +0.1% is "right on" or "at" the 50d MA
- Values beyond ±0.5% require directional language: "below", "above", "X% below", "X% above"

### Style Rule: Analyst Voice, Not Data Dump
Each subtitle should contain **AT MOST 2 numbers**. Pick the ONE or TWO most meaningful data points and weave them into a narrative. The rest should be qualitative interpretation.

The tone should be a **senior portfolio manager briefing a client**, not a quant reading a spreadsheet.

**BAD (data dump):**
> "Slipped from DMAS 78 to 45 as price sits 1.6% below 50d MA
> RSI at 42 with momentum at 57 suggests directionless drift ahead"

**GOOD (analyst voice with selective data):**
> "Sharp DMAS deterioration flags a regime shift from bullish to neutral
> Sitting just below the 50d MA — next week decides direction"

**BAD (data dump):**
> "Achieved a flawless DMAS of 100 with price 27.6% above the 200d MA
> RSI at 61 still has room before overheating — 50d MA gap of 6.7%"

**GOOD (analyst voice):**
> "Flawless technical score crowns gold as the standout bullish asset
> Rally has room to run before RSI overheats — no reversal signal yet"

### Banned phrases (never use, including partial matches)
- "recovery hinges on" / "hinges on"
- "recovery depends on"
- "outlook persists"
- "dynamics continue"
- "exceptional strength remains"
- "momentum supports further gains"
- "bullish streak remains unbroken"
- "maintains bullish trend"
- "resilience despite DMAS decline"
- "momentum strength key to"
- "hinges on reclaiming 50d MA"
- "momentum yet to confirm technical alignment"
- "remains trapped below all MAs"
- "as DMAS drops"
- "remains constructive" / "remains cautious" / "remains neutral"

### Short-form aliases (also banned)
- "streak unbroken" → banned
- "exceptional strength" → banned
- "further gains" → banned

### Quality check
After writing all 20 subtitles, scan for:
1. Any duplicate phrases across instruments (fix if found)
2. Any banned phrase violations (fix if found)
3. Any subtitle where tone contradicts the DMAS score (fix if found)

---

## Step 3: Assemble (draft_state.json → PowerPoint)

```bash
python run_ic.py --stage assemble
```

This will:
- Read complete draft_state.json (with all subtitles filled)
- Build final PowerPoint from template.pptx
- Update history.json with current week's scores
- Save PPTX to Dropbox IC folder

---

## Prerequisites

- Bloomberg Terminal running in Parallels (IP: 10.211.55.3, Port: 8194)
- Windows VM netsh portproxy active (see setup notes)
- conda environment: `ptf_opt`
- Playwright chromium installed: `playwright install chromium`
- Python packages: blpapi, pptx, plotly, pandas, numpy, yaml, kaleido

## Environment variables

```bash
IC_DROPBOX_PATH=/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/ic
```

## Fallback (Bloomberg unavailable)

```bash
python run_ic.py --stage prepare --date [DATE] --skip-bloomberg
```

Uses existing master_prices.csv without pulling new data.

## Full API fallback (outside Claude Code)

```bash
python run_ic.py --stage full --date [DATE]
```

Uses Claude API (claude-opus-4-6) for subtitle generation instead of Claude Code.
Requires ANTHROPIC_API_KEY environment variable.
