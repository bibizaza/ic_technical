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
- **2 lines**, max 12 words each, no period at end
- **Never** start with the instrument name
- Each subtitle must be **uniquely different** across the full 20-instrument batch
- All numbers from the data only — no invented numbers
- Directional claims must match scores (bullish language only when DMAS >= 60)
- **Line 1:** most important fact RIGHT NOW
- **Line 2:** what to watch next / key level or catalyst

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
