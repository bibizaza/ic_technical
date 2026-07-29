# Session Handout — 2026-07-16 09:00

> Next step: `/clear`, then in fresh session say: "Read `./SESSION_HANDOUT.md` and resume."

Working dir: `/Users/larazanella/Desktop/GitHub/Projects/ic_technical`
Branch: `main`
Last commit: `1c8644f fix: replace dead CCTON with CCRITON Index for Ton on YTD evolution chart`

## 1. Objective

Boss-driven IC2 Market Compass deck edits: bond-slide changes (add Switzerland govvies + Swiss IG corp), crypto fixes (Ton data dead → proxy), plus fixing an old interfering cron. All shipped to `main`.

## 2. Current state

All requested edits committed + pushed (through `1c8644f`). Working tree clean except generated caches + untracked helper. Standalone slide artifacts (png/pdf/pptx) delivered to Dropbox subfolders. The **full deck was last regenerated at commit `fbc2a73`-era** (08072026 deck) and does NOT yet include: gov-bond reorder (`7c0a5db`), Swiss IG corp (`96fbb92`), or the CCRITON Ton fix (`1c8644f`). No full-deck regen requested since.

## 3. Decisions made (this session)

- **Ton removed** from crypto weekly + historical (already done, `3768c0f`). On the YTD line chart, Ton's `CCTON Curncy` died in Bloomberg (last data 2026-04-30). Replaced with **`CCRITON Index`** (Compass Crypto Reference Index Toncoin), NOT the 21Shares Gram ETP the boss suggested — ETP too illiquid early-2026 (stale prints distort YTD baseline). CCRITON continues CCTON's level (Apr30: 1.319 vs 1.305).
- The **early-May Toncoin ~2x spike is REAL** (corroborated across CCRITON, CCNYTON, GRAM ETP, CoinShares CTON) — not an artifact. Ton YTD now −3.2% (real).
- **Swiss govvies added** as 5th bond country; **order = US, EU, Switzerland, Japan, China** (`7c0a5db`). Correct 2Y ticker is `GSWISS02` (not `GSWISS2`, which is a different series).
- **Swiss IG corp added** as `CHF IG Corp` via `SFC14T Index` (SBI Foreign Corporate AAA-BBB TR). No Swiss HY line — no domestic CHF HY benchmark exists (only global/EU HY hedged-to-CHF).
- **Old Tuesday cron disabled**: `~/.claude/scheduled-tasks/ic2-weekly/` (cron `0 23 * * 2`, Telegram + old `run_ic.py --stage full`) set `enabled:false` in the Claude Desktop `scheduled-tasks.json`. Real weekly run is now only the intern-PC WTS (`\IC2 Market Compass`, Wed 07:15, Claude Code CLI, email).

## 4. Files touched (this session)

| Path | Change |
|------|--------|
| `config/tickers.yaml` | breadth_fields swap (RSI/MA in, MACD/Bollinger out) |
| `pipeline/breadth.py` | conviction=MA spread, sentiment=RSI extremes |
| `pipeline/bloomberg.py:92-107` | bbg_bdp logs securityError/fieldExceptions |
| `pipeline/prepare.py:427+` | breadth usability gate checks all 6 pillar cols |
| `pipeline/assemble.py:259+` | slide titles include last price |
| `technical_analysis/**` (20 modules) | `PLOT_LOOKBACK_DAYS` 90→274; spx `TECH_V2_LOOKBACK_DAYS` 85→195 |
| `market_compass/templates/quadrant_chart.html` | "→ (1 Month)" legend |
| `market_compass/quadrant_slide.py` | append "(Over 1 Month)" to title (match by text, not `.shapes.title`) |
| `performance/crypto_perf.py` | Ton out of weekly/hist; YTD baseline=first-of-year; grey "-" for NaN horizons; Ton line → `CCRITON Index` |
| `market_compass/weekly_performance/html_template.py` | crypto `.neutral` grey class; gov-bond templates compressed (weekly row 20→14, hist cell 18→13/ytd 20→14) |
| `performance/commodity_perf.py` | `_compute_commodity_ytd_grid` date-aligns series (copper/uranium flatline fix) |
| `performance/rates_perf.py` | Switzerland in COUNTRY_CONFIG (GSWISS02/10/30), reorder, `_load_price_data` now ffills |
| `performance/corp_bonds_perf.py` | `CHF IG Corp` (SFC14T) added to CORP_BOND_CONFIG |
| `seed_swiss_rates.py` (new) | one-time Swiss-rates seeder |
| `seed_tickers.py` (new) | **generic** ticker-column seeder for master_prices.csv |

## 5. Commits made (this session)

- `dd87ebe` 4 deck changes — breadth re-spec, slide titles, 9-month charts, quadrant legend
- `81baf51` append "(Over 1 Month)" to quadrant slide title
- `9e48f1d` match quadrant title by text, not shapes.title placeholder
- `5bd683e` crypto perf table — YTD from first-of-year price, grey dash for missing
- `e565a6d` commodity YTD flatlines copper/uranium — date-align all series
- `3768c0f` remove Ton from crypto weekly and historical
- `d8ccfc0` add Switzerland to government bond weekly + historical slides
- `3a1f2a9` rates loader forward-fills; correct Swiss 2Y ticker to GSWISS02
- `fbc2a73` compress gov-bond templates so 5 countries fit
- `7c0a5db` reorder gov-bond countries to US, EU, Switzerland, Japan, China
- `96fbb92` add Swiss IG (CHF IG Corp) to corporate bond slides
- `1c8644f` replace dead CCTON with CCRITON Index for Ton on YTD chart

## 6. Uncommitted state

```
 M breadth_cache.json          (generated — do not commit)
 M fundamental_cache.json      (generated — do not commit)
?? SESSION_HANDOUT.md          (this file)
?? diagnose_breadth_fields.py  (untracked helper, can delete)
```

## 7. Open threads / in-flight work

- **Full-deck regen pending (optional).** Deck on disk (`.../ic/08072026_Herculis_Partners_Technical_Update.pptx`) predates `7c0a5db`/`96fbb92`/`1c8644f`. To regen: `conda activate ptf_opt`, then `python cli_generate.py --excel "$D/ic_file.xlsx" --prices "$D/master_prices.csv" --template "$D/template.pptx" --history "$D/history.json" --output "$D" --date latest --skip-subtitles` (D = `.../Tools_In_Construction/ic`). ~4min. Reuses draft_state.json subtitles (API path blocked).
- **`GRAM SW Equity` left seeded but unused** in master_prices.csv (harmless dead column). Remove if desired (no unseed tool; manual CSV surgery).
- **Momentum rescoring can't run here**: `run_momentum.py` lives on momentum branch `claude/export-score-to-excel-…`, not checked out on momentum repo `main`. Not needed for presentation-only changes.
- **Intern-PC duplicate check**: old Tuesday cron disabled on Mac only. If a stale WTS task also lingers on the Windows intern PC, check there: `schtasks /query /fo LIST /v | findstr /i "IC Market Compass"`.

## 8. Gotchas discovered

- **`update_master_prices_wide` reads its ticker universe from the CSV header**, not tickers.yaml — cannot onboard new tickers. New series MUST be seeded (`seed_tickers.py "TICKER Index"`; backs up, seeds from 2023, covers YTD+12M). A single 2006→now BDH request truncates at ~5100 pts and silently drops the recent year.
- **`GSWISS2 Index` ≠ 2Y** (returns a different series ~0.63); the real Swiss 2Y is **`GSWISS02`** (~0.11).
- **Rates/corp slides read from a temp Excel built from master_prices.csv** (`create_temp_excel_from_csv`), NOT the CSV directly — `_load_price_data` does `pd.read_excel`.
- **`rates_perf._load_price_data` did NOT ffill** (unique among perf loaders) → a market 1 day stale (Eurozone GECU) went NaN on the max date and the whole country dropped. Fixed with ffill.
- **Gov-bond 5 countries overflowed** into the footnote (content 92% of canvas). My HTML estimate said it fit — WRONG. Measure real height with headless playwright (`document.querySelector('.chart-container').getBoundingClientRect().height`), target ~74% (what 4 countries used).
- **Subtitle/recap generation is API-blocked** (`IC_ALLOW_ANTHROPIC_API` guard). Use `--skip-subtitles` to reuse draft_state.json narratives.
- **21Shares/CoinShares crypto ETPs are illiquid** (blank-volume days) → stale prints that snap; bad for YTD baselines. Use a crypto reference index (Compass `CCRI*`, Kaiko `KRTON*`) for clean spot.
- **The old "cron" was a Claude Desktop scheduled task** (`~/.claude/scheduled-tasks/ic2-weekly/`), registered in `.../Claude/claude-code-sessions/.../scheduled-tasks.json`, NOT unix cron/launchd.

## 9. User preferences (this session)

- Terse/fragment style ("CAVEMAN"/ponytail-lazy). Boss changes: implement exactly, no scope creep.
- Wants FULL slides (title/banner/logo/footnote), not bare charts, when asked for slide deliverables — build 2-slide PPTX from template, export via PowerPoint AppleScript → PDF → `pdftoppm` PNG.
- Confirm before overriding an explicit instruction with a better option (e.g. CCRITON over the ETP the boss named).

## 10. How to resume

1. `cd /Users/larazanella/Desktop/GitHub/Projects/ic_technical`; `source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate ptf_opt`.
2. Read this file + `SKILL.md` (pipeline spec, scheduled-task section).
3. `git log --oneline -12` to confirm state matches section 5.
4. If asked to ship the updated deck: run the `cli_generate.py` command in section 7, then verify rates (slides 6-7) + corp (8-9) + crypto YTD reflect the changes.
5. To render a single slide set (png/pdf/pptx): use the build-2-slide-PPTX + PowerPoint-export pattern (scratchpad had `build_rates_pptx.py` / `build_corp_pptx.py`; regenerate as needed).
6. New Bloomberg tickers → `python seed_tickers.py "TICKER Index"` (dry-run first with `--dry-run`).
