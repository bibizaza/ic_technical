#!/usr/bin/env python3
"""
IC Pipeline V2 — Entry Point

Usage:
    python run_ic.py --stage prepare --date 2026-03-11
    python run_ic.py --stage assemble
    python run_ic.py --stage full --date 2026-03-11

Flags:
    --stage         prepare | assemble | full
    --date          YYYY-MM-DD (default: latest in master_prices.csv)
    --skip-bloomberg  Skip Bloomberg pull, use existing data
    --output        Override output directory
    --draft         Path to draft_state.json (default: ./draft_state.json)
    --template      Path to template.pptx (overrides Dropbox path)
    --master-csv    Path to master_prices.csv (overrides Dropbox path)
    --config        Path to config/tickers.yaml
    --log-level     DEBUG | INFO | WARNING (default: INFO)
"""
from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="IC Pipeline V2 — Bloomberg → Scores → PowerPoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "assemble", "full"],
        required=True,
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Target date YYYY-MM-DD (default: latest in master_prices.csv)",
    )
    parser.add_argument(
        "--skip-bloomberg",
        action="store_true",
        help="Skip Bloomberg pull, use existing master_prices.csv",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory for draft_state.json",
    )
    parser.add_argument(
        "--draft",
        default="draft_state.json",
        help="Path to draft_state.json",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Path to template.pptx",
    )
    parser.add_argument(
        "--master-csv",
        default=None,
        help="Path to master_prices.csv",
    )
    parser.add_argument(
        "--config",
        default="config/tickers.yaml",
        help="Path to config/tickers.yaml",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level: DEBUG | INFO | WARNING",
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    try:
        if args.stage in ("prepare", "full"):
            log.info("=== Stage: prepare ===")
            from pipeline.prepare import run_prepare
            run_prepare(
                date=args.date,
                skip_bloomberg=args.skip_bloomberg,
                output_dir=args.output,
                draft_path=args.draft,
                master_csv=args.master_csv,
                config_path=args.config,
            )

        if args.stage in ("assemble", "full"):
            # Guard: draft_state.json must exist before assemble
            from pathlib import Path as _Path
            if not _Path(args.draft).exists():
                print(f"Error: {args.draft} not found. Run --stage prepare first.")
                return 1

            # Generate subtitles via Claude API whenever they're missing
            log.info("=== Stage: subtitle ===")
            _generate_subtitles_api(args.draft, args.config)

        if args.stage in ("assemble", "full"):
            log.info("=== Stage: assemble ===")
            from pipeline.assemble import run_assemble
            run_assemble(
                draft_path=args.draft,
                template_path=args.template,
                output_path=None,
                config_path=args.config,
            )

    except KeyboardInterrupt:
        log.info("Interrupted by user")
        return 1
    except Exception as e:
        log.error("Pipeline failed: %s", e, exc_info=True)
        return 1

    return 0


def _generate_subtitles_api(draft_path: str, config_path: str) -> None:
    """
    Generate subtitles via Claude API for any instruments missing them.
    Runs automatically before assemble in both --stage assemble and --stage full.
    """
    import json
    import os
    import anthropic

    log = logging.getLogger("subtitles_api")

    with open(draft_path) as f:
        draft = json.load(f)

    instruments = draft["instruments"]
    needs_instrument_subs = any(not d.get("subtitle") for d in instruments.values())
    needs_ytd_subs = not draft.get("ytd_subtitles")

    if not needs_instrument_subs and not needs_ytd_subs:
        log.info("All subtitles already present — skipping generation")
        return

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    if needs_instrument_subs:
        missing = {name: data for name, data in instruments.items() if not data.get("subtitle")}
        log.info("Generating subtitles for %d instruments: %s", len(missing), ", ".join(missing))
        batch_prompt = _build_subtitle_prompt(missing)
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": batch_prompt}],
        )
        raw = message.content[0].text
        subtitles = _parse_subtitle_response(raw, list(missing.keys()))
        for name, subtitle in subtitles.items():
            if name in draft["instruments"]:
                # Ensure line 1 ends with a period
                lines = subtitle.split("\n", 1)
                if lines and lines[0] and not lines[0].rstrip().endswith("."):
                    lines[0] = lines[0].rstrip() + "."
                subtitle = "\n".join(lines)
                draft["instruments"][name]["subtitle"] = subtitle
                log.info("Subtitle for %s: %s", name, subtitle[:60])

    if needs_ytd_subs:
        ytd_subs = _generate_ytd_subtitles_api(draft, client, log)
        draft["ytd_subtitles"] = ytd_subs

    with open(draft_path, "w") as f:
        json.dump(draft, f, indent=2, default=str)

    log.info("Subtitles written to %s", draft_path)


def _generate_ytd_subtitles_api(draft: dict, client, log) -> dict:
    """
    Compute YTD performance from price data and generate subtitles for the
    three YTD evolution slides (equity, commodity, crypto) via Claude API.
    """
    import os
    from pathlib import Path

    dropbox_path = os.environ.get(
        "IC_DROPBOX_PATH",
        "/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/ic",
    )
    csv_path = Path(dropbox_path) / "master_prices.csv"

    try:
        from data_loader import load_prices_from_csv, compute_ytd_performance
        from datetime import date as _date
        import pandas as pd

        ic_date = draft.get("date")
        data_as_of = pd.Timestamp(ic_date).date() if ic_date else None
        df_prices = load_prices_from_csv(csv_path, data_as_of=data_as_of)

        # Key tickers for each slide
        equity_tickers = {
            "S&P 500": "SPX Index", "DAX": "DAX Index", "Nikkei 225": "NKY Index",
            "TASI": "SASEIDX Index", "Sensex": "SENSEX Index", "CSI 300": "SHSZ300 Index",
            "IBOV": "IBOV Index",
        }
        commodity_tickers = {
            "Gold": "GCA Comdty", "Silver": "SIA Comdty", "Oil": "CL1 Comdty",
            "Copper": "LP1 Comdty",
        }
        crypto_tickers = {
            "Bitcoin": "XBTUSD Curncy", "Ethereum": "XETUSD Curncy", "Solana": "XSOUSD Curncy",
        }

        def _ytd_block(ticker_map: dict) -> str:
            lines = []
            for name, ticker in ticker_map.items():
                ytd = compute_ytd_performance(df_prices, ticker)
                if ytd is not None:
                    sign = "+" if ytd >= 0 else ""
                    lines.append(f"  {name}: {sign}{ytd:.1f}% YTD")
            return "\n".join(lines)

        eq_block = _ytd_block(equity_tickers)
        co_block = _ytd_block(commodity_tickers)
        cr_block = _ytd_block(crypto_tickers)

    except Exception as e:
        log.warning("Could not compute YTD performance: %s", e)
        return {}

    prompt = f"""Generate a SINGLE-LINE subtitle for each of three YTD performance slides.
Exactly one line per slide. Max 12 words. End with a period. Use only data provided.
Never start with the asset class name.

These subtitles appear on YTD evolution chart slides. They must tell a REGIONAL
or THEMATIC story, not just name the best and worst performer.

RULES:
1. Frame the narrative around regions, sectors, or themes — not individual names.
   Think "EM vs DM", "energy vs metals", "Bitcoin dominance vs altcoin weakness".
2. Highlight what is UNUSUAL or SURPRISING — the story a portfolio manager
   would tell a client.
3. Quantify only if the number is striking (e.g. "26-point spread", "+60% YTD").
   At most 1 number per line.
4. FORBIDDEN words: divergence, mixed, varied, dynamics, remarkable, exceptional,
   extraordinary, outstanding.
5. Must be factually verifiable from the data provided.
6. Do NOT name more than one individual instrument per subtitle. If you name
   one (e.g. "Oil's +61%"), the rest should be a category ("industrial metals",
   "altcoins", "Western indices") — not a second name.
7. Numbers must match actual data to within 1 percentage point. Do not round
   aggressively or use approximate language ("nearly") — either use the exact
   number or drop it.
8. Never generalize a region when performance diverges within it. If Brazil is
   +13% and India is -13%, "emerging markets" are NOT down — they are split.
   Name specific indices, not regional labels, when data diverges within a group.

GOOD examples:
- "Emerging markets lead 2026 as Western indices give back last year's gains"
- "Oil's +60% surge dwarfs everything as precious metals reverse course"
- "Bitcoin holds relative ground while altcoins sink 20%+ across the board"

BAD examples:
- "Brazil leads at +12% while India lags at -10.7% YTD" (just naming best/worst)
- "Wide divergence across global equities" (vague, says nothing)
- "Developed markets stumble as EM surges" (wrong if India is EM and worst)

FORMAT:
### equity
One line here
### commodity
One line here
### crypto
One line here

=== EQUITY YTD PERFORMANCE ===
{eq_block}

=== COMMODITY YTD PERFORMANCE ===
{co_block}

=== CRYPTO YTD PERFORMANCE ===
{cr_block}
"""

    try:
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        ytd_subtitles = {}
        for section in ("equity", "commodity", "crypto"):
            marker = f"### {section}"
            if marker in raw.lower():
                start = raw.lower().index(marker) + len(marker)
                chunk = raw[start:].strip()
                lines = [ln.strip() for ln in chunk.split("\n") if ln.strip() and not ln.startswith("###")][:1]
                ytd_subtitles[section] = lines[0] if lines else ""
                log.info("YTD subtitle [%s]: %s", section, ytd_subtitles[section][:60])
        return ytd_subtitles
    except Exception as e:
        log.warning("YTD subtitle generation failed: %s", e)
        return {}


def _build_subtitle_prompt(instruments: dict) -> str:
    """Build the subtitle generation prompt for all 20 instruments."""
    lines = [
        "Generate 2-line subtitles for each of the following 20 investment instruments.",
        "Each slide shows a 5-month price chart with 50d/100d/200d MAs, RSI(14), and a",
        "DMAS scorecard. Your subtitle sits below the instrument name and rating.",
        "",
        "CORE PHILOSOPHY: Write like a technical analyst briefing a committee.",
        "- Line 1: What is the chart structure telling us right now?",
        "- Line 2: What is the next technical level to watch, and what happens there?",
        "",
        "HARD RULES (violations = automatic reject):",
        "R1 — No scores in text: Never mention DMAS, Technical score, Momentum score,",
        "  breadth rank, or fundamental rank by name. RSI level is OK as a technical",
        "  indicator (e.g. 'RSI at 30') but NOT 'RSI score at 30'.",
        "R2 — Max 2 numbers per line. Prioritize MA distance % and RSI over raw prices.",
        "R3 — No promotional adjectives: extraordinary, exceptional, remarkable,",
        "  outstanding, perfect, relentless, impressive, stunning, incredible, powerful.",
        "R4 — No instrument name as first words.",
        "R5 — All numbers must be verifiable from the data provided.",
        "R6 — Streak counts: use EXACTLY the 'Rating streak: X for N consecutive weeks'",
        "  field. Never count history entries yourself.",
        "R7 — No investment recommendations (buy, sell, add, reduce, take profits).",
        "R8 — Both lines end with a period.",
        "R9 — No false regional generalizations: Any group claim (EM, DM, metals)",
        "  must be supported by ALL members. When performance diverges within a group,",
        "  name specific indices instead of regional labels.",
        "R10 — MA position verification (CRITICAL): When vs_50d/vs_100d/vs_200d is",
        "  NEGATIVE, price is BELOW that MA — the MA is overhead resistance, NOT support.",
        "  When POSITIVE, price is ABOVE — the MA is support below. Verify every MA",
        "  sentence against the sign of the corresponding vs_Xd field.",
        "",
        "TECHNICAL NARRATIVE — MA STACK (your primary tool):",
        "- Above all 3 MAs, fanning up → 'All MAs rising and well-spaced'",
        "- Above 200d, below 50d → 'Corrected to the 100d zone but 200d underpins'",
        "- Below all 3 MAs, curling down → 'Sliced through all MAs with 50d curling lower'",
        "- Testing a specific MA → 'Clinging to the 200d as last support'",
        "- Between MAs → 'Trapped between the 100d and 200d'",
        "",
        "LINE 2 — FORWARD-LOOKING (almost always include a scenario):",
        "- 'Needs to reclaim the 50d near X to restore bullish structure.'",
        "- 'A break below the 200d would open a new leg lower.'",
        "- 'The rising 200d near X is the first real support for a bounce.'",
        "",
        "TONE CALIBRATION (match vocabulary to DMAS range without naming the score):",
        "- DMAS 80-100: confident — 'textbook structure', 'next leg', 'resets for continuation'",
        "- DMAS 60-79: balanced — 'broader uptrend intact', 'needs to reclaim', 'would restore'",
        "- DMAS 40-59: cautious — 'approaching support', 'line in the sand', 'pivotal zone'",
        "- DMAS 20-39: urgent — 'rapid succession', 'losing levels', 'last support'",
        "- DMAS 0-19: structural damage — 'freefall', 'no base forming', 'stabilization needed'",
        "",
        "WoW DELTA modulates intensity:",
        "- Large negative (↓15+): 'sharp reversal', 'violent correction'",
        "- Small negative (↓1-5): 'continues to drift', 'edging lower'",
        "- Positive: 'beginning to stabilize', 'first uptick in X weeks'",
        "",
        "TECH vs MOMENTUM DIVERGENCE (>30pt gap):",
        "- High Momentum + Low Technical: frame as sharp correction within broader trend",
        "- Low Momentum + High Technical: near-term improved but broader trend weak",
        "",
        "BANNED PHRASES: 'recovery hinges on', 'recovery depends on', 'outlook persists',",
        "  'dynamics continue', 'exceptional strength remains', 'momentum supports further",
        "  gains', 'bullish streak remains unbroken', 'maintains bullish trend', 'resilience",
        "  despite DMAS decline', 'momentum strength key to', 'hinges on reclaiming 50d MA',",
        "  'downgraded from X to Y', 'score floored at zero', 'momentum score at X',",
        "  'technical score at X'",
        "",
        "EXAMPLES:",
        "DMAS 6 (structural damage):",
        "  ✅ 'In freefall with all MAs now sloping down and price 10.2% below the 200d.'",
        "     'RSI nearing oversold at 34 but no base forming — stabilization needed before any recovery attempt.'",
        "DMAS 91 (strong bullish):",
        "  ✅ 'All three MAs rising and well-spaced — textbook bullish structure.'",
        "     'Pulled back to the 50d after 14 weeks of gains; RSI at 41 resets overbought conditions for the next leg.'",
        "DMAS 42 (cautious, ↓38 WoW):",
        "  ✅ 'Violent correction through the 50d and 100d after a parabolic run.'",
        "     'RSI at 25 is deeply oversold — the rising 200d near 4,200 is the first real support for a bounce attempt.'",
        "",
        "R10 violation (vs_50d=-5.0%, vs_100d=-3.2%, vs_200d=+8.7%):",
        "  ❌ 'The 100d just below current price is the near-term pivot.' (WRONG: 100d is ABOVE price)",
        "  ✅ 'Corrected 5.0% below the 50d with the 100d also overhead — the 200d 8.7% below remains the structural floor.'",
        "",
        "WORKFLOW:",
        "1. Read all 20 instruments first — batch awareness prevents repetition.",
        "2. For each instrument, note the SIGN of vs_50d/vs_100d/vs_200d before writing.",
        "3. After drafting, re-read every MA sentence and verify direction matches the sign.",
        "",
        "FORMAT: For each instrument, output:",
        "### [Instrument Name]",
        "Line 1 text.",
        "Line 2 text.",
        "",
        "=== INSTRUMENT DATA ===",
        "",
    ]

    for name, data in instruments.items():
        lines.append(data.get("enriched_context", f"=== {name} ==="))
        lines.append("")

    return "\n".join(lines)


def _parse_subtitle_response(raw: str, instrument_names: list) -> dict:
    """Parse the structured subtitle response."""
    subtitles = {}
    lines = raw.strip().split("\n")

    current_name = None
    current_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("### "):
            if current_name and current_lines:
                subtitles[current_name] = "\n".join(current_lines[:2])
            current_name = line[4:].strip()
            # Find best match in instrument_names
            best_match = next(
                (nm for nm in instrument_names if nm.lower() in current_name.lower()),
                current_name,
            )
            current_name = best_match
            current_lines = []
        elif current_name and line and not line.startswith("###"):
            current_lines.append(line)

    if current_name and current_lines:
        subtitles[current_name] = "\n".join(current_lines[:2])

    return subtitles


if __name__ == "__main__":
    sys.exit(main())
