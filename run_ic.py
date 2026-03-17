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
Exactly one line per slide. Max 12 words. No period at end. Use only data provided.
Never start with the asset class name.

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
        "",
        "RULES:",
        "- 2 lines per instrument, max 12 words each, no period at end",
        "- Never start with the instrument name",
        "- Each subtitle must be uniquely different across the full batch",
        "- Use only numbers from the data",
        "- Directional claims must match scores (bullish language ↔ high DMAS)",
        "- Line 1: most important fact RIGHT NOW",
        "- Line 2: what to watch next",
        "- FORBIDDEN phrases: 'recovery hinges on', 'outlook persists', 'dynamics continue',",
        "  'exceptional strength remains', 'bullish streak remains unbroken',",
        "  'maintains bullish trend', 'momentum supports further gains'",
        "",
        "FORMAT: For each instrument, output:",
        "### [Instrument Name]",
        "Line 1 text",
        "Line 2 text",
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
