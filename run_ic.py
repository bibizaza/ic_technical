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

        if args.stage == "full":
            # Full mode: use Claude API for subtitles (fallback for non-Claude-Code runs)
            log.info("=== Stage: subtitle (API fallback) ===")
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
    Fallback subtitle generation via Claude API (for --stage full).
    Used when running outside Claude Code (e.g., scheduled cron jobs).
    """
    import json
    import os
    import anthropic

    log = logging.getLogger("subtitles_api")

    with open(draft_path) as f:
        draft = json.load(f)

    instruments = draft["instruments"]

    # Build batch prompt
    context_blocks = []
    for name, data in instruments.items():
        context_blocks.append(data.get("enriched_context", f"=== {name} ===\nNo data available."))

    batch_prompt = _build_subtitle_prompt(instruments)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": batch_prompt}],
    )

    # Parse subtitles from response
    raw = message.content[0].text
    subtitles = _parse_subtitle_response(raw, list(instruments.keys()))

    for name, subtitle in subtitles.items():
        if name in draft["instruments"]:
            draft["instruments"][name]["subtitle"] = subtitle
            log.info("Subtitle for %s: %s", name, subtitle[:60])

    with open(draft_path, "w") as f:
        json.dump(draft, f, indent=2, default=str)

    log.info("Subtitles written to %s", draft_path)


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
