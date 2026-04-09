#!/usr/bin/env python3
"""
IC Pipeline V2 — Entry Point

Usage:
    python run_ic.py --stage prepare --date 2026-03-11
    # → Claude Code reads draft_state.json and writes subtitles directly
    python run_ic.py --stage assemble

    python run_ic.py --stage full --date 2026-03-11
    # Note: --stage full runs prepare then assemble without subtitles.
    # Use prepare → (Claude Code writes subtitles) → assemble for full workflow.

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
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def _telegram(message: str) -> None:
    """Send a Telegram message via @herculis_ic_bot (best-effort — never raises)."""
    try:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "979257663")
        if not token:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception:
        pass


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
        _telegram(f"IC pipeline failed at {args.stage}: {e}")
        return 1

    _telegram(
        f"IC pipeline complete, Market Compass {args.date or 'latest'} saved to Dropbox."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
