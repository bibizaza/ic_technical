#!/usr/bin/env python3
"""
IC Technical Presentation CLI Generator

Usage:
    python cli_generate.py \
        --excel /path/to/ic_file.xlsx \
        --template /path/to/template.pptx \
        --output /path/to/output/ \
        --history /path/to/history.json \
        --date latest

Environment Variables:
    ANTHROPIC_API_KEY - Required for Claude subtitle generation
    CMC_API_KEY - Optional for crypto market caps
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import date, datetime
from typing import Dict, Any, Optional

import pandas as pd


# ==============================================================================
# INSTRUMENT DEFINITIONS
# ==============================================================================

# Asset mapping: ticker_key -> (Bloomberg ticker, display name)
ASSET_MAP = {
    # Equity
    "spx": ("SPX Index", "S&P 500"),
    "csi": ("SHSZ300 Index", "CSI 300"),
    "nikkei": ("NKY Index", "Nikkei 225"),
    "tasi": ("SASEIDX Index", "TASI"),
    "sensex": ("SENSEX Index", "Sensex"),
    "dax": ("DAX Index", "Dax"),
    "smi": ("SMI Index", "SMI"),
    "ibov": ("IBOV Index", "IBOV"),
    "mexbol": ("MEXBOL Index", "MEXBOL"),
    # Commodity
    "gold": ("GCA Comdty", "Gold"),
    "silver": ("SIA Comdty", "Silver"),
    "platinum": ("XPT Comdty", "Platinum"),
    "palladium": ("XPD Curncy", "Palladium"),
    "oil": ("CL1 Comdty", "Oil"),
    "copper": ("LP1 Comdty", "Copper"),
    # Crypto
    "bitcoin": ("XBTUSD Curncy", "Bitcoin"),
    "ethereum": ("XETUSD Curncy", "Ethereum"),
    "ripple": ("XRPUSD Curncy", "Ripple"),
    "solana": ("XSOUSD Curncy", "Solana"),
    "binance": ("XBIUSD Curncy", "Binance"),
}

# Ticker to ticker_key mapping
TICKER_TO_KEY = {
    "SPX INDEX": "spx",
    "SHSZ300 INDEX": "csi",
    "NKY INDEX": "nikkei",
    "SASEIDX INDEX": "tasi",
    "SENSEX INDEX": "sensex",
    "DAX INDEX": "dax",
    "SMI INDEX": "smi",
    "IBOV INDEX": "ibov",
    "MEXBOL INDEX": "mexbol",
    "GCA COMDTY": "gold",
    "SIA COMDTY": "silver",
    "XPT COMDTY": "platinum",
    "XPD CURNCY": "palladium",
    "CL1 COMDTY": "oil",
    "LP1 COMDTY": "copper",
    "XBTUSD CURNCY": "bitcoin",
    "XETUSD CURNCY": "ethereum",
    "XRPUSD CURNCY": "ripple",
    "XSOUSD CURNCY": "solana",
    "XBIUSD CURNCY": "binance",
}


# ==============================================================================
# STATE INITIALIZATION
# ==============================================================================

def get_max_date_from_excel(excel_path: Path) -> date:
    """Get the maximum date available in the Excel data."""
    df = pd.read_excel(excel_path, sheet_name="data_prices")
    df = df.drop(index=0)
    df = df[df[df.columns[0]] != "DATES"]
    df["Date"] = pd.to_datetime(df[df.columns[0]], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df["Date"].max().date()


def load_transition_sheet(excel_path: Path, state: Dict[str, Any]) -> None:
    """Load initial state from transition sheet."""
    print("[CLI] Loading transition sheet...")
    try:
        df = pd.read_excel(excel_path, sheet_name="transition")
        print(f"[CLI] Found {len(df)} rows in transition sheet")

        for _, row in df.iterrows():
            ticker = row.iloc[0] if len(row) > 0 else None
            if pd.isna(ticker) or str(ticker).strip() == "":
                continue

            ticker_upper = str(ticker).strip().upper()
            ticker_key = TICKER_TO_KEY.get(ticker_upper)

            if not ticker_key:
                print(f"[CLI] Unknown ticker in transition sheet: {ticker_upper}")
                continue

            # Column B: Last week DMAS
            if len(row) > 1 and not pd.isna(row.iloc[1]):
                try:
                    state[f"{ticker_key}_last_week_avg"] = float(row.iloc[1])
                except (ValueError, TypeError):
                    pass

            # Column C: Anchor date
            if len(row) > 2 and not pd.isna(row.iloc[2]):
                try:
                    anchor = pd.to_datetime(row.iloc[2])
                    state[f"{ticker_key}_anchor"] = anchor
                    state[f"{ticker_key}_enable_channel"] = True
                except Exception:
                    pass

            # Column D: Assessment
            if len(row) > 3 and not pd.isna(row.iloc[3]):
                state[f"{ticker_key}_selected_view"] = str(row.iloc[3]).strip()

            # Column E: Subtitle (if pre-defined)
            if len(row) > 4 and not pd.isna(row.iloc[4]):
                state[f"{ticker_key}_subtitle"] = str(row.iloc[4]).strip()

        print(f"[CLI] Transition sheet loaded")
    except Exception as e:
        print(f"[CLI] Warning: Could not load transition sheet: {e}")


def load_history_data(history_path: Path, state: Dict[str, Any], current_date: date) -> None:
    """Load previous week's data from history.json."""
    print(f"[CLI] Loading history from {history_path}...")

    if not history_path.exists():
        print("[CLI] No history file found, skipping")
        return

    try:
        import json
        with open(history_path, 'r') as f:
            history_data = json.load(f)

        # Asset name to ticker_key mapping
        ASSET_NAME_TO_KEY = {
            "S&P 500": "spx", "SPX": "spx",
            "CSI 300": "csi", "CSI300": "csi",
            "Nikkei 225": "nikkei", "Nikkei": "nikkei",
            "TASI": "tasi",
            "Sensex": "sensex",
            "Dax": "dax", "DAX": "dax",
            "SMI": "smi",
            "Ibov": "ibov", "IBOV": "ibov", "Ibovespa": "ibov",
            "Mexbol": "mexbol", "MEXBOL": "mexbol",
            "Gold": "gold", "GOLD": "gold",
            "Silver": "silver", "SILVER": "silver",
            "Platinum": "platinum", "PLATINUM": "platinum",
            "Palladium": "palladium", "PALLADIUM": "palladium",
            "Oil": "oil", "OIL": "oil", "WTI": "oil",
            "Copper": "copper", "COPPER": "copper",
            "Bitcoin": "bitcoin", "BTC": "bitcoin",
            "Ethereum": "ethereum", "ETH": "ethereum",
            "Ripple": "ripple", "XRP": "ripple",
            "Solana": "solana", "SOL": "solana",
            "Binance": "binance", "BNB": "binance",
        }

        loaded_count = 0
        for asset_name, entries in history_data.items():
            ticker_key = ASSET_NAME_TO_KEY.get(asset_name)
            if not ticker_key:
                continue

            # Find most recent entry BEFORE current_date
            sorted_entries = sorted(entries, key=lambda x: x.get("date", ""), reverse=True)
            for entry in sorted_entries:
                entry_date = datetime.strptime(entry["date"], "%Y-%m-%d").date()
                if entry_date < current_date:
                    # Load DMAS
                    if entry.get("dmas") is not None:
                        state[f"{ticker_key}_last_week_avg"] = float(entry["dmas"])
                        loaded_count += 1

                    # Load Technical/Momentum/RSI
                    if entry.get("technical_score") is not None:
                        state[f"{ticker_key}_last_week_tech"] = float(entry["technical_score"])
                    if entry.get("momentum_score") is not None:
                        state[f"{ticker_key}_last_week_mom"] = float(entry["momentum_score"])
                    if entry.get("rsi") is not None:
                        state[f"{ticker_key}_last_week_rsi"] = float(entry["rsi"])

                    # Days gap
                    days_gap = (current_date - entry_date).days
                    state[f"{ticker_key}_prev_days_gap"] = days_gap
                    state[f"{ticker_key}_prev_date"] = entry_date

                    break

        print(f"[CLI] Loaded previous data for {loaded_count} assets from history")
    except Exception as e:
        print(f"[CLI] Warning: Could not load history: {e}")


def compute_scores(excel_path: Path, state: Dict[str, Any], data_as_of: date) -> None:
    """Compute DMAS scores for all instruments."""
    print("[CLI] Computing technical scores...")

    from technical_score_wrapper import compute_dmas_scores

    # Read price data
    df_prices = pd.read_excel(excel_path, sheet_name="data_prices")
    df_prices = df_prices.drop(index=0)
    df_prices = df_prices[df_prices[df_prices.columns[0]] != "DATES"]
    df_prices["Date"] = pd.to_datetime(df_prices[df_prices.columns[0]], errors="coerce")
    df_prices = df_prices[df_prices["Date"] <= pd.Timestamp(data_as_of)]

    computed_count = 0
    for ticker_key, (bbg_ticker, display_name) in ASSET_MAP.items():
        try:
            # Find matching column
            bbg_upper = bbg_ticker.upper()
            matching_col = None
            for col in df_prices.columns:
                if isinstance(col, str) and col.upper() == bbg_upper:
                    matching_col = col
                    break

            if matching_col is None:
                print(f"[CLI] Warning: No data found for {display_name} ({bbg_ticker})")
                continue

            # Extract price series
            prices_df = df_prices[["Date", matching_col]].copy()
            prices_df.columns = ["Date", "Price"]
            prices_df["Price"] = pd.to_numeric(prices_df["Price"], errors="coerce")
            prices_df = prices_df.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)

            if len(prices_df) < 200:
                print(f"[CLI] Warning: Insufficient data for {display_name} ({len(prices_df)} days)")
                continue

            prices = prices_df["Price"]

            # Compute scores
            scores = compute_dmas_scores(prices, ticker=bbg_ticker, excel_path=str(excel_path))

            state[f"{ticker_key}_tech_score"] = scores["technical_score"]
            state[f"{ticker_key}_mom_score"] = scores["momentum_score"]
            state[f"{ticker_key}_dmas"] = scores["dmas"]

            computed_count += 1
            print(f"  {display_name}: DMAS={scores['dmas']:.1f}, Tech={scores['technical_score']:.0f}, Mom={scores['momentum_score']:.0f}")

        except Exception as e:
            print(f"[CLI] Error computing scores for {ticker_key}: {e}")

    print(f"[CLI] Computed scores for {computed_count} instruments")


def generate_subtitles(excel_path: Path, state: Dict[str, Any], data_as_of: str) -> None:
    """Generate subtitles via Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[CLI] No ANTHROPIC_API_KEY found, using pattern-based subtitles")
        return

    print("[CLI] Generating subtitles via Claude API...")

    try:
        from market_compass.subtitle_generator import set_api_key
        set_api_key(api_key)

        from assessment_integration import (
            generate_claude_subtitles_batch,
            get_default_assessment_from_dmas,
            is_claude_available,
            CLAUDE_GEN_AVAILABLE,
        )

        if not CLAUDE_GEN_AVAILABLE or not is_claude_available():
            print("[CLI] Claude API not available, using pattern-based subtitles")
            return

        # Read prices for MA calculations
        df_prices = pd.read_excel(excel_path, sheet_name="data_prices")
        df_prices = df_prices.drop(index=0)
        df_prices = df_prices[df_prices[df_prices.columns[0]] != "DATES"]
        df_prices["Date"] = pd.to_datetime(df_prices[df_prices.columns[0]], errors="coerce")
        df_prices = df_prices[df_prices["Date"] <= pd.Timestamp(data_as_of)]

        # Prepare assets list
        assets_list = []
        prices_dict = {}

        for ticker_key, (bbg_ticker, display_name) in ASSET_MAP.items():
            dmas = state.get(f"{ticker_key}_dmas")
            if dmas is None:
                continue

            # Find price column
            bbg_upper = bbg_ticker.upper()
            matching_col = None
            for col in df_prices.columns:
                if isinstance(col, str) and col.upper() == bbg_upper:
                    matching_col = col
                    break

            if matching_col:
                prices_df = df_prices[["Date", matching_col]].copy()
                prices_df.columns = ["Date", "Price"]
                prices_df["Price"] = pd.to_numeric(prices_df["Price"], errors="coerce")
                prices_df = prices_df.dropna().sort_values("Date").reset_index(drop=True)
                if len(prices_df) >= 200:
                    prices_dict[ticker_key] = prices_df["Price"]

            assets_list.append({
                "ticker_key": ticker_key,
                "asset_name": display_name,
                "dmas": dmas,
                "technical_score": state.get(f"{ticker_key}_tech_score", dmas),
                "momentum_score": state.get(f"{ticker_key}_mom_score", dmas),
                "dmas_prev_week": state.get(f"{ticker_key}_last_week_avg", dmas),
            })

        if not assets_list:
            print("[CLI] No assets to generate subtitles for")
            return

        # Generate subtitles
        results = generate_claude_subtitles_batch(
            assets_list,
            prices_dict,
            data_as_of=data_as_of,
            model_key="haiku_45"
        )

        # Store results
        for ticker_key, result in results.items():
            state[f"{ticker_key}_selected_view"] = result.get("assessment", "Neutral")
            state[f"{ticker_key}_subtitle"] = result.get("subtitle", "")

        print(f"[CLI] Generated subtitles for {len(results)} assets")

    except Exception as e:
        print(f"[CLI] Error generating subtitles: {e}")
        import traceback
        traceback.print_exc()


def generate_ytd_recaps(excel_path: Path, state: Dict[str, Any]) -> None:
    """Generate YTD recap subtitles."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[CLI] No ANTHROPIC_API_KEY, skipping YTD recaps")
        return

    print("[CLI] Generating YTD recap subtitles...")

    try:
        from market_compass.subtitle_generator import set_api_key
        set_api_key(api_key)

        from market_compass.subtitle_generator.claude_generator import generate_all_recaps

        # Read performance data
        df_perf = pd.read_excel(excel_path, sheet_name="data_perf")

        # Build performance data dict
        perf_data = {
            "equity": {},
            "commodities": {},
            "crypto": {},
        }

        # Map columns to categories
        EQUITY_COLS = ["SPX", "STOXX", "TASI", "NKY", "Sensex", "Hang Seng", "IBOV"]
        COMMODITY_COLS = ["Gold", "Silver", "Oil", "Copper"]
        CRYPTO_COLS = ["Bitcoin", "Ethereum", "Solana"]

        for col in df_perf.columns:
            if col in EQUITY_COLS:
                perf_data["equity"][col] = df_perf[col].iloc[-1] if len(df_perf) > 0 else 0
            elif col in COMMODITY_COLS:
                perf_data["commodities"][col] = df_perf[col].iloc[-1] if len(df_perf) > 0 else 0
            elif col in CRYPTO_COLS:
                perf_data["crypto"][col] = df_perf[col].iloc[-1] if len(df_perf) > 0 else 0

        if any(perf_data.values()):
            recaps = generate_all_recaps(perf_data)
            state["eq_subtitle"] = recaps.get("equity", "")
            state["co_subtitle"] = recaps.get("commodities", "")
            state["cr_subtitle"] = recaps.get("crypto", "")
            print("[CLI] YTD recap subtitles generated")
        else:
            print("[CLI] No performance data found for YTD recaps")

    except Exception as e:
        print(f"[CLI] Warning: Could not generate YTD recaps: {e}")


def update_history(history_path: Path, state: Dict[str, Any], data_as_of: str) -> None:
    """Update history.json with current scores and subtitles."""
    print(f"[CLI] Updating history at {history_path}...")

    try:
        import json

        # Load existing history
        if history_path.exists():
            with open(history_path, 'r') as f:
                history_data = json.load(f)
        else:
            history_data = {}

        # Prepare batch update
        assets_to_record = []
        for ticker_key, (_, display_name) in ASSET_MAP.items():
            dmas = state.get(f"{ticker_key}_dmas")
            if dmas is None:
                continue

            assets_to_record.append({
                "asset_name": display_name,
                "dmas": int(dmas),
                "technical_score": int(state.get(f"{ticker_key}_tech_score", dmas)),
                "momentum_score": int(state.get(f"{ticker_key}_mom_score", dmas)),
                "price_vs_50ma_pct": 0,  # Could compute if needed
                "price_vs_100ma_pct": 0,
                "price_vs_200ma_pct": 0,
                "rating": state.get(f"{ticker_key}_selected_view", "Neutral"),
                "subtitle": state.get(f"{ticker_key}_subtitle", ""),
            })

        # Update history for each asset
        for asset in assets_to_record:
            asset_name = asset["asset_name"]
            if asset_name not in history_data:
                history_data[asset_name] = []

            snapshot = {
                "date": data_as_of,
                "dmas": asset["dmas"],
                "technical_score": asset["technical_score"],
                "momentum_score": asset["momentum_score"],
                "price_vs_50ma_pct": asset["price_vs_50ma_pct"],
                "price_vs_100ma_pct": asset["price_vs_100ma_pct"],
                "price_vs_200ma_pct": asset["price_vs_200ma_pct"],
                "rating": asset["rating"],
                "subtitle": asset["subtitle"],
            }

            # Find and replace if same date exists
            existing_idx = None
            for i, s in enumerate(history_data[asset_name]):
                if s["date"] == data_as_of:
                    existing_idx = i
                    break

            if existing_idx is not None:
                history_data[asset_name][existing_idx] = snapshot
            else:
                history_data[asset_name].append(snapshot)

            # Sort and keep last 52 weeks
            history_data[asset_name] = sorted(
                history_data[asset_name],
                key=lambda x: x["date"]
            )[-52:]

        # Save
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)

        print(f"[CLI] Updated history for {len(assets_to_record)} assets")

    except Exception as e:
        print(f"[CLI] Warning: Could not update history: {e}")
        import traceback
        traceback.print_exc()


# ==============================================================================
# MAIN CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate IC Technical Presentation (CLI mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python cli_generate.py \\
        --excel /path/to/ic_file.xlsx \\
        --template /path/to/template.pptx \\
        --output /path/to/output/

    # With specific date and history
    python cli_generate.py \\
        --excel /path/to/ic_file.xlsx \\
        --template /path/to/template.pptx \\
        --output /path/to/output/ \\
        --history /path/to/history.json \\
        --date 2025-02-21

Environment Variables:
    ANTHROPIC_API_KEY  - For Claude subtitle generation
    CMC_API_KEY        - For crypto market caps (optional)
        """
    )

    parser.add_argument(
        "--excel", "-e",
        type=str,
        required=True,
        help="Path to consolidated Excel file (ic_file.xlsx)"
    )

    parser.add_argument(
        "--template", "-t",
        type=str,
        required=True,
        help="Path to PowerPoint template (template.pptx)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for generated PPTX"
    )

    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to history.json (optional, for previous week data)"
    )

    parser.add_argument(
        "--date",
        type=str,
        default="latest",
        help="Data as-of date: 'latest' or YYYY-MM-DD format"
    )

    parser.add_argument(
        "--price-mode",
        type=str,
        default="Last Price",
        choices=["Last Price", "PX_Mid"],
        help="Price mode for data"
    )

    parser.add_argument(
        "--skip-subtitles",
        action="store_true",
        help="Skip Claude API subtitle generation"
    )

    parser.add_argument(
        "--skip-history-update",
        action="store_true",
        help="Skip updating history.json"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate paths
    excel_path = Path(args.excel)
    template_path = Path(args.template)
    output_dir = Path(args.output)

    if not excel_path.exists():
        print(f"Error: Excel file not found: {excel_path}")
        sys.exit(1)

    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = None
    if args.history:
        history_path = Path(args.history)

    # Determine data_as_of date
    if args.date == "latest":
        data_as_of = get_max_date_from_excel(excel_path)
        print(f"[CLI] Using latest date from Excel: {data_as_of}")
    else:
        try:
            data_as_of = datetime.strptime(args.date, "%Y-%m-%d").date()
            print(f"[CLI] Using specified date: {data_as_of}")
        except ValueError:
            print(f"Error: Invalid date format: {args.date}. Use YYYY-MM-DD.")
            sys.exit(1)

    data_as_of_str = data_as_of.strftime("%Y-%m-%d")

    # Initialize state
    print("\n" + "="*60)
    print("IC TECHNICAL PRESENTATION GENERATOR (CLI)")
    print("="*60)
    print(f"Excel: {excel_path}")
    print(f"Template: {template_path}")
    print(f"Output: {output_dir}")
    print(f"Date: {data_as_of}")
    print(f"Price Mode: {args.price_mode}")
    print("="*60 + "\n")

    state: Dict[str, Any] = {
        "data_as_of": data_as_of,
        "price_mode": args.price_mode,
    }

    # Step 1: Load transition sheet
    load_transition_sheet(excel_path, state)

    # Step 2: Load history data
    if history_path:
        load_history_data(history_path, state, data_as_of)

    # Step 3: Compute technical scores
    compute_scores(excel_path, state, data_as_of)

    # Step 4: Generate subtitles
    if not args.skip_subtitles:
        generate_subtitles(excel_path, state, data_as_of_str)
        generate_ytd_recaps(excel_path, state)

    # Step 5: Generate presentation
    print("\n" + "="*60)
    print("GENERATING PRESENTATION")
    print("="*60 + "\n")

    from ic_engine import generate_presentation

    def progress_callback(msg: str):
        print(f"  {msg}")

    pptx_bytes, filename = generate_presentation(
        excel_path=excel_path,
        template_path=template_path,
        state=state,
        progress_callback=progress_callback,
    )

    # Step 6: Save output
    output_path = output_dir / filename
    with open(output_path, "wb") as f:
        f.write(pptx_bytes)

    print(f"\n[CLI] Saved: {output_path}")
    print(f"[CLI] Size: {len(pptx_bytes) / 1024 / 1024:.2f} MB")

    # Step 7: Update history
    if history_path and not args.skip_history_update:
        update_history(history_path, state, data_as_of_str)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Output: {output_path}")
    print(f"  Date: {data_as_of}")
    print(f"  Instruments: {len(ASSET_MAP)}")

    # Count generated scores/subtitles
    scores_count = sum(1 for k in state if k.endswith("_dmas"))
    subtitles_count = sum(1 for k in state if k.endswith("_subtitle") and state[k])
    print(f"  Scores computed: {scores_count}")
    print(f"  Subtitles generated: {subtitles_count}")
    print("="*60 + "\n")

    print("[CLI] Done!")


if __name__ == "__main__":
    main()
