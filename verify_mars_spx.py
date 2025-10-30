#!/usr/bin/env python3
"""
Diagnostic script to verify MARS score calculation for SPX.
"""

import pandas as pd
import pathlib
from mars_engine import generate_spx_score_history, load_prices_for_mars
from mars_engine.mars_lite_scorer import PEER_GROUP_SPX

def verify_spx_mars_score(excel_path: str):
    """
    Verify MARS score calculation and show diagnostic information.
    """
    print("=" * 80)
    print("MARS SPX Score Verification")
    print("=" * 80)

    # Load data
    print("\n1. Loading price data...")
    prices_df = load_prices_for_mars(excel_path)
    print(f"   Data shape: {prices_df.shape}")
    print(f"   Date range: {prices_df.index.min()} to {prices_df.index.max()}")
    print(f"   Total rows: {len(prices_df)}")

    # Check which columns are present
    print("\n2. Checking SPX columns...")
    spx_cols = [col for col in prices_df.columns if 'SPX' in col]
    print(f"   SPX columns found: {spx_cols}")

    # Check for high/low data
    has_spx_high = "SPX_high" in prices_df.columns
    has_spx_low = "SPX_low" in prices_df.columns
    print(f"   Has SPX_high: {has_spx_high}")
    print(f"   Has SPX_low: {has_spx_low}")

    if has_spx_high and has_spx_low:
        # Check if they're approximated (±1%)
        spx_val = prices_df["SPX"].iloc[-1]
        high_val = prices_df["SPX_high"].iloc[-1]
        low_val = prices_df["SPX_low"].iloc[-1]

        expected_high = spx_val * 1.01
        expected_low = spx_val * 0.99

        is_approximated = (abs(high_val - expected_high) < 0.01 and
                          abs(low_val - expected_low) < 0.01)
        print(f"   High/Low appears to be ±1% approximation: {is_approximated}")

    # Check peer availability
    print("\n3. Checking peer availability...")
    available_peers = [p for p in PEER_GROUP_SPX if p in prices_df.columns]
    missing_peers = [p for p in PEER_GROUP_SPX if p not in prices_df.columns]

    print(f"   Available peers: {len(available_peers)}/{len(PEER_GROUP_SPX)}")
    print(f"   Available: {available_peers}")
    if missing_peers:
        print(f"   Missing: {missing_peers}")

    # Generate MARS score
    print("\n4. Computing MARS score...")
    score_series = generate_spx_score_history(prices_df)

    if score_series is not None and not score_series.empty:
        latest_score = float(score_series.iloc[-1])
        latest_date = score_series.index[-1]

        print(f"   Latest MARS score: {latest_score:.2f}")
        print(f"   Score date: {latest_date}")
        print(f"   Score history length: {len(score_series)} days")

        # Show recent history
        print("\n5. Recent score history (last 10 days):")
        recent = score_series.tail(10)
        for date, score in recent.items():
            print(f"   {date.date()}: {score:.2f}")

        # Check benchmark used
        print("\n6. Benchmark information:")
        bench_candidates = ["MXWO Index", "MXWO", "SPX"]
        for bench in bench_candidates:
            if bench in prices_df.columns:
                print(f"   {bench}: Available ✓ (likely used as benchmark)")
                break
        else:
            print(f"   No benchmark found, using SPX itself")

        print("\n" + "=" * 80)
        print(f"FINAL SCORE: {latest_score:.2f}")
        print("=" * 80)

        return latest_score
    else:
        print("   ERROR: Could not compute MARS score!")
        return None


if __name__ == "__main__":
    import sys

    # Try to find the Excel file
    excel_candidates = [
        "/home/user/ic_technical/data.xlsx",
        "/home/user/ic_technical/data/data.xlsx",
        "/home/user/ic_technical/IC_data.xlsx",
    ]

    excel_path = None
    if len(sys.argv) > 1:
        excel_path = sys.argv[1]
    else:
        for candidate in excel_candidates:
            if pathlib.Path(candidate).exists():
                excel_path = candidate
                break

    if excel_path is None:
        print("ERROR: Could not find Excel file. Please provide path as argument.")
        print("Usage: python verify_mars_spx.py <path_to_excel>")
        sys.exit(1)

    print(f"Using Excel file: {excel_path}\n")
    verify_spx_mars_score(excel_path)
