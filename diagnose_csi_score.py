#!/usr/bin/env python3
"""
Diagnostic script to compare CSI scoring methods and identify discrepancies.
"""

import pandas as pd
import numpy as np
from mars_engine import (
    generate_csi_score_history,
    load_prices_for_mars,
    get_csi_lasso_score,
    DEFAULT_WEIGHTS,
)

def diagnose_csi_scoring(excel_path: str):
    """
    Compare different CSI scoring methods to identify the discrepancy.
    """
    print("=" * 80)
    print("CSI MARS Score Diagnostic")
    print("=" * 80)

    # Load data
    print("\n1. Loading price data...")
    prices_df = load_prices_for_mars(excel_path)
    print(f"   Data shape: {prices_df.shape}")
    print(f"   Date range: {prices_df.index.min()} to {prices_df.index.max()}")

    # Check CSI columns
    print("\n2. Checking CSI columns...")
    csi_cols = [col for col in prices_df.columns if 'CSI' in col]
    print(f"   CSI columns: {csi_cols}")

    # Test different aggregation methods
    print("\n3. Computing scores with different methods...")

    try:
        # Method 1: top2 (original, like SPX)
        score_top2 = generate_csi_score_history(prices_df, agg_method="top2")
        latest_top2 = score_top2.iloc[-1] if not score_top2.empty else None
        print(f"   ✓ top2:     {latest_top2:.1f}" if latest_top2 else "   ✗ top2: Failed")
    except Exception as e:
        print(f"   ✗ top2: Error - {e}")
        latest_top2 = None

    try:
        # Method 2: top3
        score_top3 = generate_csi_score_history(prices_df, agg_method="top3")
        latest_top3 = score_top3.iloc[-1] if not score_top3.empty else None
        print(f"   ✓ top3:     {latest_top3:.1f}" if latest_top3 else "   ✗ top3: Failed")
    except Exception as e:
        print(f"   ✗ top3: Error - {e}")
        latest_top3 = None

    try:
        # Method 3: weighted with DEFAULT_WEIGHTS
        score_weighted = generate_csi_score_history(prices_df, agg_method="weighted")
        latest_weighted = score_weighted.iloc[-1] if not score_weighted.empty else None
        print(f"   ✓ weighted: {latest_weighted:.1f}" if latest_weighted else "   ✗ weighted: Failed")
        print(f"      (using DEFAULT_WEIGHTS: pure=0.40, smooth=0.20, sharpe=0.20, idio=0.10, adx=0.10)")
    except Exception as e:
        print(f"   ✗ weighted: Error - {e}")
        latest_weighted = None

    try:
        # Method 4: LASSO (dynamic weights)
        print("\n4. Computing LASSO score (this may take a moment)...")
        latest_lasso = get_csi_lasso_score(excel_path)
        print(f"   ✓ LASSO:    {latest_lasso:.1f}" if latest_lasso else "   ✗ LASSO: Failed")
    except Exception as e:
        print(f"   ✗ LASSO: Error - {e}")
        latest_lasso = None

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Top2:     {latest_top2:.1f}" if latest_top2 else "Top2:     N/A")
    print(f"Top3:     {latest_top3:.1f}" if latest_top3 else "Top3:     N/A")
    print(f"Weighted: {latest_weighted:.1f}" if latest_weighted else "Weighted: N/A")
    print(f"LASSO:    {latest_lasso:.1f}" if latest_lasso else "LASSO:    N/A")
    print("\nExpected from MARS app: 32")
    print("=" * 80)

    # Determine which is closest
    if all([latest_top2, latest_top3, latest_weighted, latest_lasso]):
        scores = {
            "top2": latest_top2,
            "top3": latest_top3,
            "weighted": latest_weighted,
            "lasso": latest_lasso,
        }
        expected = 32
        closest = min(scores.items(), key=lambda x: abs(x[1] - expected))
        print(f"\n✓ Closest to expected (32): {closest[0]} = {closest[1]:.1f}")
        print(f"  Difference: {abs(closest[1] - expected):.1f} points")

    return {
        "top2": latest_top2,
        "top3": latest_top3,
        "weighted": latest_weighted,
        "lasso": latest_lasso,
    }


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
        import pathlib
        for candidate in excel_candidates:
            if pathlib.Path(candidate).exists():
                excel_path = candidate
                break

    if excel_path is None:
        print("ERROR: Could not find Excel file. Please provide path as argument.")
        print("Usage: python diagnose_csi_score.py <path_to_excel>")
        sys.exit(1)

    print(f"Using Excel file: {excel_path}\n")
    results = diagnose_csi_scoring(excel_path)
