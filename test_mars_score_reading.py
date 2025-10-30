#!/usr/bin/env python3
"""
Test script to verify mars_score sheet is being read correctly from Excel.

Run this to check:
1. Does mars_score sheet exist?
2. What tickers are in it?
3. What scores are in it?
4. Can we read SPX and CSI scores?
"""

import sys
from mars_engine.data_loader import load_mars_scores

# Excel file path
excel_path = "data/data.xlsx"

print("=" * 80)
print("TESTING MARS SCORE READING FROM EXCEL")
print("=" * 80)
print(f"Excel file: {excel_path}\n")

try:
    # Load all MARS scores
    print("📊 Loading mars_score sheet...")
    mars_scores = load_mars_scores(excel_path)

    if not mars_scores:
        print("❌ ERROR: No scores found!")
        print("\nPossible issues:")
        print("  1. mars_score sheet does not exist in Excel")
        print("  2. Sheet is empty")
        print("  3. Sheet has wrong format (should be: Column A=Ticker, Column B=Mars)")
        sys.exit(1)

    print(f"✅ Successfully loaded {len(mars_scores)} tickers\n")

    # Display all scores
    print("=" * 80)
    print("ALL MARS SCORES IN EXCEL")
    print("=" * 80)
    for ticker, score in sorted(mars_scores.items()):
        print(f"  {ticker:20s} → {score:6.2f}")
    print()

    # Check for SPX
    print("=" * 80)
    print("CHECKING SPX")
    print("=" * 80)
    spx_found = False
    for ticker_name in ["SPX", "SPX Index", "S&P 500"]:
        if ticker_name in mars_scores:
            print(f"✅ Found as '{ticker_name}': {mars_scores[ticker_name]:.2f}")
            spx_found = True
            break

    if not spx_found:
        print("❌ SPX not found in mars_score sheet")
        print("   Available tickers:", ", ".join(sorted(mars_scores.keys())))
    print()

    # Check for CSI
    print("=" * 80)
    print("CHECKING CSI")
    print("=" * 80)
    csi_found = False
    for ticker_name in ["CSI", "CSI 300", "SHSZ300", "SHSZ300 Index"]:
        if ticker_name in mars_scores:
            print(f"✅ Found as '{ticker_name}': {mars_scores[ticker_name]:.2f}")
            csi_found = True
            break

    if not csi_found:
        print("❌ CSI not found in mars_score sheet")
        print("   Available tickers:", ", ".join(sorted(mars_scores.keys())))
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if spx_found and csi_found:
        print("✅ SUCCESS: Both SPX and CSI scores found and readable!")
        print("\nYour ic_technical app should now display these scores.")
        print("If not, clear the Streamlit cache (press 'c' or restart app).")
    else:
        print("⚠️  INCOMPLETE: Some scores missing from mars_score sheet")
        print("\nPlease ensure mars_score sheet has:")
        print("  Column A (header: 'Ticker'): SPX, CSI, etc.")
        print("  Column B (header: 'Mars'): 95.5, 32.0, etc.")
    print("=" * 80)

except FileNotFoundError:
    print(f"❌ ERROR: Excel file not found at: {excel_path}")
    print("\nPlease update the excel_path variable in this script.")
    sys.exit(1)

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
