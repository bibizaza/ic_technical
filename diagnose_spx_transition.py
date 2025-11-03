#!/usr/bin/env python3
"""
Diagnostic script to debug SPX transition data and chart issues.

Run this to check:
1. Is transition sheet being read correctly?
2. What values are being extracted for SPX?
3. What is PLOT_LOOKBACK_DAYS set to?
"""

import sys
import pandas as pd
from pathlib import Path

# Excel file path
excel_path = "data/data.xlsx"

print("=" * 80)
print("SPX TRANSITION DATA DIAGNOSTIC")
print("=" * 80)
print(f"Excel file: {excel_path}\n")

# Test 1: Read transition sheet directly
print("=" * 80)
print("TEST 1: Reading transition sheet directly")
print("=" * 80)

try:
    df = pd.read_excel(excel_path, sheet_name="transition")
    print(f"✅ Successfully read transition sheet")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}\n")

    print("First 5 rows (raw):")
    print(df.head())
    print()

    # Check if header row needs to be skipped
    if len(df) > 1:
        print("After skipping header row (row 0):")
        df_data = df.iloc[1:]
        print(df_data.head())
        print()

        # Check SPX specifically
        print("=" * 80)
        print("TEST 2: Finding SPX Index in transition sheet")
        print("=" * 80)

        spx_found = False
        for idx, row in df_data.iterrows():
            ticker = row.iloc[0] if len(row) > 0 else None
            if pd.notna(ticker):
                ticker_str = str(ticker).strip().upper()
                if "SPX" in ticker_str:
                    print(f"✅ Found SPX at row {idx}: '{ticker}'")
                    print(f"   Raw row data: {row.tolist()}")

                    # Extract values
                    last_week_dmas = row.iloc[1] if len(row) > 1 else None
                    anchor_date = row.iloc[2] if len(row) > 2 else None
                    assessment = row.iloc[3] if len(row) > 3 else None
                    subtitle = row.iloc[4] if len(row) > 4 else None

                    print(f"   Last Week DMAS (Column B): {last_week_dmas}")
                    print(f"   Anchor Date (Column C): {anchor_date}")
                    print(f"   Assessment (Column D): {assessment}")
                    print(f"   Subtitle (Column E): {subtitle}")
                    spx_found = True
                    break

        if not spx_found:
            print("❌ SPX not found in transition sheet!")
            print("\nAll tickers found:")
            for idx, row in df_data.iterrows():
                ticker = row.iloc[0] if len(row) > 0 else None
                if pd.notna(ticker):
                    print(f"   Row {idx}: '{ticker}'")

except Exception as e:
    print(f"❌ ERROR reading transition sheet: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Use transition_loader module
print("\n" + "=" * 80)
print("TEST 3: Using transition_loader module")
print("=" * 80)

try:
    from transition_loader import read_transition_sheet, get_ticker_key_from_ticker

    transition_data = read_transition_sheet(excel_path)

    if not transition_data:
        print("❌ transition_loader returned empty dict!")
    else:
        print(f"✅ transition_loader found {len(transition_data)} tickers")
        print(f"   Tickers: {list(transition_data.keys())}\n")

        # Check for SPX variations
        spx_found = False
        for ticker in ["SPX INDEX", "SPX", "SPX Index"]:
            if ticker in transition_data:
                print(f"✅ Found SPX as '{ticker}':")
                print(f"   {transition_data[ticker]}")

                # Check ticker_key mapping
                ticker_key = get_ticker_key_from_ticker(ticker)
                print(f"   Maps to ticker_key: '{ticker_key}'")
                spx_found = True
                break

        if not spx_found:
            print("❌ SPX not found in transition_data!")
            print("   Available tickers:", list(transition_data.keys()))

except Exception as e:
    print(f"❌ ERROR using transition_loader: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check PLOT_LOOKBACK_DAYS
print("\n" + "=" * 80)
print("TEST 4: Checking PLOT_LOOKBACK_DAYS")
print("=" * 80)

try:
    import technical_analysis.equity.spx as spx_module
    print(f"Initial PLOT_LOOKBACK_DAYS: {spx_module.PLOT_LOOKBACK_DAYS}")

    # Simulate what app.py does
    spx_module.PLOT_LOOKBACK_DAYS = 365
    print(f"After setting to 365: {spx_module.PLOT_LOOKBACK_DAYS}")

    spx_module.PLOT_LOOKBACK_DAYS = 90
    print(f"After setting to 90: {spx_module.PLOT_LOOKBACK_DAYS}")

    print("✅ PLOT_LOOKBACK_DAYS can be modified correctly")

except Exception as e:
    print(f"❌ ERROR with PLOT_LOOKBACK_DAYS: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("If you see errors above, that's the source of your problem.")
print("If all tests pass, the issue may be in Streamlit session state or caching.")
print("=" * 80)
