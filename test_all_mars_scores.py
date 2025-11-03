#!/usr/bin/env python3
"""
Test that all asset classes (equity, commodities, crypto) can read
their MARS scores from the mars_score sheet.
"""

from mars_engine.data_loader import load_mars_scores

# Excel file path
excel_path = "data/data.xlsx"

# Define all tickers used by the app
TICKERS = {
    "Equity": [
        ("SPX INDEX", "spx"),
        ("SHSZ300 INDEX", "csi"),
        ("NKY INDEX", "nikkei"),
        ("SASEIDX INDEX", "tasi"),
        ("SENSEX INDEX", "sensex"),
        ("DAX INDEX", "dax"),
        ("SMI INDEX", "smi"),
        ("IBOV INDEX", "ibov"),
        ("MEXBOL INDEX", "mexbol"),
    ],
    "Commodities": [
        ("GCA COMDTY", "gold"),
        ("SIA COMDTY", "silver"),
        ("XPT COMDTY", "platinum"),
        ("XPD CURNCY", "palladium"),
        ("CL1 COMDTY", "oil"),
        ("LP1 COMDTY", "copper"),
    ],
    "Crypto": [
        ("XBTUSD CURNCY", "bitcoin"),
        ("XETUSD CURNCY", "ethereum"),
        ("XRPUSD CURNCY", "ripple"),
        ("XSOUSD CURNCY", "solana"),
        ("XBIUSD CURNCY", "binance"),
    ],
}

print("=" * 80)
print("TESTING ALL MOMENTUM SCORES FROM mars_score SHEET")
print("=" * 80)
print(f"Excel file: {excel_path}\n")

# Load MARS scores
print("Loading mars_score sheet...")
mars_scores = load_mars_scores(excel_path)

if not mars_scores:
    print("❌ ERROR: Could not load mars_score sheet or it's empty!")
    exit(1)

print(f"✅ Loaded {len(mars_scores)} scores\n")

# Test each category
all_found = True

for category, tickers in TICKERS.items():
    print("=" * 80)
    print(f"{category.upper()}")
    print("=" * 80)

    for excel_ticker, app_name in tickers:
        found = False
        score = None

        # Try exact match
        ticker_upper = excel_ticker.strip().upper()
        for key, s in mars_scores.items():
            if key.upper() == ticker_upper:
                found = True
                score = s
                break

        # Try partial match if exact match failed
        if not found:
            ticker_parts = ticker_upper.split()
            if ticker_parts:
                first_part = ticker_parts[0]
                for key, s in mars_scores.items():
                    if first_part in key.upper():
                        found = True
                        score = s
                        break

        if found:
            print(f"✅ {app_name:15s} ({excel_ticker:20s}) → {score:6.2f}")
        else:
            print(f"❌ {app_name:15s} ({excel_ticker:20s}) → NOT FOUND!")
            all_found = False

    print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if all_found:
    print("✅ SUCCESS: All tickers found in mars_score sheet!")
    print("\nYour ic_technical app should now display MARS scores for:")
    print("  - All equity indices (SPX, CSI, etc.)")
    print("  - All commodities (Gold, Silver, Oil, etc.)")
    print("  - All cryptocurrencies (Bitcoin, Ethereum, etc.)")
else:
    print("⚠️  INCOMPLETE: Some tickers missing from mars_score sheet")
    print("\nPlease add missing tickers to mars_score sheet:")
    print("  Column A: Ticker (e.g., 'GCA COMDTY', 'XBTUSD CURNCY')")
    print("  Column B: MARS score (e.g., 68.5, 29.8)")

print("=" * 80)
