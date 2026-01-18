"""Configuration for Technical Analysis slide generator."""

from pptx.dml.color import RGBColor

# =============================================================================
# ASSET LISTS
# =============================================================================

EQUITY_ASSETS = [
    ("S&P 500", "SPX Index"),
    ("CSI 300", "SHSZ300 Index"),
    ("Nikkei 225", "NKY Index"),
    ("TASI", "SASEIDX Index"),
    ("Sensex", "SENSEX Index"),
    ("DAX", "DAX Index"),
    ("SMI", "SMI Index"),
    ("Mexbol", "MEXBOL Index"),
    ("IBOV", "IBOV Index"),
]

COMMO_ASSETS = [
    ("Gold", "GCA Comdty"),
    ("Silver", "SIA Comdty"),
    ("Platinum", "XPT Comdty"),
    ("Palladium", "XPD Curncy"),
    ("Oil (WTI)", "CL1 Comdty"),
    ("Copper", "LP1 Comdty"),
]

CRYPTO_ASSETS = [
    ("Bitcoin", "XBTUSD Curncy", "btc"),
    ("Ethereum", "XETUSD Curncy", "eth"),
    ("Ripple", "XRPUSD Curncy", "xrp"),
    ("Solana", "XSOUSD Curncy", "sol"),
    ("Binance", "XBIUSD Curncy", "bnb"),
]

# =============================================================================
# COMMODITY RESERVES (for market cap calculation)
# Market Cap = Above-Ground Reserves × Current Price
# =============================================================================

# Above-ground reserves in TONNES
# Sources: World Gold Council, Silver Institute, WPIC
COMMODITY_RESERVES_TONNES = {
    "Gold": 215_000,          # World Gold Council estimate
    "Silver": 1_700_000,      # Silver Institute (above-ground)
    "Platinum": 8_000,        # World Platinum Investment Council
    "Palladium": 10_000,      # Industry estimates
    "Copper": None,           # Not applicable (industrial commodity)
    "Oil (WTI)": None,        # Not applicable (consumption commodity)
}

# Conversion: 1 tonne = 32,150.7 troy ounces
TROY_OZ_PER_TONNE = 32_150.7

# Bloomberg tickers for commodity prices (per troy oz)
# Same tickers used in COMMO_ASSETS
COMMODITY_SPOT_TICKERS = {
    "Gold": "GCA Comdty",
    "Silver": "SIA Comdty",
    "Platinum": "XPT Comdty",
    "Palladium": "XPD Curncy",
}

# =============================================================================
# COLOR DEFINITIONS - Clean white/grey style
# =============================================================================

COLORS = {
    # Header styling
    "header_bg": RGBColor(27, 58, 90),       # #1B3A5A - Navy blue
    "header_text": RGBColor(255, 255, 255),  # White

    # Data row backgrounds (alternating)
    "row_white": RGBColor(255, 255, 255),    # White
    "row_grey": RGBColor(248, 248, 248),     # Very light grey #F8F8F8

    # Value colors
    "positive": RGBColor(22, 163, 74),       # #16A34A - Green
    "negative": RGBColor(220, 38, 38),       # #DC2626 - Red
    "neutral_text": RGBColor(26, 26, 46),    # #1A1A2E - Dark
    "gold_accent": RGBColor(201, 162, 39),   # #C9A227 - Gold
    "gray_text": RGBColor(102, 102, 102),    # #666666 - Gray
    "light_gray": RGBColor(136, 136, 136),   # #888888 - Light gray

    # Outlook backgrounds and text colors - MORE EXPRESSIVE
    "outlook_bullish_bg": RGBColor(187, 247, 208),       # Brighter green
    "outlook_bullish_text": RGBColor(20, 83, 45),        # Dark green

    "outlook_constructive_bg": RGBColor(217, 249, 157),  # Bright lime
    "outlook_constructive_text": RGBColor(54, 83, 20),   # Dark lime

    "outlook_neutral_bg": RGBColor(254, 240, 138),       # Bright yellow
    "outlook_neutral_text": RGBColor(113, 63, 18),       # Dark amber

    "outlook_cautious_bg": RGBColor(254, 215, 170),      # Bright orange
    "outlook_cautious_text": RGBColor(154, 52, 18),      # Dark orange

    "outlook_bearish_bg": RGBColor(254, 202, 202),       # Bright red
    "outlook_bearish_text": RGBColor(153, 27, 27),       # Dark red
}

# =============================================================================
# TWO-COLUMN TABLE LAYOUT - EXACT CENTIMETER DIMENSIONS
# =============================================================================

# Table dimensions in centimeters (from manual testing)
TABLE_DIMS = {
    "equity": {
        "left": 1.13,
        "top": 5.67,
        "width": 11.91,
        "height": 8.53,
        "rows": 10,  # 1 header + 9 data rows
        "header_height": 0.68,
        "col_widths": [2.5, 1.8, 1.0, 1.3, 1.1, 2.0],  # Will be normalized
    },
    "commodities": {
        "left": 13.25,
        "top": 5.69,
        "width": 11.91,
        "height": 4.45,
        "rows": 7,  # 1 header + 6 data rows
        "header_height": 0.68,
        "col_widths": [2.3, 1.6, 1.0, 1.4, 1.1, 2.0],
    },
    "crypto": {
        "left": 13.25,
        "top": 10.41,
        "width": 11.91,
        "height": 3.81,
        "rows": 6,  # 1 header + 5 data rows
        "header_height": 0.68,
        "col_widths": [2.0, 1.6, 1.0, 1.4, 1.1, 2.0],
    },
}

# Font sizes - CRITICAL: Must be explicitly set on both paragraph AND runs
FONT_SIZE = 9       # All data cells
FONT_SIZE_HEADER = 9
FONT_SIZE_OUTLOOK = 8

# Headers per asset class (first column name varies)
HEADERS = {
    "equity": ["Equity", "Mkt Cap", "RSI", "vs 50d", "DMAS", "Outlook"],
    "commodities": ["Commodity", "Mkt Cap", "RSI", "vs 50d", "DMAS", "Outlook"],
    "crypto": ["Crypto", "Mkt Cap", "RSI", "vs 50d", "DMAS", "Outlook"],
}

# Title and footer positions in centimeters
SLIDE_LAYOUT = {
    "title_left": 1.5,
    "title_top": 1.5,
    "subtitle_top": 3.0,
    "footer_y": 18.0,
}


# ============================================================
# CONFIG VALIDATION TEST
# ============================================================
if __name__ == "__main__":
    print("=== TECHNICAL SLIDE CONFIG TEST ===")
    print(f"\nHEADERS keys: {list(HEADERS.keys())}")
    for k, v in HEADERS.items():
        print(f"  {k}: {len(v)} items - {v}")

    print(f"\nTABLE_DIMS keys: {list(TABLE_DIMS.keys())}")
    for k, v in TABLE_DIMS.items():
        print(f"  {k}: left={v['left']}, top={v['top']}, col_widths={v['col_widths']}")

    # Validation checks
    errors = []
    if len(HEADERS) != 3:
        errors.append(f"HEADERS should have 3 keys, has {len(HEADERS)}")
    if len(TABLE_DIMS) != 3:
        errors.append(f"TABLE_DIMS should have 3 keys, has {len(TABLE_DIMS)}")

    for asset_class in ["equity", "commodities", "crypto"]:
        if asset_class not in HEADERS:
            errors.append(f"HEADERS missing '{asset_class}'")
        elif len(HEADERS[asset_class]) != 6:
            errors.append(f"HEADERS['{asset_class}'] should have 6 items, has {len(HEADERS[asset_class])}")

        if asset_class not in TABLE_DIMS:
            errors.append(f"TABLE_DIMS missing '{asset_class}'")
        elif len(TABLE_DIMS[asset_class].get("col_widths", [])) != 6:
            errors.append(f"TABLE_DIMS['{asset_class}']['col_widths'] should have 6 items")

    if errors:
        print("\n❌ CONFIG ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n✅ Config looks correct!")
