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
    ("Bitcoin", "XBTUSD Curncy"),
    ("Ethereum", "XETUSD Curncy"),
    ("Ripple", "XRPUSD Curncy"),
    ("Solana", "XSOUSD Curncy"),
    ("Binance", "XBIUSD Curncy"),
]

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

    # Outlook backgrounds and text colors
    "outlook_bullish_bg": RGBColor(220, 252, 231),       # #DCFCE7
    "outlook_bullish_text": RGBColor(22, 101, 52),       # #166534
    "outlook_constructive_bg": RGBColor(236, 252, 203),  # #ECFCCB
    "outlook_constructive_text": RGBColor(63, 98, 18),   # #3F6212
    "outlook_neutral_bg": RGBColor(254, 249, 195),       # #FEF9C3
    "outlook_neutral_text": RGBColor(133, 77, 14),       # #854D0E
    "outlook_cautious_bg": RGBColor(255, 237, 213),      # #FFEDD5
    "outlook_cautious_text": RGBColor(194, 65, 12),      # #C2410C
    "outlook_bearish_bg": RGBColor(254, 226, 226),       # #FEE2E2
    "outlook_bearish_text": RGBColor(220, 38, 38),       # #DC2626
}

# =============================================================================
# TWO-COLUMN TABLE LAYOUT
# =============================================================================

# Column widths in inches
# Left column (Equity): wider first column for longer names
COL_WIDTHS_LEFT = [1.1, 0.55, 0.35, 0.5, 0.4, 0.65]   # Total: 3.55"
# Right column (Commodities, Crypto): slightly narrower
COL_WIDTHS_RIGHT = [0.9, 0.5, 0.35, 0.5, 0.4, 0.65]   # Total: 3.3"

# Row height
ROW_HEIGHT = 0.22  # inches

# Font sizes
HEADER_FONT_SIZE = 8
DATA_FONT_SIZE = 8
OUTLOOK_FONT_SIZE = 7

# Headers per asset class (first column name varies)
HEADERS = {
    "equity": ["Equity", "Mkt Cap", "RSI", "vs 50d", "DMAS", "Outlook"],
    "commodities": ["Commodity", "Mkt Cap", "RSI", "vs 50d", "DMAS", "Outlook"],
    "crypto": ["Crypto", "Mkt Cap", "RSI", "vs 50d", "DMAS", "Outlook"],
}

# Slide positions (in inches) - TWO COLUMN LAYOUT
SLIDE_LAYOUT = {
    # Title area
    "title_left": 0.45,
    "title_top": 0.3,

    # Left column (Equity)
    "left_table_x": 0.3,
    "left_table_y": 1.0,

    # Right column (Commodities + Crypto)
    "right_table_x": 4.1,
    "right_commo_y": 1.0,
    "right_crypto_gap": 0.15,  # Gap between commo and crypto tables

    # Footer
    "footer_y": 7.1,
}
