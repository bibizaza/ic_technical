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
    "row_grey": RGBColor(245, 245, 245),     # Light grey #F5F5F5

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
# TABLE LAYOUT - COMPACT
# =============================================================================

# Column widths in inches (compact)
COLUMN_WIDTHS = [1.0, 0.6, 0.4, 0.6, 0.5, 0.7]  # Total ~3.8 inches

# Headers per asset class
HEADERS = {
    "equity": ["Index", "Mkt Cap", "RSI", "vs 50d", "DMAS", "Outlook"],
    "commodities": ["Commodity", "Mkt Cap", "RSI", "vs 50d", "DMAS", "Outlook"],
    "crypto": ["Asset", "Mkt Cap", "RSI", "vs 50d", "DMAS", "Outlook"],
}

# Slide positions (in inches) - COMPACT layout
SLIDE_LAYOUT = {
    "title_left": 0.4,
    "title_top": 0.2,
    "table_width": 3.8,
    "row_height": 0.22,           # Compact row height
    "header_font_size": 8,        # Smaller header font
    "data_font_size": 9,          # Smaller data font
    "equity_top": 0.85,           # After title + section label
    "commodities_top": 3.35,      # After equity table
    "crypto_top": 5.15,           # After commodities table
    "footer_top": 6.6,
}
