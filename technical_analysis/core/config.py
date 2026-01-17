"""
Configuration for Technical Analysis V2 slides.

This module contains:
- Instrument-specific configurations
- Default colors and styling
- Dimension constants
"""

from typing import Dict, List, Any

# =============================================================================
# DEFAULT STYLING
# =============================================================================

DEFAULT_COLORS = {
    "price_line": "#1B3A5A",
    "ma50": "#22C55E",      # Green
    "ma100": "#EAB308",     # Yellow/Gold
    "ma200": "#EF4444",     # Red
    "regression": "#22C55E",
    "channel_fill": "rgba(34, 197, 94, 0.15)",

    # Score colors
    "bullish": "#22C55E",
    "constructive": "#84CC16",
    "neutral": "#EAB308",
    "cautious": "#F97316",
    "bearish": "#EF4444",

    # RSI colors
    "overbought": "#F97316",
    "oversold": "#10B981",
    "rsi_neutral": "#C9A227",
}

DEFAULT_MA_PERIODS = [50, 100, 200]

# =============================================================================
# CHART DIMENSIONS
# =============================================================================

CHART_DIMENSIONS = {
    "base_width": 950,
    "base_height": 420,
    "device_scale": 4,
    "html_scale": 1,
    "lookback_days": 85,
}

# =============================================================================
# INSTRUMENT CONFIGURATIONS
# =============================================================================

INSTRUMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # EQUITIES
    # =========================================================================
    "spx": {
        "ticker": "SPX Index",
        "display_name": "S&P 500",
        "asset_class": "equity",
        "region": "us",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "spx",
        "placeholder_v2": "spx_v2",
    },
    "csi": {
        "ticker": "SHSZ300 Index",
        "display_name": "CSI 300",
        "asset_class": "equity",
        "region": "china",
        "currency": "CNY",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "csi",
        "placeholder_v2": "csi_v2",
    },
    "nikkei": {
        "ticker": "NKY Index",
        "display_name": "Nikkei 225",
        "asset_class": "equity",
        "region": "japan",
        "currency": "JPY",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "nikkei",
        "placeholder_v2": "nikkei_v2",
    },
    "tasi": {
        "ticker": "SASEIDX Index",
        "display_name": "TASI",
        "asset_class": "equity",
        "region": "saudi",
        "currency": "SAR",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "tasi",
        "placeholder_v2": "tasi_v2",
    },
    "sensex": {
        "ticker": "SENSEX Index",
        "display_name": "Sensex",
        "asset_class": "equity",
        "region": "india",
        "currency": "INR",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "sensex",
        "placeholder_v2": "sensex_v2",
    },
    "dax": {
        "ticker": "DAX Index",
        "display_name": "DAX",
        "asset_class": "equity",
        "region": "germany",
        "currency": "EUR",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "dax",
        "placeholder_v2": "dax_v2",
    },
    "smi": {
        "ticker": "SMI Index",
        "display_name": "SMI",
        "asset_class": "equity",
        "region": "switzerland",
        "currency": "CHF",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "smi",
        "placeholder_v2": "smi_v2",
    },
    "ibov": {
        "ticker": "IBOV Index",
        "display_name": "Bovespa",
        "asset_class": "equity",
        "region": "brazil",
        "currency": "BRL",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "ibov",
        "placeholder_v2": "ibov_v2",
    },
    "mexbol": {
        "ticker": "MEXBOL Index",
        "display_name": "Mexbol",
        "asset_class": "equity",
        "region": "mexico",
        "currency": "MXN",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "mexbol",
        "placeholder_v2": "mexbol_v2",
    },

    # =========================================================================
    # COMMODITIES
    # =========================================================================
    "gold": {
        "ticker": "XAU Curncy",
        "display_name": "Gold",
        "asset_class": "commodity",
        "sub_class": "precious_metal",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "gold",
        "placeholder_v2": "gold_v2",
    },
    "silver": {
        "ticker": "XAG Curncy",
        "display_name": "Silver",
        "asset_class": "commodity",
        "sub_class": "precious_metal",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "silver",
        "placeholder_v2": "silver_v2",
    },
    "platinum": {
        "ticker": "XPT Curncy",
        "display_name": "Platinum",
        "asset_class": "commodity",
        "sub_class": "precious_metal",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "platinum",
        "placeholder_v2": "platinum_v2",
    },
    "palladium": {
        "ticker": "XPD Curncy",
        "display_name": "Palladium",
        "asset_class": "commodity",
        "sub_class": "precious_metal",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "palladium",
        "placeholder_v2": "palladium_v2",
    },
    "oil": {
        "ticker": "CL1 Comdty",
        "display_name": "WTI Oil",
        "asset_class": "commodity",
        "sub_class": "energy",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "oil",
        "placeholder_v2": "oil_v2",
    },
    "copper": {
        "ticker": "HG1 Comdty",
        "display_name": "Copper",
        "asset_class": "commodity",
        "sub_class": "industrial_metal",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "copper",
        "placeholder_v2": "copper_v2",
    },

    # =========================================================================
    # CRYPTO
    # =========================================================================
    "bitcoin": {
        "ticker": "XBTUSD Curncy",
        "display_name": "Bitcoin",
        "asset_class": "crypto",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "bitcoin",
        "placeholder_v2": "bitcoin_v2",
    },
    "ethereum": {
        "ticker": "XETUSD Curncy",
        "display_name": "Ethereum",
        "asset_class": "crypto",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "ethereum",
        "placeholder_v2": "ethereum_v2",
    },
    "ripple": {
        "ticker": "XRPUSD Curncy",
        "display_name": "Ripple",
        "asset_class": "crypto",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "ripple",
        "placeholder_v2": "ripple_v2",
    },
    "solana": {
        "ticker": "XSOUSD Curncy",
        "display_name": "Solana",
        "asset_class": "crypto",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "solana",
        "placeholder_v2": "solana_v2",
    },
    "binance": {
        "ticker": "XBIUSD Curncy",
        "display_name": "Binance",
        "asset_class": "crypto",
        "currency": "USD",
        "ma_periods": [50, 100, 200],
        "show_rsi": True,
        "show_regression": True,
        "placeholder_v1": "binance",
        "placeholder_v2": "binance_v2",
    },
}

# Helper functions
def get_instrument_config(key: str) -> Dict[str, Any]:
    """Get configuration for an instrument by key."""
    return INSTRUMENT_CONFIGS.get(key.lower(), {})

def get_instruments_by_asset_class(asset_class: str) -> List[str]:
    """Get all instrument keys for a given asset class."""
    return [
        key for key, config in INSTRUMENT_CONFIGS.items()
        if config.get("asset_class") == asset_class
    ]
