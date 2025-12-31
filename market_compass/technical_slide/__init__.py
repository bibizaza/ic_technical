"""Technical Analysis slide generator for Market Compass.

Generates the "Technical Analysis In A Nutshell" slide with:
- Equity (9 assets)
- Commodities (6 assets)
- Crypto (5 assets)

Columns: Asset Name, Mkt Cap, RSI (14), vs 50d MA, DMAS, Outlook
"""

from .data_prep import (
    calculate_rsi,
    calculate_vs_ma,
    get_outlook,
    prepare_slide_data,
    AssetRow,
)
from .slide_generator import (
    generate_technical_analysis_slide,
    insert_technical_analysis_slide,
)
from .market_caps import (
    get_equity_market_caps,
    COMMO_MARKET_CAPS,
    CRYPTO_MARKET_CAPS,
    format_market_cap,
)

__all__ = [
    "calculate_rsi",
    "calculate_vs_ma",
    "get_outlook",
    "prepare_slide_data",
    "AssetRow",
    "generate_technical_analysis_slide",
    "insert_technical_analysis_slide",
    "get_equity_market_caps",
    "COMMO_MARKET_CAPS",
    "CRYPTO_MARKET_CAPS",
    "format_market_cap",
]
