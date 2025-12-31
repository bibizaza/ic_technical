"""Data preparation for Technical Analysis slide."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .config import EQUITY_ASSETS, COMMO_ASSETS, CRYPTO_ASSETS
from .market_caps import (
    get_equity_market_caps,
    COMMO_MARKET_CAPS,
    CRYPTO_MARKET_CAPS,
)


@dataclass
class AssetRow:
    """Data structure for a single row in the technical analysis table."""
    name: str
    ticker: str
    market_cap: str
    rsi: int
    vs_50d_ma: float
    dmas: int
    outlook: str
    asset_class: str  # "equity", "commodities", "crypto"


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index).

    Parameters
    ----------
    prices : pd.Series
        Price series (most recent last)
    period : int
        RSI period (default 14)

    Returns
    -------
    float
        RSI value (0-100)
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral if insufficient data

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)

    # Calculate average gains and losses (Wilder's smoothing)
    avg_gain = gains.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Get the last valid value
    last_rsi = rsi.iloc[-1]
    if pd.isna(last_rsi):
        return 50.0

    return round(last_rsi, 0)


def calculate_vs_ma(prices: pd.Series, ma_period: int = 50) -> float:
    """
    Calculate percentage distance from moving average.

    Parameters
    ----------
    prices : pd.Series
        Price series (most recent last)
    ma_period : int
        Moving average period (default 50)

    Returns
    -------
    float
        Percentage above/below MA (e.g., +2.17 or -3.40)
    """
    if len(prices) < ma_period:
        return 0.0

    current_price = prices.iloc[-1]
    ma = prices.rolling(window=ma_period).mean().iloc[-1]

    if pd.isna(ma) or ma == 0:
        return 0.0

    pct_vs_ma = ((current_price - ma) / ma) * 100
    return round(pct_vs_ma, 2)


def get_outlook(dmas: int) -> str:
    """
    Derive outlook from DMAS score.

    Parameters
    ----------
    dmas : int
        DMAS score (0-100)

    Returns
    -------
    str
        Outlook rating
    """
    if dmas >= 70:
        return "Bullish"
    elif dmas >= 55:
        return "Constructive"
    elif dmas >= 45:
        return "Neutral"
    elif dmas >= 30:
        return "Cautious"
    else:
        return "Bearish"


def _get_price_series(
    prices_df: pd.DataFrame,
    ticker: str,
    date_col: str = "Date"
) -> Optional[pd.Series]:
    """
    Extract price series for a specific ticker from DataFrame.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with Date column and price columns
    ticker : str
        Bloomberg ticker to extract
    date_col : str
        Name of date column

    Returns
    -------
    pd.Series or None
        Price series if found, None otherwise
    """
    # Try exact match first
    if ticker in prices_df.columns:
        series = prices_df[ticker].dropna()
        return series

    # Try case-insensitive match
    for col in prices_df.columns:
        if col.lower() == ticker.lower():
            series = prices_df[col].dropna()
            return series

    # Try partial match
    for col in prices_df.columns:
        if ticker.split()[0].lower() in col.lower():
            series = prices_df[col].dropna()
            return series

    return None


def prepare_slide_data(
    prices_df: pd.DataFrame,
    dmas_scores: Dict[str, int],
    excel_path: str,
    price_mode: str = "Last Price",
) -> List[AssetRow]:
    """
    Prepare all data for the Technical Analysis slide.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Price data with Date column and price columns for each asset
    dmas_scores : Dict[str, int]
        Mapping of ticker_key -> DMAS score (from session state)
    excel_path : str
        Path to Excel file (for market cap data)
    price_mode : str
        "Last Price" or "Last Close"

    Returns
    -------
    List[AssetRow]
        List of prepared asset rows for the slide
    """
    rows = []

    # Get market caps
    equity_mkt_caps = get_equity_market_caps(excel_path)

    # Helper to get DMAS from various key formats
    def get_dmas(ticker_key: str, display_name: str) -> int:
        # Try various key formats
        for key in [ticker_key, ticker_key.lower(), display_name, display_name.lower()]:
            if key in dmas_scores:
                return dmas_scores[key]
        return 50  # Default neutral

    # EQUITY
    for display_name, ticker in EQUITY_ASSETS:
        prices = _get_price_series(prices_df, ticker)
        if prices is None or len(prices) < 50:
            continue

        ticker_key = display_name.lower().replace(" ", "_").replace("&", "")
        # Map common variations
        key_map = {
            "s_p_500": "spx",
            "csi_300": "csi",
            "nikkei_225": "nikkei",
            "dax": "dax",
        }
        ticker_key = key_map.get(ticker_key, ticker_key.split("_")[0] if "_" in ticker_key else ticker_key)

        dmas = get_dmas(ticker_key, display_name)

        rows.append(AssetRow(
            name=display_name,
            ticker=ticker,
            market_cap=equity_mkt_caps.get(display_name, "—"),
            rsi=int(calculate_rsi(prices)),
            vs_50d_ma=calculate_vs_ma(prices, 50),
            dmas=dmas,
            outlook=get_outlook(dmas),
            asset_class="equity"
        ))

    # COMMODITIES
    for display_name, ticker in COMMO_ASSETS:
        prices = _get_price_series(prices_df, ticker)
        if prices is None or len(prices) < 50:
            continue

        ticker_key = display_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        # Map common variations
        key_map = {
            "oil_wti": "oil",
        }
        ticker_key = key_map.get(ticker_key, ticker_key)

        dmas = get_dmas(ticker_key, display_name)

        rows.append(AssetRow(
            name=display_name,
            ticker=ticker,
            market_cap=COMMO_MARKET_CAPS.get(display_name, "—"),
            rsi=int(calculate_rsi(prices)),
            vs_50d_ma=calculate_vs_ma(prices, 50),
            dmas=dmas,
            outlook=get_outlook(dmas),
            asset_class="commodities"
        ))

    # CRYPTO
    for display_name, ticker in CRYPTO_ASSETS:
        prices = _get_price_series(prices_df, ticker)
        if prices is None or len(prices) < 50:
            continue

        ticker_key = display_name.lower()
        dmas = get_dmas(ticker_key, display_name)

        rows.append(AssetRow(
            name=display_name,
            ticker=ticker,
            market_cap=CRYPTO_MARKET_CAPS.get(display_name, "—"),
            rsi=int(calculate_rsi(prices)),
            vs_50d_ma=calculate_vs_ma(prices, 50),
            dmas=dmas,
            outlook=get_outlook(dmas),
            asset_class="crypto"
        ))

    return rows
