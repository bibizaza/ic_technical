"""
Technical Indicators for V2 Charts.

This module provides shared calculations for:
- RSI (Relative Strength Index)
- Fibonacci retracement levels
- Score status interpretation
- Moving averages
"""

import pandas as pd
from typing import List, Tuple, Optional


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI (Relative Strength Index) for a price series.

    Parameters
    ----------
    prices : pd.Series
        Price series (typically closing prices)
    period : int
        RSI calculation period (default: 14)

    Returns
    -------
    pd.Series
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_fibonacci_levels(high: float, low: float) -> List[dict]:
    """
    Compute Fibonacci retracement levels.

    Parameters
    ----------
    high : float
        Recent high price
    low : float
        Recent low price

    Returns
    -------
    List[dict]
        List of Fibonacci levels with value and label
    """
    diff = high - low
    levels = [
        {"value": high, "label": "0%"},
        {"value": high - diff * 0.236, "label": "23.6%"},
        {"value": high - diff * 0.382, "label": "38.2%"},
        {"value": high - diff * 0.5, "label": "50%"},
        {"value": high - diff * 0.618, "label": "61.8%"},
        {"value": low, "label": "100%"},
    ]
    return levels


def get_score_status(score: int) -> Tuple[str, str]:
    """
    Get status text and color for a DMAS/Technical/Momentum score.

    Parameters
    ----------
    score : int
        Score value (0-100)

    Returns
    -------
    Tuple[str, str]
        (status_text, color_hex)
    """
    if score >= 70:
        return "Bullish", "#22C55E"
    elif score >= 55:
        return "Constructive", "#84CC16"
    elif score >= 45:
        return "Neutral", "#EAB308"
    elif score >= 21:
        return "Weak", "#F97316"
    else:
        return "Very Weak", "#EF4444"


def get_rsi_interpretation(rsi: float) -> Tuple[str, str, str]:
    """
    Get RSI interpretation text, color, and context.

    Parameters
    ----------
    rsi : float
        RSI value (0-100)

    Returns
    -------
    Tuple[str, str, str]
        (interpretation, color_hex, context_text)
    """
    if rsi >= 70:
        return "Overbought", "#F97316", "Caution: Potential pullback"
    elif rsi <= 30:
        return "Oversold", "#10B981", "Potential bounce opportunity"
    elif rsi >= 50:
        return "Neutral", "#C9A227", "Room to run"
    else:
        return "Neutral", "#C9A227", "Room to run"


def compute_moving_average(prices: pd.Series, period: int) -> pd.Series:
    """
    Compute simple moving average.

    Parameters
    ----------
    prices : pd.Series
        Price series
    period : int
        MA period

    Returns
    -------
    pd.Series
        Moving average values
    """
    return prices.rolling(window=period, min_periods=1).mean()


def compute_trend(
    current: float,
    previous: float
) -> Tuple[str, str]:
    """
    Compute trend direction (up/down/flat).

    Parameters
    ----------
    current : float
        Current value
    previous : float
        Previous value

    Returns
    -------
    Tuple[str, str]
        (trend_symbol, color_hex)
    """
    if current > previous:
        return "▲", "#22C55E"  # Green - up
    elif current < previous:
        return "▼", "#EF4444"  # Red - down
    else:
        return "—", "#9CA3AF"  # Gray - flat


def compute_price_vs_ma_pct(price: float, ma_value: float) -> float:
    """
    Compute percentage difference between price and moving average.

    Parameters
    ----------
    price : float
        Current price
    ma_value : float
        Moving average value

    Returns
    -------
    float
        Percentage difference (positive = above MA, negative = below)
    """
    if ma_value == 0 or pd.isna(ma_value):
        return 0.0
    return ((price - ma_value) / ma_value) * 100


def compute_trading_range(
    prices: pd.Series,
    lookback_days: int = 85
) -> Tuple[float, float, float, float]:
    """
    Compute trading range (high/low) and percentage from current price.

    Parameters
    ----------
    prices : pd.Series
        Price series
    lookback_days : int
        Number of days to look back

    Returns
    -------
    Tuple[float, float, float, float]
        (higher_range, lower_range, higher_pct, lower_pct)
    """
    recent = prices.tail(lookback_days)
    current = prices.iloc[-1]

    higher_range = recent.max()
    lower_range = recent.min()

    higher_pct = ((higher_range - current) / current) * 100 if current > 0 else 0
    lower_pct = ((current - lower_range) / current) * 100 if current > 0 else 0

    return higher_range, lower_range, higher_pct, lower_pct
