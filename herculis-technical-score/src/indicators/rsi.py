"""
RSI (Relative Strength Index) indicator with contrarian scoring.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from ..utils import get_last_value, clip_score


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index.

    The RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Parameters
    ----------
    prices : pd.Series
        Price series
    period : int, default 14
        Number of periods for RSI calculation

    Returns
    -------
    pd.Series
        RSI values (0-100)

    Notes
    -----
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over the period
    """
    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate the exponentially weighted moving average (Wilder's smoothing)
    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def score_rsi(rsi_value: float, overbought: float = 70, oversold: float = 30) -> float:
    """
    Score RSI using contrarian logic.

    CONTRARIAN SCORING:
    - RSI > 70 (overbought): return 0.0 (bearish - potential reversal down)
    - RSI < 30 (oversold): return 1.0 (bullish - potential reversal up)
    - RSI 30-70 (neutral): return 0.5 (no strong signal)

    This differs from trend-following RSI interpretation and matches the
    observed BQL behavior.

    Parameters
    ----------
    rsi_value : float
        Current RSI value (0-100)
    overbought : float, default 70
        Overbought threshold
    oversold : float, default 30
        Oversold threshold

    Returns
    -------
    float
        Score between 0.0 and 1.0

    Examples
    --------
    >>> score_rsi(75)  # Overbought
    0.0
    >>> score_rsi(25)  # Oversold
    1.0
    >>> score_rsi(50)  # Neutral
    0.5
    """
    if pd.isna(rsi_value):
        return 0.5  # Neutral if no data

    if rsi_value > overbought:
        # Overbought = bearish (contrarian)
        return 0.0
    elif rsi_value < oversold:
        # Oversold = bullish (contrarian)
        return 1.0
    else:
        # Neutral zone
        return 0.5
