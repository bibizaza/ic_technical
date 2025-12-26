"""
Moving Average indicators: SMA and EMA.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
sys.path.append('..')
from ..utils import get_last_value, clip_score


def compute_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Compute Simple Moving Average.

    Parameters
    ----------
    prices : pd.Series
        Price series
    period : int
        Number of periods for the moving average

    Returns
    -------
    pd.Series
        SMA values
    """
    return prices.rolling(window=period, min_periods=period).mean()


def compute_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Compute Exponential Moving Average.

    Parameters
    ----------
    prices : pd.Series
        Price series
    period : int
        Number of periods for the moving average

    Returns
    -------
    pd.Series
        EMA values
    """
    return prices.ewm(span=period, adjust=False, min_periods=period).mean()


def score_ma(
    current_price: float,
    ma_values: Dict[int, float],
    ma_type: str = "SMA"
) -> float:
    """
    Score based on price position relative to moving averages.

    The score is the average of binary flags: 1 if price > MA, else 0.
    This indicates bullish strength when price is above most/all MAs.

    Parameters
    ----------
    current_price : float
        Current price
    ma_values : Dict[int, float]
        Dictionary mapping period to MA value
        e.g., {5: 100.5, 10: 99.8, 20: 98.2, ...}
    ma_type : str
        Type of MA ("SMA" or "EMA") for debugging

    Returns
    -------
    float
        Score between 0.0 and 1.0

    Examples
    --------
    >>> ma_vals = {5: 100, 10: 99, 20: 98, 50: 97, 100: 96, 200: 95}
    >>> score_ma(101, ma_vals)  # Price above all MAs
    1.0
    >>> score_ma(96.5, ma_vals)  # Price above only 200-day MA
    0.166...
    """
    if not ma_values:
        return 0.5  # Neutral if no MA data

    # Count how many MAs the price is above
    above_count = sum(1 for ma_val in ma_values.values() if current_price > ma_val)
    total_count = len(ma_values)

    score = above_count / total_count
    return clip_score(score)


def compute_all_mas(prices: pd.Series, periods: list) -> Dict[int, pd.Series]:
    """
    Compute multiple moving averages at once.

    Parameters
    ----------
    prices : pd.Series
        Price series
    periods : list
        List of periods to compute

    Returns
    -------
    Dict[int, pd.Series]
        Dictionary mapping period to MA series
    """
    return {period: compute_sma(prices, period) for period in periods}


def compute_all_emas(prices: pd.Series, periods: list) -> Dict[int, pd.Series]:
    """
    Compute multiple EMAs at once.

    Parameters
    ----------
    prices : pd.Series
        Price series
    periods : list
        List of periods to compute

    Returns
    -------
    Dict[int, pd.Series]
        Dictionary mapping period to EMA series
    """
    return {period: compute_ema(prices, period) for period in periods}
