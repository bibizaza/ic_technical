"""
MACD (Moving Average Convergence Divergence) indicator.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from ..utils import get_value_n_periods_ago, get_last_value, clip_score


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> dict:
    """
    Compute MACD, Signal line, and Histogram.

    The MACD shows the relationship between two exponential moving averages.
    The signal line is an EMA of the MACD itself.
    The histogram is the difference between MACD and signal.

    Parameters
    ----------
    prices : pd.Series
        Price series
    fast : int, default 12
        Fast EMA period
    slow : int, default 26
        Slow EMA period
    signal : int, default 9
        Signal line EMA period

    Returns
    -------
    dict
        {
            'macd': float,           # Current MACD value
            'signal': float,         # Current signal line value
            'histogram': float,      # Current histogram (MACD - Signal)
            'histogram_1w': float    # Histogram from 1 week ago
        }
    """
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, adjust=False, min_periods=slow).mean()

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line (EMA of MACD)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()

    # Histogram
    histogram = macd_line - signal_line

    result = {
        'macd': get_last_value(macd_line) or 0.0,
        'signal': get_last_value(signal_line) or 0.0,
        'histogram': get_last_value(histogram) or 0.0,
        'histogram_1w': get_value_n_periods_ago(histogram, 5) or 0.0,  # 5 trading days
    }

    return result


def score_macd(macd_data: dict) -> float:
    """
    Score MACD based on crossover and histogram momentum.

    Scoring breakdown (max 1.0):
    - MACD > Signal (bullish crossover): +0.75
    - Histogram > Histogram_1w (strengthening): +0.25

    Parameters
    ----------
    macd_data : dict
        Output from compute_macd()

    Returns
    -------
    float
        Score between 0.0 and 1.0

    Examples
    --------
    >>> macd = {'macd': 1.5, 'signal': 1.0, 'histogram': 0.5, 'histogram_1w': 0.3}
    >>> score_macd(macd)  # Bullish crossover and improving
    1.0
    >>> macd = {'macd': 1.0, 'signal': 1.5, 'histogram': -0.5, 'histogram_1w': -0.3}
    >>> score_macd(macd)  # Bearish crossover and weakening
    0.0
    """
    score = 0.0

    # Component 1: MACD crossover (0.75)
    # MACD above signal = bullish
    if macd_data['macd'] > macd_data['signal']:
        score += 0.75

    # Component 2: Histogram momentum (0.25)
    # Histogram increasing = strengthening trend
    if macd_data['histogram'] > macd_data['histogram_1w']:
        score += 0.25

    return clip_score(score)
