"""
Stochastic Oscillator (%K and %D) indicator.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from ..utils import get_value_n_periods_ago, get_last_value, clip_score


def compute_stochastics(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> dict:
    """
    Compute Stochastic Oscillator (%K and %D).

    The Stochastic Oscillator compares the closing price to the price range
    over a given period. %K is the raw stochastic, %D is a moving average of %K.

    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    k_period : int, default 14
        Lookback period for %K
    d_period : int, default 3
        Moving average period for %D

    Returns
    -------
    dict
        {
            'k': float,      # Current %K
            'k_1w': float,   # %K from 1 week ago
            'd': float,      # Current %D
            'd_1w': float    # %D from 1 week ago
        }
    """
    # Calculate %K
    # %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_k = stoch_k.fillna(50)  # Neutral if calculation fails

    # Calculate %D (SMA of %K)
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()

    result = {
        'k': get_last_value(stoch_k) or 50.0,
        'k_1w': get_value_n_periods_ago(stoch_k, 5) or 50.0,
        'd': get_last_value(stoch_d) or 50.0,
        'd_1w': get_value_n_periods_ago(stoch_d, 5) or 50.0,
    }

    return result


def score_stochastics(
    stoch_data: dict,
    neutral_low: float = 30,
    neutral_high: float = 70
) -> float:
    """
    Score Stochastic Oscillator based on position and momentum.

    Scoring breakdown (max 1.0):
    - %K in neutral zone (30-70) AND %K_1w outside: +0.25 (entering neutral from extreme)
    - %K > %K_1w (improving): +0.25
    - %K > %D (bullish): +0.25
    - (%K - %D) > (%K_1w - %D_1w) (spread improving): +0.25

    Parameters
    ----------
    stoch_data : dict
        Output from compute_stochastics()
    neutral_low : float, default 30
        Lower bound of neutral zone
    neutral_high : float, default 70
        Upper bound of neutral zone

    Returns
    -------
    float
        Score between 0.0 and 1.0

    Examples
    --------
    >>> stoch = {'k': 55, 'k_1w': 25, 'd': 50, 'd_1w': 22}
    >>> score_stochastics(stoch)  # All conditions met
    1.0
    """
    score = 0.0

    k = stoch_data['k']
    k_1w = stoch_data['k_1w']
    d = stoch_data['d']
    d_1w = stoch_data['d_1w']

    # Component 1: Entering neutral zone from extreme (0.25)
    # %K now in neutral zone (30-70) but was outside last week
    k_in_neutral = neutral_low <= k <= neutral_high
    k_1w_outside_neutral = k_1w < neutral_low or k_1w > neutral_high

    if k_in_neutral and k_1w_outside_neutral:
        score += 0.25

    # Component 2: %K momentum (0.25)
    # %K increasing = bullish
    if k > k_1w:
        score += 0.25

    # Component 3: %K vs %D crossover (0.25)
    # %K above %D = bullish
    if k > d:
        score += 0.25

    # Component 4: Spread momentum (0.25)
    # (%K - %D) increasing = strengthening
    spread_now = k - d
    spread_1w = k_1w - d_1w

    if spread_now > spread_1w:
        score += 0.25

    return clip_score(score)
