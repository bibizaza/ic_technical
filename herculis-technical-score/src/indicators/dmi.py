"""
DMI (Directional Movement Index), ADX, and ADXR indicators.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from ..utils import get_value_n_periods_ago, get_last_value, clip_score


def compute_dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> dict:
    """
    Compute Directional Movement Index (DMI), ADX, and ADXR.

    The DMI system uses the Positive Directional Indicator (+DI) and
    Negative Directional Indicator (-DI) to identify trend direction.
    ADX measures trend strength regardless of direction.

    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    period : int, default 14
        Period for smoothing

    Returns
    -------
    dict
        {
            'plus_di': float,      # Current +DI
            'minus_di': float,     # Current -DI
            'adx': float,          # Current ADX
            'adx_1w': float,       # ADX from 1 week ago (5 trading days)
            'adxr': float,         # Current ADXR
            'adxr_1w': float       # ADXR from 1 week ago
        }
    """
    # Calculate True Range (TR)
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate Directional Movement
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)

    # +DM when high_diff > low_diff and > 0
    plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff

    # -DM when low_diff > high_diff and > 0
    minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff

    # Smooth TR and DM using Wilder's smoothing (exponential with alpha = 1/period)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    # Calculate Directional Index (DX)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.fillna(0)

    # Calculate ADX (smoothed DX)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate ADXR (Average of current ADX and ADX from period days ago)
    adxr = (adx + adx.shift(period)) / 2

    # Get current and 1-week-ago values
    result = {
        'plus_di': get_last_value(plus_di) or 0.0,
        'minus_di': get_last_value(minus_di) or 0.0,
        'adx': get_last_value(adx) or 0.0,
        'adx_1w': get_value_n_periods_ago(adx, 5) or 0.0,  # 5 trading days = 1 week
        'adxr': get_last_value(adxr) or 0.0,
        'adxr_1w': get_value_n_periods_ago(adxr, 5) or 0.0,
    }

    return result


def score_dmi(dmi_data: dict, adx_threshold: float = 25) -> float:
    """
    Score the DMI system based on trend direction and strength.

    Scoring breakdown (max 1.0):
    - +DI > -DI (uptrend): +0.5
    - ADX > threshold (strong trend): +0.125
    - ADX > ADX_1w (strengthening): +0.125
    - ADXR > threshold (sustained strength): +0.125
    - ADXR > ADXR_1w (improving): +0.125

    Parameters
    ----------
    dmi_data : dict
        Output from compute_dmi()
    adx_threshold : float, default 25
        Threshold for strong trend

    Returns
    -------
    float
        Score between 0.0 and 1.0

    Examples
    --------
    >>> dmi = {
    ...     'plus_di': 30, 'minus_di': 20,
    ...     'adx': 28, 'adx_1w': 25,
    ...     'adxr': 27, 'adxr_1w': 24
    ... }
    >>> score_dmi(dmi)  # All conditions met
    1.0
    """
    score = 0.0

    # Component 1: Trend direction (0.5)
    if dmi_data['plus_di'] > dmi_data['minus_di']:
        score += 0.5

    # Component 2: ADX strength (0.125)
    if dmi_data['adx'] > adx_threshold:
        score += 0.125

    # Component 3: ADX improving (0.125)
    if dmi_data['adx'] > dmi_data['adx_1w']:
        score += 0.125

    # Component 4: ADXR strength (0.125)
    if dmi_data['adxr'] > adx_threshold:
        score += 0.125

    # Component 5: ADXR improving (0.125)
    if dmi_data['adxr'] > dmi_data['adxr_1w']:
        score += 0.125

    return clip_score(score)
