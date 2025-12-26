"""
Parabolic SAR (Stop and Reverse) indicator with FIXED logic.

IMPORTANT: The BQL implementation had inverted logic. This module implements
the CORRECT trend-following logic where SAR below price = uptrend.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from ..utils import get_value_n_periods_ago, get_last_value, clip_score


def compute_parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2
) -> pd.Series:
    """
    Compute Parabolic SAR (Stop and Reverse).

    The Parabolic SAR provides potential stop-loss levels and identifies
    trend reversals. When SAR is below price, the trend is up; when above, down.

    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    af_start : float, default 0.02
        Initial acceleration factor
    af_increment : float, default 0.02
        Acceleration factor increment
    af_max : float, default 0.2
        Maximum acceleration factor

    Returns
    -------
    pd.Series
        Parabolic SAR values

    Notes
    -----
    This implementation follows the standard Wilder logic:
    - SAR starts below price (long position)
    - SAR = SAR_prev + AF * (EP - SAR_prev)
    - EP (Extreme Point) = highest high in uptrend, lowest low in downtrend
    - AF increases by increment each time EP is updated
    """
    length = len(close)
    sar = pd.Series(index=close.index, dtype=float)

    if length < 2:
        return sar

    # Initialize
    is_long = close.iloc[1] > close.iloc[0]
    af = af_start

    if is_long:
        sar.iloc[0] = low.iloc[0]
        ep = high.iloc[0]
    else:
        sar.iloc[0] = high.iloc[0]
        ep = low.iloc[0]

    for i in range(1, length):
        # Calculate SAR
        sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])

        if is_long:
            # Long position
            # SAR should not be above the prior two lows
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1])
            if i > 1:
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-2])

            # Check for reversal
            if low.iloc[i] < sar.iloc[i]:
                # Reverse to short
                is_long = False
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                # Update extreme point and acceleration factor
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_increment, af_max)
        else:
            # Short position
            # SAR should not be below the prior two highs
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1])
            if i > 1:
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-2])

            # Check for reversal
            if high.iloc[i] > sar.iloc[i]:
                # Reverse to long
                is_long = True
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                # Update extreme point and acceleration factor
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_increment, af_max)

    return sar


def score_parabolic(
    price: float,
    price_1w: float,
    sar: float,
    sar_1w: float
) -> float:
    """
    Score Parabolic SAR with FIXED trend-following logic.

    CORRECTED LOGIC (trend-following):
    - SAR < Price (uptrend): +0.75
    - SAR_1w > Price_1w (was downtrend, now reversed): +0.25

    The original BQL logic was inverted (contrarian), which was incorrect
    for a trend-following indicator like Parabolic SAR.

    Parameters
    ----------
    price : float
        Current price
    price_1w : float
        Price from 1 week ago
    sar : float
        Current SAR value
    sar_1w : float
        SAR from 1 week ago

    Returns
    -------
    float
        Score between 0.0 and 1.0

    Examples
    --------
    >>> score_parabolic(100, 98, 95, 99)  # Uptrend now, reversed from downtrend
    1.0
    >>> score_parabolic(100, 98, 95, 97)  # Uptrend now, was uptrend before
    0.75
    >>> score_parabolic(100, 102, 105, 99)  # Downtrend now
    0.0
    """
    score = 0.0

    # Component 1: Current trend (0.75)
    # SAR below price = uptrend (bullish)
    if sar < price:
        score += 0.75

    # Component 2: Trend reversal from down to up (0.25)
    # Was downtrend (SAR_1w > Price_1w), now uptrend = bullish reversal
    if sar_1w > price_1w and sar < price:
        score += 0.25

    return clip_score(score)
