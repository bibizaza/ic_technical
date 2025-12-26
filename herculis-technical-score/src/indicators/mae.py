"""
MAE (Moving Average Envelope) indicator with contrarian scoring.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from ..utils import get_value_n_periods_ago, get_last_value, clip_score


def compute_mae(
    prices: pd.Series,
    period: int = 15,
    envelope_pct: float = 2.5,
    osc_period: int = 9
) -> dict:
    """
    Compute Moving Average Envelope.

    The MA Envelope creates a band around a moving average at a fixed
    percentage distance. The MA Oscillator compares a short-term MA to
    a longer-term signal MA.

    Parameters
    ----------
    prices : pd.Series
        Price series
    period : int, default 15
        Period for the moving average
    envelope_pct : float, default 2.5
        Percentage distance for upper/lower bands
    osc_period : int, default 9
        Period for MA oscillator signal line

    Returns
    -------
    dict
        {
            'ma': float,           # Moving average
            'upper': float,        # Upper band
            'lower': float,        # Lower band
            'ma_osc': float,       # MA oscillator (MA - Signal)
            'ma_osc_1w': float     # MA oscillator from 1 week ago
        }
    """
    # Calculate the moving average
    ma = prices.rolling(window=period, min_periods=period).mean()

    # Calculate envelope bands
    multiplier = envelope_pct / 100.0
    upper_band = ma * (1 + multiplier)
    lower_band = ma * (1 - multiplier)

    # Calculate MA Oscillator
    # This is the difference between the MA and its signal line (longer MA)
    signal_ma = ma.rolling(window=osc_period, min_periods=osc_period).mean()
    ma_oscillator = ma - signal_ma

    result = {
        'ma': get_last_value(ma) or 0.0,
        'upper': get_last_value(upper_band) or 0.0,
        'lower': get_last_value(lower_band) or 0.0,
        'ma_osc': get_last_value(ma_oscillator) or 0.0,
        'ma_osc_1w': get_value_n_periods_ago(ma_oscillator, 5) or 0.0,
    }

    return result


def score_mae(price: float, mae_data: dict) -> float:
    """
    Score Moving Average Envelope with CONTRARIAN logic.

    The MA Envelope is used as a contrarian indicator:
    - Price near/below lower band = oversold = BULLISH
    - Price near/above upper band = overbought = BEARISH
    - Price in middle = neutral

    Scoring breakdown (max 1.0):
    - Position score (0.5): Based on where price is in the band
      * Below lower band = 1.0 (most bullish)
      * At middle (MA) = 0.5 (neutral)
      * Above upper band = 0.0 (most bearish)
    - MA Osc > 0 (MA above signal): +0.25
    - MA Osc > MA Osc_1w (improving): +0.25

    Parameters
    ----------
    price : float
        Current price
    mae_data : dict
        Output from compute_mae()

    Returns
    -------
    float
        Score between 0.0 and 1.0

    Examples
    --------
    >>> mae = {'upper': 105, 'lower': 95, 'ma': 100, 'ma_osc': 0.5, 'ma_osc_1w': 0.3}
    >>> score_mae(94, mae)  # Below lower band, oscillator bullish
    1.0
    >>> score_mae(106, mae)  # Above upper band
    0.5  # Only oscillator components
    """
    score = 0.0

    upper = mae_data['upper']
    lower = mae_data['lower']
    ma = mae_data['ma']
    ma_osc = mae_data['ma_osc']
    ma_osc_1w = mae_data['ma_osc_1w']

    # Component 1: Contrarian position score (0.5)
    # Calculate where price is relative to the bands
    band_width = upper - lower

    if band_width > 0:
        # Normalize price position to 0-1 range
        # 0 = at upper band (overbought)
        # 1 = at lower band (oversold)
        position = (upper - price) / band_width

        # Clip to [0, 1] range
        position = max(0.0, min(1.0, position))

        # This gives us contrarian scoring:
        # - Price < lower band → position > 1.0 → clipped to 1.0 = bullish
        # - Price at MA → position = 0.5 = neutral
        # - Price > upper band → position < 0.0 → clipped to 0.0 = bearish

        score += 0.5 * position
    else:
        # If band width is zero (shouldn't happen), use neutral
        score += 0.25

    # Component 2: MA Oscillator direction (0.25)
    # MA above its signal = bullish
    if ma_osc > 0:
        score += 0.25

    # Component 3: MA Oscillator momentum (0.25)
    # Oscillator improving = bullish
    if ma_osc > ma_osc_1w:
        score += 0.25

    return clip_score(score)
