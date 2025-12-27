"""
Pattern templates for subtitle generation.

Contains all the template variations organized by rating category
(Positive, Constructive, Neutral, Cautious, Negative) to ensure
variety and avoid repetition.

RATING VOCABULARY (per directive):
- Positive: Bullish, strong momentum and/or technical (DMAS >= 70)
- Constructive: Positive outlook, favorable technical setup (DMAS 55-69)
- Neutral: Mixed signals, no clear direction (DMAS 45-54)
- Cautious: Negative bias, warning signs present (DMAS 30-44)
- Negative: Bearish, weak technical and/or momentum (DMAS < 30)

52-WEEK HIGH/LOW DETECTION:
Uses 52-week (252 trading days) high as ATH proxy since full history
is not always available. 52-week low serves as support proxy.
"""

import pandas as pd
from typing import Optional


def get_rating(dmas: int, technical_score: int = None, momentum_score: int = None) -> str:
    """
    Determine rating based on DMAS and component scores.

    Parameters
    ----------
    dmas : int
        DMAS score (0-100)
    technical_score : int, optional
        Technical score (0-100), for future refinement
    momentum_score : int, optional
        Momentum score (0-100), for future refinement

    Returns
    -------
    str
        Rating: Positive, Constructive, Neutral, Cautious, or Negative
    """
    if dmas >= 70:
        return "Positive"
    elif dmas >= 55:
        return "Constructive"
    elif dmas >= 45:
        return "Neutral"
    elif dmas >= 30:
        return "Cautious"
    else:
        return "Negative"


def get_ma_dynamics(row: dict) -> dict:
    """
    Determine MA position and dynamics for subtitle generation.

    Parameters
    ----------
    row : dict
        Asset data with price_vs_50ma_pct, price_vs_100ma_pct, price_vs_200ma_pct

    Returns
    -------
    dict
        MA dynamics flags for decision making
    """
    price_vs_50ma = row.get("price_vs_50ma_pct", 0)
    price_vs_100ma = row.get("price_vs_100ma_pct", 0)
    price_vs_200ma = row.get("price_vs_200ma_pct", 0)

    return {
        "above_50ma": price_vs_50ma > 0,
        "above_100ma": price_vs_100ma > 0,
        "above_200ma": price_vs_200ma > 0,
        "at_50ma": abs(price_vs_50ma) < 1.0,  # Within 1%
        "at_100ma": abs(price_vs_100ma) < 1.0,
        "at_200ma": abs(price_vs_200ma) < 1.0,
        "far_above_all": price_vs_50ma > 3 and price_vs_100ma > 5 and price_vs_200ma > 8,
        "far_below_all": price_vs_50ma < -3 and price_vs_100ma < -5 and price_vs_200ma < -8,
        "between_50_100": price_vs_50ma < 0 and price_vs_100ma > 0,
        "between_100_200": price_vs_100ma < 0 and price_vs_200ma > 0,
        "price_vs_50ma_pct": price_vs_50ma,
        "price_vs_100ma_pct": price_vs_100ma,
        "price_vs_200ma_pct": price_vs_200ma,
    }


def get_relevant_ma(row: dict) -> int:
    """
    Determine the most relevant MA for the current situation.

    Parameters
    ----------
    row : dict
        Asset data with price_vs_*ma_pct fields

    Returns
    -------
    int
        MA period (50, 100, or 200)
    """
    price_vs_50ma = row.get("price_vs_50ma_pct", 0)
    price_vs_100ma = row.get("price_vs_100ma_pct", 0)

    if price_vs_50ma < 0:
        return 50
    elif price_vs_100ma < 0:
        return 100
    else:
        return 200


def get_high_low_dynamics(
    prices: pd.Series,
    threshold_pct: float = 2.0,
    lookback_days: int = 252
) -> dict:
    """
    Detect 52-week high/low proximity for subtitle generation.

    Uses 52-week high as ATH proxy since full price history is not
    always available. 52-week low serves as support proxy.

    Parameters
    ----------
    prices : pd.Series
        Price series (requires at least lookback_days of data)
    threshold_pct : float, default=2.0
        Percentage threshold for "near" detection
    lookback_days : int, default=252
        Number of trading days for 52-week window

    Returns
    -------
    dict
        High/low dynamics with keys:
        - near_52w_high (bool): Within threshold_pct of 52-week high
        - near_52w_low (bool): Within threshold_pct of 52-week low
        - pct_from_high (float): Percentage below 52-week high
        - pct_from_low (float): Percentage above 52-week low
        - high_52w (float): 52-week high value
        - low_52w (float): 52-week low value
        - at_ath (bool): Alias for near_52w_high (ATH proxy)
        - near_support (bool): Alias for near_52w_low (support proxy)
    """
    result = {
        "near_52w_high": False,
        "near_52w_low": False,
        "pct_from_high": 0.0,
        "pct_from_low": 0.0,
        "high_52w": None,
        "low_52w": None,
        "at_ath": False,
        "near_support": False,
    }

    if prices is None or len(prices) < lookback_days:
        return result

    current_price = prices.iloc[-1]
    lookback_prices = prices.iloc[-lookback_days:]

    high_52w = lookback_prices.max()
    low_52w = lookback_prices.min()

    result["high_52w"] = high_52w
    result["low_52w"] = low_52w

    # Calculate percentage from high/low
    if high_52w > 0:
        pct_from_high = ((high_52w - current_price) / high_52w) * 100
        result["pct_from_high"] = pct_from_high
        result["near_52w_high"] = pct_from_high <= threshold_pct
        result["at_ath"] = result["near_52w_high"]  # ATH proxy

    if low_52w > 0:
        pct_from_low = ((current_price - low_52w) / low_52w) * 100
        result["pct_from_low"] = pct_from_low
        result["near_52w_low"] = pct_from_low <= threshold_pct
        result["near_support"] = result["near_52w_low"]  # Support proxy

    return result


# Pattern templates organized by RATING category
# Use {asset}, {ma}, {target}, {ordinal} as placeholders
# Maximum 15 words per pattern

PATTERNS = {
    # =========================================================================
    # POSITIVE Rating Patterns (DMAS >= 70)
    # =========================================================================
    "positive_strong": [
        "The picture remains bullish with strong momentum",
        "Momentum remains strong as well as technical. The setup is clean and bullish",
        "{asset} has all the technical elements to go higher in the coming weeks",
        "No cloud on the technical horizon",
        "Strong technical and momentum keep DMAS high and {asset} stretched over its MA",
        "All the stars seem aligned to propel {asset} even higher",
    ],

    "positive_ath": [
        "New 52-week highs are supported by strong technical and momentum",
        "At 52-week highs with the technical picture calling for more gains",
        "Trading near 52-week highs with strong momentum behind it",
    ],

    "positive_target": [
        "The bull trend is intact, {asset} could go to {target} in the coming weeks",
        "We are getting close to the first target of {target}",
    ],

    "positive_continuation": [
        "As expected last week, {asset} continues its ascent",
        "Last week's call confirmed: {asset} propels higher and has still fuel",
        "The positive medium-term trend may continue with strong technical setup",
        "The price action should continue its positive ascent",
    ],

    "positive_rebound": [
        "Successful rebound on the {ma}d MA",
        "Revived momentum pushes DMAS higher and keeps {asset} marching above key averages",
    ],

    # =========================================================================
    # CONSTRUCTIVE Rating Patterns (DMAS 55-69)
    # =========================================================================
    "constructive_caution": [
        "While the momentum is still high, the technical score calls for some caution",
        "The momentum has now switched to 100 but the technical is still calling for caution",
    ],

    "constructive_correction": [
        "The picture remains bullish despite the current correction",
        "Despite the correction, the picture is still positive for {asset}",
        "The correction may stabilize. Momentum is still strong",
        "The correction has been contained thanks to a high momentum",
    ],

    "constructive_improving": [
        "The technical signal is slightly lower but the picture remains bullish",
        "The global technical picture is unchanged and calls for more gains",
    ],

    # =========================================================================
    # NEUTRAL Rating Patterns (DMAS 45-54)
    # =========================================================================
    "neutral_tech_offset": [
        "The technical score is still supporting the bull movement despite the poor momentum",
        "High technical score is offset by weak momentum",
        "The technical score has improved to bull level but it is offset by poor momentum",
    ],

    "neutral_mom_offset": [
        "Despite a strong momentum, the technical picture has now shifted to neutral territory",
        "High momentum score may offset the current technical weakness",
    ],

    "neutral_consolidation": [
        "The technical score has been stable over the past weeks. Ongoing consolidation",
        "Still hovering above its two moving averages in a tight channel",
    ],

    "neutral_turning": [
        "{asset} is at a turning point: technical suggest uptrend but momentum calls caution",
        "{asset} is now hovering on its support, waiting for a clear new trend",
        "No clear trend despite a robust technical score",
    ],

    # =========================================================================
    # CAUTIOUS Rating Patterns (DMAS 30-44)
    # =========================================================================
    "cautious_weakening": [
        "The technical are weakening fast and open the door to a test to the {ma}d MA",
        "The technical setup is weakening even more",
    ],

    "cautious_stuck": [
        "Still below the {ma}-day MA that seems to act as a strong resistance",
        "After the macro rebound, {asset} remains stuck below the {ma}d MA",
    ],

    "cautious_silver": [
        "The picture is weak but {asset} managed to break above its 2 MA",
        "Very weak picture but {asset} has so far managed to stay above the {ma}d MA",
        "Despite a small bump in DMAS, the picture remains very negative",
    ],

    "cautious_rebound": [
        "Small rebound but still not enough to trigger a new positive trend",
        "A short-lived rebound accompanied by a death cross",
    ],

    "cautious_near_52w_low": [
        "{asset} is testing 52-week lows with weak momentum",
        "Trading near 52-week lows, technical setup remains fragile",
        "Close to 52-week lows with no clear reversal signal yet",
    ],

    # =========================================================================
    # NEGATIVE Rating Patterns (DMAS < 30)
    # =========================================================================
    "negative_deep": [
        "{asset} is deeply engulfed in negative territory",
        "The choppy trading range remains deeply negative",
        "Strong headwinds for {asset}",
    ],

    "negative_no_hope": [
        "No short term hopes of a sustained rebound with challenging technical and momentum",
        "Weak technical, poor momentum and the cross below the {ma}d MA does not bode well",
        "A weak technical setup and momentum are no help for a rebound",
    ],

    "negative_dramatic": [
        "A dramatic picture for {asset} that crossed all its MA in a week",
        "No respite for {asset} that crossed down all the MA",
    ],

    "negative_near_52w_low": [
        "{asset} is at 52-week lows with no sign of a bottom",
        "Trading at 52-week lows amid deep negative momentum",
        "At 52-week lows with both technical and momentum severely impaired",
    ],

    # =========================================================================
    # Special Event Patterns
    # =========================================================================
    "ma_cross_up": [
        "Successful rebound on the {ma}d MA",
        "Breakout confirmed! Despite poor momentum, {asset} closed above {ma}-day MA",
    ],

    "ma_cross_down": [
        "Dangerous setup: {asset} just closed below the {ma}d MA",
    ],

    "golden_cross": [
        "Golden cross confirms the constructive setup",
    ],

    "golden_cross_weak": [
        "{asset} shows a golden cross but DMAS remains weak",
    ],

    "death_cross": [
        "A short-lived rebound accompanied by a death cross",
    ],

    "dramatic_surge": [
        "A surge in momentum lifts DMAS and moves {asset} closer to bullish territory",
    ],

    "dramatic_collapse": [
        "The technical picture has worsened significantly this week",
    ],
}


# MA number mappings for extracting from event strings
MA_EVENT_MAPPING = {
    "crossed_above_50": 50,
    "crossed_below_50": 50,
    "crossed_above_100": 100,
    "crossed_below_100": 100,
    "crossed_above_200": 200,
    "crossed_below_200": 200
}


def get_ma_number(ma_cross_event: str) -> int:
    """
    Extract MA period number from cross event string.

    Parameters
    ----------
    ma_cross_event : str
        MA cross event identifier

    Returns
    -------
    int
        MA period (50, 100, or 200)
    """
    return MA_EVENT_MAPPING.get(ma_cross_event, 50)


def get_ordinal(n: int) -> str:
    """
    Convert number to ordinal string (1st, 2nd, 3rd, etc.).

    Parameters
    ----------
    n : int
        Number to convert

    Returns
    -------
    str
        Ordinal string
    """
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"
