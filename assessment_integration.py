"""
Integration helpers for assessment classification and subtitle generation.

Uses the Market Compass subtitle generator for proper pattern-based
subtitle generation with anti-repetition and MA dynamics.

Rating Vocabulary:
- Positive: Bullish, strong momentum and/or technical (DMAS >= 70)
- Constructive: Positive outlook, favorable technical setup (DMAS 55-69)
- Neutral: Mixed signals, no clear direction (DMAS 45-54)
- Cautious: Negative bias, warning signs present (DMAS 30-44)
- Negative: Bearish, weak technical and/or momentum (DMAS < 30)
"""

import pandas as pd
from typing import Optional, Dict

# Try to import the Market Compass subtitle generator
try:
    from market_compass.subtitle_generator import (
        SubtitleGenerator,
        generate_subtitle as mc_generate_subtitle,
        get_rating,
    )
    SUBTITLE_GEN_AVAILABLE = True
except ImportError:
    SUBTITLE_GEN_AVAILABLE = False


# Assessment options for Streamlit dropdown (5-level system)
# Maps to rating vocabulary: Negative, Cautious, Neutral, Constructive, Positive
ASSESSMENT_OPTIONS = [
    "Negative",
    "Cautious",
    "Neutral",
    "Constructive",
    "Positive",
]


def get_default_assessment_from_dmas(dmas: float) -> str:
    """
    Get default assessment label from DMAS score using 5-level system.

    Thresholds (per directive):
    - DMAS >= 70: Positive
    - DMAS >= 55: Constructive
    - DMAS >= 45: Neutral
    - DMAS >= 30: Cautious
    - DMAS < 30: Negative

    Parameters
    ----------
    dmas : float
        DMAS score (0-100)

    Returns
    -------
    str
        Assessment label: Positive, Constructive, Neutral, Cautious, or Negative
    """
    if SUBTITLE_GEN_AVAILABLE:
        return get_rating(int(dmas))

    # Fallback if subtitle generator not available
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


def _calculate_ma_pct(prices: pd.Series, ma_period: int) -> float:
    """
    Calculate percentage difference between current price and moving average.

    Parameters
    ----------
    prices : pd.Series
        Price series
    ma_period : int
        Moving average period (50, 100, or 200)

    Returns
    -------
    float
        Percentage above/below MA (positive = above, negative = below)
    """
    if prices is None or len(prices) < ma_period:
        return 0.0

    current_price = prices.iloc[-1]
    ma_value = prices.iloc[-ma_period:].mean()

    if ma_value == 0:
        return 0.0

    return ((current_price - ma_value) / ma_value) * 100


def generate_subtitle(
    asset_name: str,
    assessment: str,
    dmas: float,
    dmas_prev_week: Optional[float] = None,
    technical_score: float = None,
    momentum_score: float = None,
    prices: pd.Series = None,
    subtitle_generator: "SubtitleGenerator" = None,
) -> str:
    """
    Generate a subtitle for the asset using the Market Compass pattern library.

    Parameters
    ----------
    asset_name : str
        Name of the asset (e.g., "S&P 500", "Gold")
    assessment : str
        Assessment level (Positive, Constructive, Neutral, Cautious, Negative)
    dmas : float
        Current DMAS score
    dmas_prev_week : float, optional
        Previous week's DMAS score
    technical_score : float, optional
        Technical score (0-100)
    momentum_score : float, optional
        Momentum score (0-100)
    prices : pd.Series, optional
        Price series for calculating MA dynamics
    subtitle_generator : SubtitleGenerator, optional
        Existing generator instance for anti-repetition tracking

    Returns
    -------
    str
        Generated subtitle (max 15 words, one sentence)
    """
    # Calculate MA dynamics if prices available
    price_vs_50ma_pct = 0.0
    price_vs_100ma_pct = 0.0
    price_vs_200ma_pct = 0.0

    if prices is not None and len(prices) >= 200:
        price_vs_50ma_pct = _calculate_ma_pct(prices, 50)
        price_vs_100ma_pct = _calculate_ma_pct(prices, 100)
        price_vs_200ma_pct = _calculate_ma_pct(prices, 200)

    # Use Market Compass subtitle generator if available
    if SUBTITLE_GEN_AVAILABLE:
        asset_data = {
            "asset_name": asset_name,
            "asset_class": "equity",  # Default, could be passed in
            "dmas": int(dmas) if dmas is not None else 50,
            "technical_score": int(technical_score) if technical_score is not None else 50,
            "momentum_score": int(momentum_score) if momentum_score is not None else 50,
            "dmas_prev_week": int(dmas_prev_week) if dmas_prev_week is not None else int(dmas) if dmas is not None else 50,
            "price_vs_50ma_pct": price_vs_50ma_pct,
            "price_vs_100ma_pct": price_vs_100ma_pct,
            "price_vs_200ma_pct": price_vs_200ma_pct,
            "at_ath": False,
            "near_support": False,
            "near_resistance": False,
            "ma_cross_event": None,
            "price_target": None,
        }

        try:
            if subtitle_generator is not None:
                result = subtitle_generator.generate(asset_data)
            else:
                result = mc_generate_subtitle(asset_data)
            return result["subtitle"]
        except Exception:
            pass  # Fall through to fallback

    # Fallback: simple template-based generation
    change_desc = ""
    if dmas_prev_week is not None and dmas != dmas_prev_week:
        change = dmas - dmas_prev_week
        if abs(change) >= 10:
            direction = "improving" if change > 0 else "weakening"
            change_desc = f" Shows {direction} momentum."

    templates = {
        "Positive": [
            f"The picture remains bullish with strong momentum.",
            f"{asset_name} has all the technical elements to go higher.",
            f"No cloud on the technical horizon.",
        ],
        "Constructive": [
            f"The picture remains bullish despite the current correction.",
            f"Despite the correction, the picture is still positive for {asset_name}.",
            f"The global technical picture is unchanged and calls for more gains.",
        ],
        "Neutral": [
            f"No clear trend despite a robust technical score.",
            f"{asset_name} is at a turning point.",
            f"The technical score has been stable. Ongoing consolidation.",
        ],
        "Cautious": [
            f"The technical setup is weakening.",
            f"Small rebound but still not enough to trigger a new positive trend.",
            f"The picture is weak but {asset_name} managed to stay above key MA.",
        ],
        "Negative": [
            f"{asset_name} is deeply engulfed in negative territory.",
            f"Strong headwinds for {asset_name}.",
            f"No short term hopes of a sustained rebound.",
        ],
    }

    templates_list = templates.get(assessment, templates["Neutral"])
    idx = int(dmas) % len(templates_list)
    return templates_list[idx] + change_desc


def generate_assessment_and_subtitle(
    ticker_key: str,
    asset_name: str,
    prices: pd.Series,
    dmas: float,
    technical_score: float,
    momentum_score: float,
    dmas_prev_week: Optional[float] = None,
    subtitle_generator: "SubtitleGenerator" = None,
    **kwargs
) -> Dict[str, str]:
    """
    Generate both assessment and subtitle for an asset.

    Uses the Market Compass subtitle generator for pattern-based subtitles
    with anti-repetition and proper MA dynamics.

    Parameters
    ----------
    ticker_key : str
        Internal ticker key (e.g., "spx", "gold")
    asset_name : str
        Display name (e.g., "S&P 500", "Gold")
    prices : pd.Series
        Price series
    dmas : float
        Current DMAS score (0-100)
    technical_score : float
        Technical score (0-100)
    momentum_score : float
        Momentum score (0-100)
    dmas_prev_week : float, optional
        Previous week's DMAS
    subtitle_generator : SubtitleGenerator, optional
        Existing generator instance for anti-repetition tracking

    Returns
    -------
    dict
        Dictionary with keys: 'assessment', 'subtitle', 'rating'
    """
    # Generate assessment from DMAS
    assessment = get_default_assessment_from_dmas(dmas)

    # Generate subtitle using Market Compass generator
    subtitle = generate_subtitle(
        asset_name=asset_name,
        assessment=assessment,
        dmas=dmas,
        dmas_prev_week=dmas_prev_week,
        technical_score=technical_score,
        momentum_score=momentum_score,
        prices=prices,
        subtitle_generator=subtitle_generator,
    )

    return {
        "assessment": assessment,
        "subtitle": subtitle,
        "rating": assessment,  # Same as assessment in new vocabulary
    }


__all__ = [
    "ASSESSMENT_OPTIONS",
    "get_default_assessment_from_dmas",
    "generate_assessment_and_subtitle",
    "generate_subtitle",
    "SUBTITLE_GEN_AVAILABLE",
]
