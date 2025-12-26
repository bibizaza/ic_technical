"""
Integration helpers for assessment classification and subtitle generation.

Provides simplified assessment classification and subtitle generation
without complex module dependencies.
"""

import pandas as pd
from typing import Optional, Dict


# Assessment options for Streamlit dropdown (5-level system)
ASSESSMENT_OPTIONS = [
    "Bearish",
    "Cautious",
    "Neutral",
    "Constructive",
    "Bullish",
]


def get_default_assessment_from_dmas(dmas: float) -> str:
    """
    Get default assessment label from DMAS score using new 5-level system.

    Thresholds:
    - DMAS >= 70: Bullish
    - DMAS >= 55: Constructive
    - DMAS >= 45: Neutral
    - DMAS >= 30: Cautious
    - DMAS < 30: Bearish

    Parameters
    ----------
    dmas : float
        DMAS score (0-100)

    Returns
    -------
    str
        Assessment label
    """
    if dmas >= 70:
        return "Bullish"
    elif dmas >= 55:
        return "Constructive"
    elif dmas >= 45:
        return "Neutral"
    elif dmas >= 30:
        return "Cautious"
    else:
        return "Bearish"


def generate_subtitle(
    asset_name: str,
    assessment: str,
    dmas: float,
    dmas_prev_week: Optional[float] = None,
    technical_score: float = None,
    momentum_score: float = None
) -> str:
    """
    Generate a simple subtitle for the asset based on its assessment.

    Parameters
    ----------
    asset_name : str
        Name of the asset (e.g., "S&P 500", "Gold")
    assessment : str
        Assessment level ("Bearish", "Cautious", etc.)
    dmas : float
        Current DMAS score
    dmas_prev_week : float, optional
        Previous week's DMAS score
    technical_score : float, optional
        Technical score
    momentum_score : float, optional
        Momentum score

    Returns
    -------
    str
        Generated subtitle
    """
    # Calculate change if previous DMAS available
    change_desc = ""
    if dmas_prev_week is not None and dmas != dmas_prev_week:
        change = dmas - dmas_prev_week
        if abs(change) >= 10:
            direction = "improving" if change > 0 else "weakening"
            change_desc = f" {asset_name} shows {direction} momentum with DMAS {'rising' if change > 0 else 'falling'} by {abs(change):.0f} points."
        elif abs(change) >= 5:
            direction = "strengthening" if change > 0 else "softening"
            change_desc = f" Technical indicators show {direction} signals."

    # Base subtitle templates by assessment
    templates = {
        "Bullish": [
            f"{asset_name} maintains strong uptrend with positive technical setup.{change_desc}",
            f"Technical indicators remain constructive for {asset_name}.{change_desc}",
            f"{asset_name} shows bullish momentum across key timeframes.{change_desc}",
        ],
        "Constructive": [
            f"{asset_name} displays constructive price action with favorable risk/reward.{change_desc}",
            f"Technical setup for {asset_name} suggests potential for further gains.{change_desc}",
            f"{asset_name} maintains position above key moving averages.{change_desc}",
        ],
        "Neutral": [
            f"{asset_name} trades in neutral range with mixed technical signals.{change_desc}",
            f"Price action for {asset_name} shows consolidation pattern.{change_desc}",
            f"{asset_name} lacks clear directional bias near current levels.{change_desc}",
        ],
        "Cautious": [
            f"{asset_name} shows cautious signals as momentum weakens.{change_desc}",
            f"Technical indicators suggest increasing caution for {asset_name}.{change_desc}",
            f"{asset_name} faces resistance with deteriorating breadth.{change_desc}",
        ],
        "Bearish": [
            f"{asset_name} remains under pressure with negative technical outlook.{change_desc}",
            f"Technical setup for {asset_name} suggests continued weakness.{change_desc}",
            f"{asset_name} trades below key support levels with bearish momentum.{change_desc}",
        ],
    }

    # Select template based on DMAS score to add variety
    templates_list = templates.get(assessment, templates["Neutral"])
    idx = int(dmas) % len(templates_list)
    return templates_list[idx]


def generate_assessment_and_subtitle(
    ticker_key: str,
    asset_name: str,
    prices: pd.Series,
    dmas: float,
    technical_score: float,
    momentum_score: float,
    dmas_prev_week: Optional[float] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Generate both assessment and subtitle for an asset.

    This is a simplified version that doesn't require the complex
    herculis-assessment or subtitle_generator modules.

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
        Technical score
    momentum_score : float
        Momentum score
    dmas_prev_week : float, optional
        Previous week's DMAS

    Returns
    -------
    dict
        Dictionary with keys: 'assessment', 'subtitle'
    """
    # Generate assessment from DMAS
    assessment = get_default_assessment_from_dmas(dmas)

    # Generate subtitle
    subtitle = generate_subtitle(
        asset_name=asset_name,
        assessment=assessment,
        dmas=dmas,
        dmas_prev_week=dmas_prev_week,
        technical_score=technical_score,
        momentum_score=momentum_score
    )

    return {
        "assessment": assessment,
        "subtitle": subtitle
    }


__all__ = [
    "ASSESSMENT_OPTIONS",
    "get_default_assessment_from_dmas",
    "generate_assessment_and_subtitle",
]
