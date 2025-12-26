"""
Integration helpers for herculis-assessment and subtitle generation.

Provides convenience functions to integrate the assessment classification
and subtitle generation modules into the main Streamlit app.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any


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

    Parameters
    ----------
    dmas : float
        DMAS score (0-100)

    Returns
    -------
    str
        Assessment label: "Bearish", "Cautious", "Neutral", "Constructive", or "Bullish"
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


def _import_assessment_modules():
    """Import assessment modules with temporary path modification."""
    original_path = sys.path.copy()
    try:
        # Add modules to path temporarily
        _BASE_PATH = Path(__file__).parent
        sys.path.insert(0, str(_BASE_PATH / "herculis-assessment" / "src"))
        sys.path.insert(0, str(_BASE_PATH / "market_compass"))

        # Import modules
        from config import Assessment, ASSESSMENT_LABELS
        from classifier import classify
        from subtitle_generator import SubtitleGenerator

        return Assessment, ASSESSMENT_LABELS, classify, SubtitleGenerator
    finally:
        # Restore original sys.path
        sys.path = original_path


def generate_assessment_and_subtitle(
    ticker_key: str,
    asset_name: str,
    prices: pd.Series,
    dmas: float,
    technical_score: float,
    momentum_score: float,
    dmas_prev_week: Optional[float] = None,
    price_vs_50ma: str = "above",
    price_vs_100ma: str = "above",
    price_vs_200ma: str = "above",
    ma_cross_event: Optional[str] = None,
    channel_color: str = "green",
    near_support: bool = False,
    near_resistance: bool = False,
    at_ath: bool = False,
    price_target: Optional[float] = None,
    subtitle_generator = None
) -> Dict[str, Any]:
    """
    Auto-generate assessment and subtitle for an asset.

    Parameters
    ----------
    ticker_key : str
        Ticker key (e.g., "spx", "gold")
    asset_name : str
        Display name (e.g., "S&P 500", "Gold")
    prices : pd.Series
        Price series (needs 200+ data points for structure analysis)
    dmas : float
        Current DMAS score (0-100)
    technical_score : float
        Technical score (0-100)
    momentum_score : float
        Momentum score (0-100)
    dmas_prev_week : float, optional
        Previous week DMAS for WoW change
    price_vs_50ma : str
        "above", "below", or "at"
    price_vs_100ma : str
        "above", "below", or "at"
    price_vs_200ma : str
        "above", "below", or "at"
    ma_cross_event : str, optional
        MA cross event identifier
    channel_color : str
        "green" or "red"
    near_support : bool
        Price within 2% of support
    near_resistance : bool
        Price within 2% of resistance
    at_ath : bool
        At or near all-time high
    price_target : float, optional
        Price target if applicable
    subtitle_generator : SubtitleGenerator, optional
        Subtitle generator instance (reuse for anti-repetition)

    Returns
    -------
    dict
        {
            "assessment": str,  # Assessment label
            "subtitle": str,    # Generated subtitle
            "pattern_used": str # Pattern category (for debugging)
        }
    """
    # Import modules temporarily
    Assessment, ASSESSMENT_LABELS, classify, SubtitleGenerator = _import_assessment_modules()

    # Try to use herculis-assessment for intelligent classification
    try:
        if len(prices) >= 200:
            result = classify(
                ticker=asset_name,
                prices=prices,
                dmas=dmas,
                dmas_1w=dmas_prev_week
            )
            assessment = ASSESSMENT_LABELS[result.assessment]
        else:
            # Fall back to simple DMAS-based assessment
            assessment = get_default_assessment_from_dmas(dmas)
    except Exception as e:
        print(f"Warning: Could not generate intelligent assessment for {asset_name}: {e}")
        assessment = get_default_assessment_from_dmas(dmas)

    # Generate subtitle
    try:
        if subtitle_generator is None:
            subtitle_generator = SubtitleGenerator()

        asset_data = {
            "asset_name": asset_name,
            "asset_class": "equity",  # Can be made more specific if needed
            "dmas": int(round(dmas)),
            "technical_score": int(round(technical_score)),
            "momentum_score": int(round(momentum_score)),
            "rating": assessment,
            "price_vs_50ma": price_vs_50ma,
            "price_vs_100ma": price_vs_100ma,
            "price_vs_200ma": price_vs_200ma,
            "dmas_prev_week": int(round(dmas_prev_week)) if dmas_prev_week is not None else int(round(dmas)),
            "rating_prev_week": assessment,
            "ma_cross_event": ma_cross_event,
            "channel_color": channel_color,
            "near_support": near_support,
            "near_resistance": near_resistance,
            "at_ath": at_ath,
            "price_target": price_target
        }

        subtitle_result = subtitle_generator.generate(asset_data, max_length=120)
        subtitle = subtitle_result["subtitle"]
        pattern_used = subtitle_result["pattern_used"]

    except Exception as e:
        print(f"Warning: Could not generate subtitle for {asset_name}: {e}")
        import traceback
        print(traceback.format_exc())
        subtitle = f"DMAS {dmas:.0f} - {assessment}"
        pattern_used = "fallback"

    return {
        "assessment": assessment,
        "subtitle": subtitle,
        "pattern_used": pattern_used
    }


# For the SubtitleGenerator class, create a lazy import wrapper
class SubtitleGenerator:
    """Wrapper class that lazily imports the real SubtitleGenerator."""

    def __init__(self):
        self._generator = None

    def _ensure_loaded(self):
        """Lazy load the real generator."""
        if self._generator is None:
            _, _, _, RealSubtitleGenerator = _import_assessment_modules()
            self._generator = RealSubtitleGenerator()

    def generate(self, *args, **kwargs):
        """Generate subtitle."""
        self._ensure_loaded()
        return self._generator.generate(*args, **kwargs)

    def generate_batch(self, *args, **kwargs):
        """Generate batch of subtitles."""
        self._ensure_loaded()
        return self._generator.generate_batch(*args, **kwargs)

    def reset_history(self):
        """Reset history."""
        self._ensure_loaded()
        return self._generator.reset_history()

    def get_stats(self):
        """Get stats."""
        self._ensure_loaded()
        return self._generator.get_stats()
