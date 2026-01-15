"""
Integration helpers for assessment classification and subtitle generation.

Uses the Market Compass subtitle generator for proper pattern-based
subtitle generation with anti-repetition and MA dynamics.

Rating Vocabulary (v3.1):
- Bullish: Strong momentum and/or technical (DMAS >= 70)
- Constructive: Favorable technical setup (DMAS 55-69)
- Neutral: Mixed signals, no clear direction (DMAS 45-54)
- Cautious: Warning signs present (DMAS 30-44)
- Bearish: Weak technical and/or momentum (DMAS < 30)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

# Try to import the Market Compass subtitle generator
try:
    from market_compass.subtitle_generator import (
        SubtitleGenerator,
        generate_subtitle as mc_generate_subtitle,
        get_rating,
        get_high_low_dynamics,
    )
    SUBTITLE_GEN_AVAILABLE = True
except ImportError:
    SUBTITLE_GEN_AVAILABLE = False
    get_high_low_dynamics = None

# Try to import Claude API generator
try:
    from market_compass.subtitle_generator import (
        generate_subtitle as claude_generate_subtitle,
        generate_batch as claude_generate_batch,
        is_claude_available,
        set_api_key as set_claude_api_key,
    )
    CLAUDE_GEN_AVAILABLE = True
except ImportError:
    CLAUDE_GEN_AVAILABLE = False
    is_claude_available = lambda: False
    set_claude_api_key = lambda x: None


# Assessment options for Streamlit dropdown (5-level system)
# Maps to rating vocabulary: Bearish, Cautious, Neutral, Constructive, Bullish
ASSESSMENT_OPTIONS = [
    "Bearish",
    "Cautious",
    "Neutral",
    "Constructive",
    "Bullish",
]


def get_default_assessment_from_dmas(dmas: float) -> str:
    """
    Get default assessment label from DMAS score using 5-level system.

    Thresholds (per directive):
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
        Assessment label: Bullish, Constructive, Neutral, Cautious, or Bearish
    """
    if SUBTITLE_GEN_AVAILABLE:
        rating = get_rating(int(dmas))
        # get_rating returns a Rating enum, extract its value
        return rating.value if hasattr(rating, 'value') else str(rating)

    # Fallback if subtitle generator not available
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


def _calculate_ma(prices: pd.Series, period: int) -> float:
    """
    Calculate moving average value.

    Parameters
    ----------
    prices : pd.Series
        Price series
    period : int
        Moving average period (50, 100, or 200)

    Returns
    -------
    float
        Moving average value
    """
    if prices is None or len(prices) < period:
        return np.nan
    return prices.iloc[-period:].mean()


def _calculate_ma_pct(current_price: float, ma_value: float) -> float:
    """
    Calculate percentage difference between current price and moving average.

    Parameters
    ----------
    current_price : float
        Current price
    ma_value : float
        Moving average value

    Returns
    -------
    float
        Percentage above/below MA (positive = above, negative = below)
    """
    if np.isnan(ma_value) or ma_value == 0:
        return 0.0
    return ((current_price - ma_value) / ma_value) * 100


def detect_ma_cross(prices: pd.Series, lookback: int = 5) -> Optional[str]:
    """
    Detect MA cross events in recent price action.

    Parameters
    ----------
    prices : pd.Series
        Price series (at least 200+ lookback days)
    lookback : int
        Number of days to look back for cross detection

    Returns
    -------
    str or None
        Cross event: "crossed_above_50", "crossed_below_50",
        "crossed_above_100", "crossed_below_100",
        "crossed_above_200", "crossed_below_200",
        "golden_cross", "death_cross", or None
    """
    if prices is None or len(prices) < 200 + lookback:
        return None

    # Current and previous values
    current_price = prices.iloc[-1]
    prev_prices = prices.iloc[-lookback-1:-1]

    # Calculate current MAs
    ma_50 = prices.iloc[-50:].mean()
    ma_100 = prices.iloc[-100:].mean()
    ma_200 = prices.iloc[-200:].mean()

    # Calculate previous MAs (lookback days ago)
    prev_ma_50 = prices.iloc[-50-lookback:-lookback].mean()
    prev_ma_100 = prices.iloc[-100-lookback:-lookback].mean()

    # Check for golden cross (50MA crosses above 200MA)
    prev_50_vs_200 = prev_ma_50 - prices.iloc[-200-lookback:-lookback].mean()
    curr_50_vs_200 = ma_50 - ma_200
    if prev_50_vs_200 < 0 and curr_50_vs_200 > 0:
        return "golden_cross"

    # Check for death cross (50MA crosses below 200MA)
    if prev_50_vs_200 > 0 and curr_50_vs_200 < 0:
        return "death_cross"

    # Check for price crossing MAs
    prev_avg_price = prev_prices.mean()

    # 50MA cross
    if prev_avg_price < prev_ma_50 and current_price > ma_50:
        return "crossed_above_50"
    if prev_avg_price > prev_ma_50 and current_price < ma_50:
        return "crossed_below_50"

    # 100MA cross
    if prev_avg_price < prev_ma_100 and current_price > ma_100:
        return "crossed_above_100"
    if prev_avg_price > prev_ma_100 and current_price < ma_100:
        return "crossed_below_100"

    # 200MA cross (using longer lookback for 200MA)
    prev_ma_200 = prices.iloc[-200-lookback:-lookback].mean()
    if prev_avg_price < prev_ma_200 and current_price > ma_200:
        return "crossed_above_200"
    if prev_avg_price > prev_ma_200 and current_price < ma_200:
        return "crossed_below_200"

    return None


def detect_ath(prices: pd.Series, lookback_days: int = 252, threshold_pct: float = 2.0) -> bool:
    """
    Detect if price is at or near all-time high (uses 52-week high as proxy).

    Parameters
    ----------
    prices : pd.Series
        Price series
    lookback_days : int
        Number of days to look back for ATH (default 252 = ~1 year / 52 weeks)
    threshold_pct : float
        Percentage threshold for "near" detection (default 2.0)

    Returns
    -------
    bool
        True if current price is within threshold_pct of the highest price in lookback period
    """
    # Use get_high_low_dynamics if available
    if SUBTITLE_GEN_AVAILABLE and get_high_low_dynamics is not None:
        dynamics = get_high_low_dynamics(prices, threshold_pct, lookback_days)
        return dynamics.get("at_ath", False)

    # Fallback
    if prices is None or len(prices) < lookback_days:
        return False

    current_price = prices.iloc[-1]
    max_price = prices.iloc[-lookback_days:].max()

    # Within threshold_pct of 52-week high (ATH proxy)
    return current_price >= max_price * (1 - threshold_pct / 100)


def detect_52w_low(prices: pd.Series, lookback_days: int = 252, threshold_pct: float = 2.0) -> bool:
    """
    Detect if price is at or near 52-week low (support proxy).

    Parameters
    ----------
    prices : pd.Series
        Price series
    lookback_days : int
        Number of days to look back (default 252 = ~1 year / 52 weeks)
    threshold_pct : float
        Percentage threshold for "near" detection (default 2.0)

    Returns
    -------
    bool
        True if current price is within threshold_pct of the lowest price in lookback period
    """
    # Use get_high_low_dynamics if available
    if SUBTITLE_GEN_AVAILABLE and get_high_low_dynamics is not None:
        dynamics = get_high_low_dynamics(prices, threshold_pct, lookback_days)
        return dynamics.get("near_52w_low", False)

    # Fallback
    if prices is None or len(prices) < lookback_days:
        return False

    current_price = prices.iloc[-1]
    min_price = prices.iloc[-lookback_days:].min()

    # Within threshold_pct of 52-week low
    return current_price <= min_price * (1 + threshold_pct / 100)


def detect_support_resistance(
    prices: pd.Series,
    lookback_days: int = 60
) -> Tuple[bool, bool]:
    """
    Detect if price is near support or resistance levels.

    Uses simple high/low detection over lookback period.

    Parameters
    ----------
    prices : pd.Series
        Price series
    lookback_days : int
        Number of days to look back

    Returns
    -------
    tuple[bool, bool]
        (near_support, near_resistance)
    """
    if prices is None or len(prices) < lookback_days:
        return False, False

    current_price = prices.iloc[-1]
    recent_prices = prices.iloc[-lookback_days:]
    high = recent_prices.max()
    low = recent_prices.min()

    # Within 2% of support/resistance
    near_resistance = current_price >= high * 0.98
    near_support = current_price <= low * 1.02

    return near_support, near_resistance


def generate_subtitle(
    asset_name: str,
    assessment: str,
    dmas: float,
    dmas_prev_week: Optional[float] = None,
    technical_score: float = None,
    momentum_score: float = None,
    prices: pd.Series = None,
    subtitle_generator: "SubtitleGenerator" = None,
    at_ath: bool = None,
    near_support: bool = None,
    near_resistance: bool = None,
    ma_cross_event: str = None,
    price_target: float = None,
    near_52w_low: bool = None,
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
    at_ath : bool, optional
        True if at 52-week high (ATH proxy, auto-detected if prices provided)
    near_support : bool, optional
        True if near support level (auto-detected if prices provided)
    near_resistance : bool, optional
        True if near resistance level (auto-detected if prices provided)
    ma_cross_event : str, optional
        MA cross event (auto-detected if prices provided)
    price_target : float, optional
        Price target for target-based patterns
    near_52w_low : bool, optional
        True if near 52-week low (auto-detected if prices provided)

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
        current_price = prices.iloc[-1]
        ma_50 = _calculate_ma(prices, 50)
        ma_100 = _calculate_ma(prices, 100)
        ma_200 = _calculate_ma(prices, 200)

        price_vs_50ma_pct = _calculate_ma_pct(current_price, ma_50)
        price_vs_100ma_pct = _calculate_ma_pct(current_price, ma_100)
        price_vs_200ma_pct = _calculate_ma_pct(current_price, ma_200)

        # Auto-detect flags if not provided
        if at_ath is None:
            at_ath = detect_ath(prices)

        if near_52w_low is None:
            near_52w_low = detect_52w_low(prices)

        if near_support is None or near_resistance is None:
            detected_support, detected_resistance = detect_support_resistance(prices)
            if near_support is None:
                near_support = detected_support
            if near_resistance is None:
                near_resistance = detected_resistance

        if ma_cross_event is None:
            ma_cross_event = detect_ma_cross(prices)

    # Default values for flags
    if at_ath is None:
        at_ath = False
    if near_52w_low is None:
        near_52w_low = False
    if near_support is None:
        near_support = False
    if near_resistance is None:
        near_resistance = False

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
            "at_ath": at_ath,
            "near_support": near_support,
            "near_resistance": near_resistance,
            "near_52w_low": near_52w_low,
            "ma_cross_event": ma_cross_event,
            "price_target": price_target,
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
    at_ath: bool = None,
    near_support: bool = None,
    near_resistance: bool = None,
    ma_cross_event: str = None,
    price_target: float = None,
    near_52w_low: bool = None,
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
    at_ath : bool, optional
        True if at 52-week high (ATH proxy, auto-detected if not provided)
    near_support : bool, optional
        True if near support level (auto-detected if not provided)
    near_resistance : bool, optional
        True if near resistance level (auto-detected if not provided)
    ma_cross_event : str, optional
        MA cross event (auto-detected if not provided)
    price_target : float, optional
        Price target for target-based patterns
    near_52w_low : bool, optional
        True if near 52-week low (auto-detected if not provided)

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
        at_ath=at_ath,
        near_support=near_support,
        near_resistance=near_resistance,
        ma_cross_event=ma_cross_event,
        price_target=price_target,
        near_52w_low=near_52w_low,
    )

    return {
        "assessment": assessment,
        "subtitle": subtitle,
        "rating": assessment,  # Same as assessment in new vocabulary
    }


def generate_claude_subtitles_batch(
    assets_list: list,
    prices_dict: dict = None,
    data_as_of: str = None,
) -> Dict[str, dict]:
    """
    Generate subtitles for multiple assets using Claude API.

    This is the preferred method for the "Run Full Analysis" workflow
    as it generates unique, high-quality subtitles via Claude Sonnet.

    Parameters
    ----------
    assets_list : list[dict]
        List of asset dictionaries with keys:
        - ticker_key (str): Internal key (e.g., "spx", "gold")
        - asset_name (str): Display name (e.g., "S&P 500", "Gold")
        - dmas (float): DMAS score
        - technical_score (float): Technical score
        - momentum_score (float): Momentum score
        - dmas_prev_week (float, optional): Previous week's DMAS
    prices_dict : dict, optional
        Dict mapping ticker_key to price Series for MA calculations

    Returns
    -------
    dict
        Dict mapping ticker_key to result dict with:
        - assessment (str)
        - subtitle (str)
        - rating (str)
        - facts (list)
        - tokens_used (int)
    """
    if not CLAUDE_GEN_AVAILABLE or not is_claude_available():
        # Fall back to legacy generation
        print("Claude API not available, using legacy generator")
        results = {}
        for asset in assets_list:
            result = generate_assessment_and_subtitle(
                ticker_key=asset.get("ticker_key", "unknown"),
                asset_name=asset["asset_name"],
                prices=prices_dict.get(asset.get("ticker_key")) if prices_dict else None,
                dmas=asset["dmas"],
                technical_score=asset["technical_score"],
                momentum_score=asset["momentum_score"],
                dmas_prev_week=asset.get("dmas_prev_week"),
            )
            results[asset.get("ticker_key", asset["asset_name"])] = result
        return results

    # Prepare asset data for Claude API
    claude_assets = []
    for asset in assets_list:
        ticker_key = asset.get("ticker_key", "unknown")
        prices = prices_dict.get(ticker_key) if prices_dict else None

        # Calculate MA percentages
        price_vs_50ma_pct = 0.0
        price_vs_100ma_pct = 0.0
        price_vs_200ma_pct = 0.0
        at_ath = False
        near_52w_low = False
        ma_cross_event = None

        if prices is not None and len(prices) >= 200:
            current_price = prices.iloc[-1]
            ma_50 = _calculate_ma(prices, 50)
            ma_100 = _calculate_ma(prices, 100)
            ma_200 = _calculate_ma(prices, 200)

            price_vs_50ma_pct = _calculate_ma_pct(current_price, ma_50)
            price_vs_100ma_pct = _calculate_ma_pct(current_price, ma_100)
            price_vs_200ma_pct = _calculate_ma_pct(current_price, ma_200)

            at_ath = detect_ath(prices)
            near_52w_low = detect_52w_low(prices)
            ma_cross_event = detect_ma_cross(prices)

        claude_assets.append({
            "asset_name": asset["asset_name"],
            "ticker_key": ticker_key,
            "dmas": int(asset["dmas"]),
            "technical_score": int(asset["technical_score"]),
            "momentum_score": int(asset["momentum_score"]),
            "dmas_prev_week": int(asset.get("dmas_prev_week", asset["dmas"])),
            "price_vs_50ma_pct": price_vs_50ma_pct,
            "price_vs_100ma_pct": price_vs_100ma_pct,
            "price_vs_200ma_pct": price_vs_200ma_pct,
            "at_ath": at_ath,
            "near_52w_low": near_52w_low,
            "ma_cross_event": ma_cross_event,
            "rsi": asset.get("rsi"),
        })

    # Call Claude API for batch generation
    try:
        claude_results = claude_generate_batch(claude_assets, data_as_of=data_as_of)

        # Format results
        results = {}
        for i, asset in enumerate(assets_list):
            ticker_key = asset.get("ticker_key", "unknown")
            claude_result = claude_results[i] if i < len(claude_results) else {}

            results[ticker_key] = {
                "assessment": claude_result.get("rating", get_default_assessment_from_dmas(asset["dmas"])),
                "subtitle": claude_result.get("subtitle", "Technical analysis under review."),
                "rating": claude_result.get("rating", get_default_assessment_from_dmas(asset["dmas"])),
                "facts": claude_result.get("facts", []),
                "tokens_used": claude_result.get("tokens_used", 0),
            }

        return results

    except Exception as e:
        print(f"Claude API error: {e}, falling back to legacy generator")
        # Fall back to legacy generation
        results = {}
        for asset in assets_list:
            result = generate_assessment_and_subtitle(
                ticker_key=asset.get("ticker_key", "unknown"),
                asset_name=asset["asset_name"],
                prices=prices_dict.get(asset.get("ticker_key")) if prices_dict else None,
                dmas=asset["dmas"],
                technical_score=asset["technical_score"],
                momentum_score=asset["momentum_score"],
                dmas_prev_week=asset.get("dmas_prev_week"),
            )
            results[asset.get("ticker_key", asset["asset_name"])] = result
        return results


__all__ = [
    "ASSESSMENT_OPTIONS",
    "get_default_assessment_from_dmas",
    "generate_assessment_and_subtitle",
    "generate_subtitle",
    "generate_claude_subtitles_batch",
    "detect_ma_cross",
    "detect_ath",
    "detect_52w_low",
    "detect_support_resistance",
    "SUBTITLE_GEN_AVAILABLE",
    "CLAUDE_GEN_AVAILABLE",
    "is_claude_available",
    "set_claude_api_key",
]
