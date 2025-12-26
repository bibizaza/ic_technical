"""
Decision tree logic for subtitle generation.

Implements priority-based routing to select the most appropriate
pattern based on asset data.
"""

from typing import Optional
from .patterns import get_ma_number


def route_subtitle(
    asset_data: dict,
    pattern_selector: callable
) -> tuple[str, str]:
    """
    Route to appropriate subtitle pattern based on asset data.

    Processes in priority order:
    1. MA cross events (most newsworthy)
    2. Dramatic WoW changes (|delta| > 15)
    3. DMAS level routing (bullish/neutral/bearish)

    Parameters
    ----------
    asset_data : dict
        Asset data dictionary (see INPUT SCHEMA in directive)
    pattern_selector : callable
        Function to select pattern from category (handles anti-repetition)

    Returns
    -------
    tuple[str, str]
        (subtitle_text, pattern_category_used)
    """
    asset = asset_data["asset_name"]
    dmas = asset_data["dmas"]
    dmas_prev = asset_data.get("dmas_prev_week", dmas)
    technical = asset_data["technical_score"]
    momentum = asset_data["momentum_score"]
    ma_cross = asset_data.get("ma_cross_event")
    channel = asset_data.get("channel_color", "green")
    at_ath = asset_data.get("at_ath", False)
    near_support = asset_data.get("near_support", False)
    near_resistance = asset_data.get("near_resistance", False)
    price_target = asset_data.get("price_target")
    price_vs_50ma = asset_data.get("price_vs_50ma", "above")
    price_vs_100ma = asset_data.get("price_vs_100ma", "above")
    price_vs_200ma = asset_data.get("price_vs_200ma", "above")

    dmas_change = dmas - dmas_prev

    # PRIORITY 1: MA Cross Events
    if ma_cross:
        return _handle_ma_cross(
            asset, ma_cross, dmas, momentum, technical,
            pattern_selector
        )

    # PRIORITY 2: Dramatic WoW Changes
    if abs(dmas_change) > 15:
        return _handle_dramatic_change(
            asset, dmas_change, pattern_selector
        )

    # PRIORITY 3: Route by DMAS Level
    if dmas > 65:
        return _handle_bullish(
            asset, dmas, technical, momentum, dmas_change,
            at_ath, near_resistance, price_target, channel,
            pattern_selector
        )
    elif dmas >= 45:
        return _handle_neutral(
            asset, dmas, technical, momentum, dmas_change,
            price_vs_50ma, near_support, near_resistance,
            pattern_selector
        )
    else:
        return _handle_bearish(
            asset, dmas, technical, momentum, dmas_change,
            price_vs_50ma, price_vs_100ma, price_vs_200ma,
            pattern_selector
        )


def _handle_ma_cross(
    asset: str,
    ma_cross: str,
    dmas: int,
    momentum: int,
    technical: int,
    pattern_selector: callable
) -> tuple[str, str]:
    """Handle MA cross events (Priority 1)."""

    if ma_cross == "death_cross":
        pattern = pattern_selector("death_cross")
        return pattern.format(asset=asset), "death_cross"

    if ma_cross == "golden_cross":
        if dmas < 50:
            pattern = pattern_selector("golden_cross_weak")
            # Note: price_range would need to be calculated from data
            return pattern.format(asset=asset, price_range="current"), "golden_cross_weak"
        else:
            pattern = pattern_selector("golden_cross")
            return pattern.format(asset=asset), "golden_cross"

    if "crossed_above" in ma_cross:
        ma_num = get_ma_number(ma_cross)
        pattern = pattern_selector("ma_cross_up")
        return pattern.format(asset=asset, ma=ma_num), "ma_cross_up"

    if "crossed_below" in ma_cross:
        ma_num = get_ma_number(ma_cross)
        pattern = pattern_selector("ma_cross_down")
        return pattern.format(asset=asset, ma=ma_num), "ma_cross_down"

    # Fallback (should not reach here)
    return _handle_bullish(
        asset, dmas, technical, momentum, 0,
        False, False, None, "green", pattern_selector
    )


def _handle_dramatic_change(
    asset: str,
    dmas_change: float,
    pattern_selector: callable
) -> tuple[str, str]:
    """Handle dramatic WoW changes (Priority 2)."""

    if dmas_change > 15:
        pattern = pattern_selector("dramatic_surge")
        return pattern.format(asset=asset), "dramatic_surge"
    else:  # dmas_change < -15
        pattern = pattern_selector("dramatic_collapse")
        return pattern.format(asset=asset), "dramatic_collapse"


def _handle_bullish(
    asset: str,
    dmas: int,
    technical: int,
    momentum: int,
    dmas_change: float,
    at_ath: bool,
    near_resistance: bool,
    price_target: Optional[float],
    channel: str,
    pattern_selector: callable
) -> tuple[str, str]:
    """Handle bullish scenarios (DMAS > 65)."""

    # Extreme momentum (100)
    if momentum == 100:
        pattern = pattern_selector("extreme_momentum_high")
        return pattern.format(asset=asset), "extreme_momentum_high"

    # Strong bullish - both components high
    if technical >= 60 and momentum >= 80:
        if at_ath:
            pattern = pattern_selector("bullish_ath")
            return pattern.format(asset=asset), "bullish_ath"

        if near_resistance and price_target:
            pattern = pattern_selector("bullish_target")
            return pattern.format(
                asset=asset,
                target=int(price_target),
                ordinal="first"
            ), "bullish_target"

        pattern = pattern_selector("bullish_strong")
        return pattern.format(asset=asset), "bullish_strong"

    # Bullish with caution - high momentum, weaker technical
    if technical < 55 and momentum >= 80:
        pattern = pattern_selector("bullish_caution")
        return pattern.format(asset=asset), "bullish_caution"

    # Bullish despite correction
    if channel == "red" or dmas_change < 0:
        pattern = pattern_selector("bullish_correction")
        return pattern.format(asset=asset), "bullish_correction"

    # Default bullish
    pattern = pattern_selector("bullish_strong")
    return pattern.format(asset=asset), "bullish_strong"


def _handle_neutral(
    asset: str,
    dmas: int,
    technical: int,
    momentum: int,
    dmas_change: float,
    price_vs_50ma: str,
    near_support: bool,
    near_resistance: bool,
    pattern_selector: callable
) -> tuple[str, str]:
    """Handle neutral scenarios (DMAS 45-65)."""

    # High technical, low momentum
    if technical >= 60 and momentum < 40:
        pattern = pattern_selector("neutral_tech_offset")
        return pattern.format(asset=asset), "neutral_tech_offset"

    # Low technical, high momentum
    if technical < 50 and momentum >= 70:
        pattern = pattern_selector("neutral_mom_offset")
        return pattern.format(asset=asset), "neutral_mom_offset"

    # Consolidation (small WoW change)
    if abs(dmas_change) < 5:
        if price_vs_50ma == "at":
            pattern = pattern_selector("neutral_turning")
            return pattern.format(asset=asset), "neutral_turning"

        pattern = pattern_selector("neutral_consolidation")
        return pattern.format(asset=asset), "neutral_consolidation"

    # Turning point
    if near_support or near_resistance:
        pattern = pattern_selector("neutral_turning")
        return pattern.format(asset=asset), "neutral_turning"

    # Default neutral
    pattern = pattern_selector("neutral_default")
    return pattern.format(asset=asset), "neutral_default"


def _handle_bearish(
    asset: str,
    dmas: int,
    technical: int,
    momentum: int,
    dmas_change: float,
    price_vs_50ma: str,
    price_vs_100ma: str,
    price_vs_200ma: str,
    pattern_selector: callable
) -> tuple[str, str]:
    """Handle bearish scenarios (DMAS < 45)."""

    # Extreme momentum (0)
    if momentum == 0:
        pattern = pattern_selector("extreme_momentum_low")
        return pattern.format(asset=asset), "extreme_momentum_low"

    # Deep bearish
    if dmas < 25:
        pattern = pattern_selector("bearish_deep")
        return pattern.format(asset=asset), "bearish_deep"

    # Both components weak
    if technical < 40 and momentum < 40:
        pattern = pattern_selector("bearish_no_hope")
        ma = 100 if price_vs_100ma == "below" else 200
        return pattern.format(asset=asset, ma=ma), "bearish_no_hope"

    # Stuck below MA
    if price_vs_50ma == "below":
        pattern = pattern_selector("bearish_stuck")
        return pattern.format(asset=asset, ma=50), "bearish_stuck"

    if price_vs_100ma == "below":
        pattern = pattern_selector("bearish_stuck")
        return pattern.format(asset=asset, ma=100), "bearish_stuck"

    # Bearish with silver lining
    if technical >= 50 or momentum >= 50:
        if price_vs_200ma == "above":
            pattern = pattern_selector("bearish_silver")
            return pattern.format(asset=asset, ma=200), "bearish_silver"

        pattern = pattern_selector("bearish_silver")
        return pattern.format(asset=asset), "bearish_silver"

    # Deteriorating
    if dmas_change < -5:
        pattern = pattern_selector("bearish_deteriorating")
        ma = 100 if price_vs_100ma == "below" else 200
        return pattern.format(asset=asset, ma=ma), "bearish_deteriorating"

    # Default bearish
    pattern = pattern_selector("bearish_default")
    return pattern.format(asset=asset), "bearish_default"
