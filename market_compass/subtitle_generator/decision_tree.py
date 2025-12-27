"""
Decision tree logic for subtitle generation.

Implements priority-based routing to select the most appropriate
pattern based on asset data.

Priority order:
1. MA cross events (most newsworthy)
2. Dramatic WoW changes (|delta| > 10)
3. Rating-based pattern selection (Positive/Constructive/Neutral/Cautious/Negative)
"""

from typing import Optional, Callable, Tuple
from .patterns import get_ma_number, get_ma_dynamics, get_relevant_ma, get_rating


def route_subtitle(
    asset_data: dict,
    pattern_selector: Callable[[str], str]
) -> Tuple[str, str]:
    """
    Route to appropriate subtitle pattern based on asset data.

    Processes in priority order:
    1. MA cross events (most newsworthy)
    2. Dramatic WoW changes (|delta| > 10)
    3. Rating-based routing (Positive/Constructive/Neutral/Cautious/Negative)

    Parameters
    ----------
    asset_data : dict
        Asset data dictionary with:
        - asset_name (str)
        - dmas (int): 0-100
        - technical_score (int): 0-100
        - momentum_score (int): 0-100
        - dmas_prev_week (int): Previous week DMAS
        - ma_cross_event (str | None): golden_cross, death_cross, crossed_above_*, crossed_below_*
        - price_vs_50ma_pct (float): % above/below 50MA
        - price_vs_100ma_pct (float): % above/below 100MA
        - price_vs_200ma_pct (float): % above/below 200MA
        - at_ath (bool): At all-time high
        - near_support (bool): Near support level
        - near_resistance (bool): Near resistance level
        - price_target (float | None): Optional price target

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

    dmas_change = dmas - dmas_prev

    # Get MA dynamics
    ma_dynamics = get_ma_dynamics(asset_data)

    # Get rating based on DMAS
    rating = get_rating(dmas, technical, momentum)

    # PRIORITY 1: MA Cross Events
    if ma_cross:
        return _handle_ma_cross(
            asset, ma_cross, dmas, momentum, technical,
            pattern_selector
        )

    # PRIORITY 2: Dramatic WoW Changes (>10 points per directive)
    if abs(dmas_change) > 10:
        return _handle_dramatic_change(
            asset, dmas_change, pattern_selector
        )

    # PRIORITY 3: Route by Rating
    if rating == "Positive":
        return _handle_positive(
            asset, dmas, technical, momentum, dmas_change,
            asset_data, ma_dynamics, pattern_selector
        )
    elif rating == "Constructive":
        return _handle_constructive(
            asset, dmas, technical, momentum, dmas_change,
            asset_data, ma_dynamics, pattern_selector
        )
    elif rating == "Neutral":
        return _handle_neutral(
            asset, dmas, technical, momentum, dmas_change,
            asset_data, ma_dynamics, pattern_selector
        )
    elif rating == "Cautious":
        return _handle_cautious(
            asset, dmas, technical, momentum, dmas_change,
            asset_data, ma_dynamics, pattern_selector
        )
    else:  # Negative
        return _handle_negative(
            asset, dmas, technical, momentum, dmas_change,
            asset_data, ma_dynamics, pattern_selector
        )


def _handle_ma_cross(
    asset: str,
    ma_cross: str,
    dmas: int,
    momentum: int,
    technical: int,
    pattern_selector: Callable[[str], str]
) -> Tuple[str, str]:
    """Handle MA cross events (Priority 1)."""

    if ma_cross == "death_cross":
        pattern = pattern_selector("death_cross")
        return pattern.format(asset=asset), "death_cross"

    if ma_cross == "golden_cross":
        if dmas < 50:
            pattern = pattern_selector("golden_cross_weak")
            return pattern.format(asset=asset), "golden_cross_weak"
        else:
            pattern = pattern_selector("golden_cross")
            return pattern.format(asset=asset), "golden_cross"

    if "crossed_above" in ma_cross:
        ma_num = get_ma_number(ma_cross)
        pattern = pattern_selector("ma_cross_up")
        if momentum < 40:
            # Use the "despite poor momentum" variant
            return f"Breakout confirmed! Despite poor momentum, {asset} closed above {ma_num}-day MA", "ma_cross_up"
        return pattern.format(asset=asset, ma=ma_num), "ma_cross_up"

    if "crossed_below" in ma_cross:
        ma_num = get_ma_number(ma_cross)
        pattern = pattern_selector("ma_cross_down")
        return pattern.format(asset=asset, ma=ma_num), "ma_cross_down"

    # Fallback - shouldn't reach here
    return _handle_positive(
        asset, dmas, technical, momentum, 0,
        {}, {}, pattern_selector
    )


def _handle_dramatic_change(
    asset: str,
    dmas_change: float,
    pattern_selector: Callable[[str], str]
) -> Tuple[str, str]:
    """Handle dramatic WoW changes (Priority 2)."""

    if dmas_change > 10:
        pattern = pattern_selector("dramatic_surge")
        return pattern.format(asset=asset), "dramatic_surge"
    else:  # dmas_change < -10
        pattern = pattern_selector("dramatic_collapse")
        return pattern.format(asset=asset), "dramatic_collapse"


def _handle_positive(
    asset: str,
    dmas: int,
    technical: int,
    momentum: int,
    dmas_change: float,
    asset_data: dict,
    ma_dynamics: dict,
    pattern_selector: Callable[[str], str]
) -> Tuple[str, str]:
    """Handle Positive rating scenarios (DMAS >= 70)."""

    at_ath = asset_data.get("at_ath", False)
    price_target = asset_data.get("price_target")
    near_resistance = asset_data.get("near_resistance", False)

    # At all-time high
    if at_ath:
        pattern = pattern_selector("positive_ath")
        return pattern.format(asset=asset), "positive_ath"

    # Near target
    if near_resistance and price_target:
        pattern = pattern_selector("positive_target")
        return pattern.format(asset=asset, target=int(price_target)), "positive_target"

    # Strong momentum and technical
    if technical >= 60 and momentum >= 80:
        pattern = pattern_selector("positive_strong")
        return pattern.format(asset=asset), "positive_strong"

    # Rebound scenario
    if dmas_change > 5 and ma_dynamics.get("above_50ma"):
        pattern = pattern_selector("positive_rebound")
        ma = get_relevant_ma(asset_data)
        return pattern.format(asset=asset, ma=ma), "positive_rebound"

    # Continuation
    if dmas_change >= 0:
        pattern = pattern_selector("positive_continuation")
        return pattern.format(asset=asset), "positive_continuation"

    # Default positive
    pattern = pattern_selector("positive_strong")
    return pattern.format(asset=asset), "positive_strong"


def _handle_constructive(
    asset: str,
    dmas: int,
    technical: int,
    momentum: int,
    dmas_change: float,
    asset_data: dict,
    ma_dynamics: dict,
    pattern_selector: Callable[[str], str]
) -> Tuple[str, str]:
    """Handle Constructive rating scenarios (DMAS 55-69)."""

    # High momentum but technical calls for caution
    if momentum >= 80 and technical < 55:
        pattern = pattern_selector("constructive_caution")
        return pattern.format(asset=asset), "constructive_caution"

    # Correction scenario (DMAS dropped but still constructive)
    if dmas_change < -3:
        pattern = pattern_selector("constructive_correction")
        return pattern.format(asset=asset), "constructive_correction"

    # Improving
    pattern = pattern_selector("constructive_improving")
    return pattern.format(asset=asset), "constructive_improving"


def _handle_neutral(
    asset: str,
    dmas: int,
    technical: int,
    momentum: int,
    dmas_change: float,
    asset_data: dict,
    ma_dynamics: dict,
    pattern_selector: Callable[[str], str]
) -> Tuple[str, str]:
    """Handle Neutral rating scenarios (DMAS 45-54)."""

    near_support = asset_data.get("near_support", False)
    near_resistance = asset_data.get("near_resistance", False)

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
        # Check if at turning point
        if near_support or near_resistance or ma_dynamics.get("at_50ma"):
            pattern = pattern_selector("neutral_turning")
            return pattern.format(asset=asset), "neutral_turning"

        pattern = pattern_selector("neutral_consolidation")
        return pattern.format(asset=asset), "neutral_consolidation"

    # Default - turning point
    pattern = pattern_selector("neutral_turning")
    return pattern.format(asset=asset), "neutral_turning"


def _handle_cautious(
    asset: str,
    dmas: int,
    technical: int,
    momentum: int,
    dmas_change: float,
    asset_data: dict,
    ma_dynamics: dict,
    pattern_selector: Callable[[str], str]
) -> Tuple[str, str]:
    """Handle Cautious rating scenarios (DMAS 30-44)."""

    ma = get_relevant_ma(asset_data)

    # Weakening scenario
    if dmas_change < -5:
        pattern = pattern_selector("cautious_weakening")
        return pattern.format(asset=asset, ma=ma), "cautious_weakening"

    # Stuck below MA
    if not ma_dynamics.get("above_50ma", True):
        pattern = pattern_selector("cautious_stuck")
        return pattern.format(asset=asset, ma=50), "cautious_stuck"

    if not ma_dynamics.get("above_100ma", True):
        pattern = pattern_selector("cautious_stuck")
        return pattern.format(asset=asset, ma=100), "cautious_stuck"

    # Small rebound but not enough
    if dmas_change > 0:
        pattern = pattern_selector("cautious_rebound")
        return pattern.format(asset=asset), "cautious_rebound"

    # Silver lining - managed to stay above some MA
    if ma_dynamics.get("above_200ma"):
        pattern = pattern_selector("cautious_silver")
        return pattern.format(asset=asset, ma=200), "cautious_silver"

    # Default cautious
    pattern = pattern_selector("cautious_weakening")
    return pattern.format(asset=asset, ma=ma), "cautious_weakening"


def _handle_negative(
    asset: str,
    dmas: int,
    technical: int,
    momentum: int,
    dmas_change: float,
    asset_data: dict,
    ma_dynamics: dict,
    pattern_selector: Callable[[str], str]
) -> Tuple[str, str]:
    """Handle Negative rating scenarios (DMAS < 30)."""

    ma = get_relevant_ma(asset_data)

    # Both components very weak
    if technical < 30 and momentum < 30:
        pattern = pattern_selector("negative_no_hope")
        return pattern.format(asset=asset, ma=ma), "negative_no_hope"

    # Dramatic fall (crossed all MAs)
    if ma_dynamics.get("far_below_all"):
        pattern = pattern_selector("negative_dramatic")
        return pattern.format(asset=asset), "negative_dramatic"

    # Deep negative
    if dmas < 20:
        pattern = pattern_selector("negative_deep")
        return pattern.format(asset=asset), "negative_deep"

    # Default negative
    pattern = pattern_selector("negative_no_hope")
    return pattern.format(asset=asset, ma=ma), "negative_no_hope"
