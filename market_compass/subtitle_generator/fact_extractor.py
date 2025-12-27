"""
Fact Extraction Module for Market Compass Subtitle Generation.

Extracts key market facts from technical scores and MA positions
to provide structured input for Claude API subtitle generation.

This is the DETERMINISTIC part of the hybrid system.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class Rating(Enum):
    BULLISH = "Bullish"          # Was "Positive"
    CONSTRUCTIVE = "Constructive"
    NEUTRAL = "Neutral"
    CAUTIOUS = "Cautious"
    BEARISH = "Bearish"          # Was "Negative"


@dataclass
class MarketFacts:
    """Structured container for extracted market facts."""
    asset_name: str
    rating: Rating
    facts: List[str]
    primary_condition: str  # Most important fact
    dmas: int
    technical_score: int
    momentum_score: int


def get_rating(dmas: int) -> Rating:
    """Determine rating from DMAS score."""
    if dmas >= 70:
        return Rating.BULLISH
    elif dmas >= 55:
        return Rating.CONSTRUCTIVE
    elif dmas >= 45:
        return Rating.NEUTRAL
    elif dmas >= 30:
        return Rating.CAUTIOUS
    else:
        return Rating.BEARISH


def _get_momentum_fact(momentum: int) -> Optional[str]:
    """Extract momentum-related fact."""
    if momentum >= 85:
        return f"Very strong momentum ({momentum})"
    elif momentum >= 70:
        return f"Strong momentum ({momentum})"
    elif momentum >= 55:
        return f"Moderate-to-strong momentum ({momentum})"
    elif momentum >= 40:
        return f"Moderate momentum ({momentum})"
    elif momentum >= 25:
        return f"Weak momentum ({momentum})"
    else:
        return f"Very weak momentum ({momentum})"


def _get_technical_fact(technical: int) -> Optional[str]:
    """Extract technical-related fact."""
    if technical >= 85:
        return f"Very strong technical score ({technical})"
    elif technical >= 70:
        return f"Strong technical score ({technical})"
    elif technical >= 55:
        return f"Moderate-to-strong technical score ({technical})"
    elif technical >= 40:
        return f"Moderate technical score ({technical})"
    elif technical >= 25:
        return f"Weak technical score ({technical})"
    else:
        return f"Very weak technical score ({technical})"


def _get_divergence_fact(momentum: int, technical: int) -> Optional[str]:
    """Detect momentum-technical divergence."""
    diff = momentum - technical

    if diff >= 30:
        return "Strong divergence: momentum leads technical significantly"
    elif diff >= 15:
        return "Momentum leads technical"
    elif diff <= -30:
        return "Strong divergence: technical leads momentum significantly"
    elif diff <= -15:
        return "Technical leads momentum"
    else:
        return None  # No significant divergence


def _get_ma_position_facts(
    price_vs_50ma: float,
    price_vs_100ma: float,
    price_vs_200ma: float
) -> List[str]:
    """Extract MA position facts."""
    facts = []

    # 50d MA position
    if price_vs_50ma > 5:
        facts.append(f"Well above 50d MA (+{price_vs_50ma:.1f}%)")
    elif price_vs_50ma > 2:
        facts.append(f"Above 50d MA (+{price_vs_50ma:.1f}%)")
    elif price_vs_50ma >= -2:
        facts.append(f"At the 50d MA ({price_vs_50ma:+.1f}%)")
    elif price_vs_50ma >= -5:
        facts.append(f"Below 50d MA ({price_vs_50ma:.1f}%)")
    else:
        facts.append(f"Well below 50d MA ({price_vs_50ma:.1f}%)")

    # 100d MA position (only if different story from 50d)
    if price_vs_100ma > 2 and price_vs_50ma < -2:
        facts.append("Between 50d and 100d MA")
    elif price_vs_100ma >= -2 and price_vs_100ma <= 2:
        facts.append(f"At the 100d MA ({price_vs_100ma:+.1f}%)")
    elif price_vs_100ma < -2 and price_vs_50ma < -2:
        facts.append(f"Below 100d MA ({price_vs_100ma:.1f}%)")

    # 200d MA position (only if critical)
    if price_vs_200ma >= -2 and price_vs_200ma <= 2:
        facts.append(f"Testing the 200d MA ({price_vs_200ma:+.1f}%)")
    elif price_vs_200ma < -2:
        facts.append(f"Below 200d MA ({price_vs_200ma:.1f}%)")

    # All MAs alignment
    if price_vs_50ma > 2 and price_vs_100ma > 2 and price_vs_200ma > 2:
        facts.append("Above all major moving averages")
    elif price_vs_50ma < -2 and price_vs_100ma < -2 and price_vs_200ma < -2:
        facts.append("Below all major moving averages")

    return facts


def _get_ma_dynamics_facts(
    price_vs_50ma: float,
    price_vs_50ma_prev: float,
    price_vs_100ma: float,
    price_vs_100ma_prev: float
) -> List[str]:
    """Extract MA dynamics (movement relative to MAs)."""
    facts = []

    delta_50 = price_vs_50ma - price_vs_50ma_prev
    delta_100 = price_vs_100ma - price_vs_100ma_prev

    # 50d MA dynamics
    if price_vs_50ma_prev < -2 and price_vs_50ma >= -2:
        facts.append("Just crossed above 50d MA")
    elif price_vs_50ma_prev >= -2 and price_vs_50ma < -2:
        facts.append("Just crossed below 50d MA")
    elif price_vs_50ma < 0 and delta_50 > 2:
        facts.append("Approaching 50d MA from below")
    elif price_vs_50ma > 0 and delta_50 < -2:
        facts.append("Pulling back toward 50d MA")
    elif price_vs_50ma < -2 and delta_50 < -2:
        facts.append("Moving further below 50d MA")
    elif price_vs_50ma > 2 and delta_50 > 2:
        facts.append("Extending gains above 50d MA")

    # 100d MA dynamics
    if price_vs_100ma_prev < -2 and price_vs_100ma >= -2:
        facts.append("Rebounding on 100d MA")
    elif price_vs_100ma_prev >= -2 and price_vs_100ma < -2:
        facts.append("Breaking below 100d MA")
    elif price_vs_100ma >= -2 and price_vs_100ma <= 2 and abs(delta_100) < 1:
        facts.append("Consolidating at 100d MA")

    return facts


def _get_dmas_change_fact(dmas: int, dmas_prev: int) -> Optional[str]:
    """Extract DMAS week-over-week change fact."""
    if dmas_prev is None:
        return None

    change = dmas - dmas_prev

    if change >= 15:
        return f"DMAS surged (+{change} points WoW)"
    elif change >= 8:
        return f"DMAS improved significantly (+{change} points WoW)"
    elif change >= 3:
        return f"DMAS improving (+{change} points WoW)"
    elif change <= -15:
        return f"DMAS collapsed ({change} points WoW)"
    elif change <= -8:
        return f"DMAS deteriorated significantly ({change} points WoW)"
    elif change <= -3:
        return f"DMAS weakening ({change} points WoW)"
    else:
        return None  # Stable


def _get_special_condition_facts(
    at_52w_high: bool,
    at_52w_low: bool,
    ma_cross_event: Optional[str]
) -> List[str]:
    """Extract special condition facts."""
    facts = []

    if at_52w_high:
        facts.append("At or near 52-week high")
    if at_52w_low:
        facts.append("At or near 52-week low")

    if ma_cross_event == "golden_cross":
        facts.append("Golden cross (50d crossed above 200d)")
    elif ma_cross_event == "death_cross":
        facts.append("Death cross (50d crossed below 200d)")

    return facts


def _determine_primary_condition(
    rating: Rating,
    facts: List[str],
    momentum: int,
    technical: int,
    price_vs_50ma: float,
    dmas_change: Optional[int]
) -> str:
    """Determine the most important fact to highlight."""

    # ALWAYS include MA position in primary condition
    ma_context = ""
    if price_vs_50ma < -2:
        ma_context = " while below the 50d MA"
    elif price_vs_50ma > 2:
        ma_context = " above the 50d MA"
    else:
        ma_context = " at the 50d MA"

    # Special events take priority (but keep MA context)
    for fact in facts:
        if "52-week" in fact:
            return fact + ma_context
        if "cross" in fact.lower() and ("golden" in fact.lower() or "death" in fact.lower()):
            return fact
        if "Just crossed" in fact:
            return fact

    # Dramatic DMAS change
    if dmas_change is not None and abs(dmas_change) >= 10:
        for fact in facts:
            if "DMAS" in fact:
                return fact + ma_context

    # Divergence with MA context
    for fact in facts:
        if "divergence" in fact.lower():
            return fact + ma_context

    # MA dynamics
    for fact in facts:
        if "Rebounding" in fact or "Breaking" in fact:
            return fact

    # Default: emphasize MA position
    if rating in [Rating.BULLISH, Rating.CONSTRUCTIVE]:
        if momentum >= 70:
            return f"Strong momentum ({momentum}){ma_context}"
        elif technical >= 60:
            return f"Solid technical ({technical}){ma_context}"
    else:
        if price_vs_50ma < -2:
            return f"Below the 50d MA with {rating.value.lower()} outlook"
        elif momentum < 40:
            return f"Weak momentum ({momentum}){ma_context}"
        elif technical < 40:
            return f"Poor technical ({technical}){ma_context}"

    return f"{rating.value} outlook{ma_context}"


def extract_facts(asset_data: dict) -> MarketFacts:
    """
    Extract all relevant market facts from asset data.

    Parameters
    ----------
    asset_data : dict
        Asset data with keys:
        - asset_name (str)
        - dmas (int): 0-100
        - technical_score (int): 0-100
        - momentum_score (int): 0-100
        - dmas_prev_week (int): Previous DMAS
        - price_vs_50ma_pct (float)
        - price_vs_100ma_pct (float)
        - price_vs_200ma_pct (float)
        - price_vs_50ma_pct_prev (float): Optional, previous week
        - price_vs_100ma_pct_prev (float): Optional, previous week
        - at_52w_high (bool): Optional
        - at_52w_low (bool): Optional
        - ma_cross_event (str): Optional

    Returns
    -------
    MarketFacts
        Structured facts for Claude API prompt
    """
    asset_name = asset_data["asset_name"]
    dmas = asset_data["dmas"]
    technical = asset_data["technical_score"]
    momentum = asset_data["momentum_score"]
    dmas_prev = asset_data.get("dmas_prev_week")

    price_vs_50ma = asset_data.get("price_vs_50ma_pct", 0)
    price_vs_100ma = asset_data.get("price_vs_100ma_pct", 0)
    price_vs_200ma = asset_data.get("price_vs_200ma_pct", 0)
    price_vs_50ma_prev = asset_data.get("price_vs_50ma_pct_prev", price_vs_50ma)
    price_vs_100ma_prev = asset_data.get("price_vs_100ma_pct_prev", price_vs_100ma)

    at_52w_high = asset_data.get("at_52w_high", False) or asset_data.get("at_ath", False)
    at_52w_low = asset_data.get("at_52w_low", False) or asset_data.get("near_52w_low", False)
    ma_cross_event = asset_data.get("ma_cross_event")

    # Get rating
    rating = get_rating(dmas)

    # Collect all facts
    facts = []

    # Momentum fact
    mom_fact = _get_momentum_fact(momentum)
    if mom_fact:
        facts.append(mom_fact)

    # Technical fact
    tech_fact = _get_technical_fact(technical)
    if tech_fact:
        facts.append(tech_fact)

    # Divergence
    div_fact = _get_divergence_fact(momentum, technical)
    if div_fact:
        facts.append(div_fact)

    # MA position
    ma_pos_facts = _get_ma_position_facts(price_vs_50ma, price_vs_100ma, price_vs_200ma)
    facts.extend(ma_pos_facts)

    # MA dynamics
    ma_dyn_facts = _get_ma_dynamics_facts(
        price_vs_50ma, price_vs_50ma_prev,
        price_vs_100ma, price_vs_100ma_prev
    )
    facts.extend(ma_dyn_facts)

    # DMAS change
    dmas_chg_fact = _get_dmas_change_fact(dmas, dmas_prev)
    if dmas_chg_fact:
        facts.append(dmas_chg_fact)

    # Special conditions
    special_facts = _get_special_condition_facts(at_52w_high, at_52w_low, ma_cross_event)
    facts.extend(special_facts)

    # Determine primary condition
    dmas_change = (dmas - dmas_prev) if dmas_prev else None
    primary = _determine_primary_condition(
        rating, facts, momentum, technical, price_vs_50ma, dmas_change
    )

    return MarketFacts(
        asset_name=asset_name,
        rating=rating,
        facts=facts,
        primary_condition=primary,
        dmas=dmas,
        technical_score=technical,
        momentum_score=momentum
    )


def format_facts_for_prompt(market_facts: MarketFacts) -> str:
    """
    Format extracted facts for Claude API prompt.

    Parameters
    ----------
    market_facts : MarketFacts
        Extracted market facts

    Returns
    -------
    str
        Formatted string for inclusion in prompt
    """
    lines = [
        f"Asset: {market_facts.asset_name}",
        f"Rating: {market_facts.rating.value}",
        f"DMAS: {market_facts.dmas} | Technical: {market_facts.technical_score} | Momentum: {market_facts.momentum_score}",
        "",
        "Key facts:",
    ]

    for fact in market_facts.facts:
        lines.append(f"  - {fact}")

    lines.append("")
    lines.append(f"Primary condition: {market_facts.primary_condition}")

    return "\n".join(lines)


# Convenience function for testing
def extract_and_format(asset_data: dict) -> str:
    """Extract facts and format for prompt in one call."""
    facts = extract_facts(asset_data)
    return format_facts_for_prompt(facts)
