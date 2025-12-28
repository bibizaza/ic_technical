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


def _classify_ma_distance(pct: float) -> str:
    """Classify distance from MA into buckets."""
    if pct > 5:
        return "FAR_ABOVE"
    elif pct > 2:
        return "ABOVE"
    elif pct > 0.5:
        return "TESTING_FROM_ABOVE"
    elif pct >= -0.5:
        return "AT"
    elif pct >= -2:
        return "TESTING_FROM_BELOW"
    elif pct >= -5:
        return "BELOW"
    else:
        return "FAR_BELOW"


def _classify_ma_direction(current: float, previous: float) -> str:
    """Classify movement direction relative to MA."""
    delta = current - previous

    # Crosses
    if previous < -0.5 and current > 0.5:
        return "BREAKING_ABOVE"
    if previous > 0.5 and current < -0.5:
        return "BREAKING_BELOW"

    # Bounces and rejections (was near, now moved)
    if abs(previous) <= 2 and current > 2 and delta > 1:
        return "BOUNCING_UP"
    if abs(previous) <= 2 and current < -2 and delta < -1:
        return "REJECTED_DOWN"

    # Approaches
    if current < 0 and delta > 1:
        return "APPROACHING_UP"
    if current > 0 and delta < -1:
        return "APPROACHING_DOWN"

    # Extensions
    if current > 2 and delta > 1:
        return "EXTENDING_UP"
    if current < -2 and delta < -1:
        return "DRIFTING_DOWN"

    return "STABLE"


def _get_ma_fact(
    ma_period: int,
    current_pct: float,
    previous_pct: float
) -> Optional[str]:
    """
    Generate specific MA fact based on position and direction.

    Returns a single, specific fact about this MA.
    """
    distance = _classify_ma_distance(current_pct)
    direction = _classify_ma_direction(current_pct, previous_pct)

    ma = f"{ma_period}d MA"

    # CROSSES - Highest priority, most newsworthy
    if direction == "BREAKING_ABOVE":
        return f"Just broke above the {ma}"
    if direction == "BREAKING_BELOW":
        return f"Just broke below the {ma}"

    # BOUNCES AND REJECTIONS
    if direction == "BOUNCING_UP":
        return f"Successfully bounced off the {ma}"
    if direction == "REJECTED_DOWN":
        return f"Rejected at the {ma}, now falling away"

    # TESTING scenarios (within 2%)
    if distance == "AT":
        return f"Trading right at the {ma}"
    if distance == "TESTING_FROM_ABOVE":
        if direction == "APPROACHING_DOWN":
            return f"Pulling back to test the {ma} from above"
        return f"Hovering just above the {ma}"
    if distance == "TESTING_FROM_BELOW":
        if direction == "APPROACHING_UP":
            return f"Approaching the {ma} resistance from below"
        return f"Struggling just below the {ma}"

    # EXTENDED scenarios
    if distance == "FAR_ABOVE":
        if direction == "EXTENDING_UP":
            return f"Extending gains well above the {ma} (+{current_pct:.1f}%)"
        return f"Well above the {ma} (+{current_pct:.1f}%)"

    if distance == "FAR_BELOW":
        if direction == "DRIFTING_DOWN":
            return f"Drifting further below the {ma} ({current_pct:.1f}%)"
        return f"Well below the {ma} ({current_pct:.1f}%)"

    # MODERATE distance
    if distance == "ABOVE":
        if direction == "APPROACHING_DOWN":
            return f"Retreating toward the {ma} support"
        if direction == "EXTENDING_UP":
            return f"Building distance above the {ma}"
        return f"Above the {ma} (+{current_pct:.1f}%)"

    if distance == "BELOW":
        if direction == "APPROACHING_UP":
            return f"Recovering toward the {ma}"
        if direction == "DRIFTING_DOWN":
            return f"Slipping further below the {ma}"
        return f"Below the {ma} ({current_pct:.1f}%)"

    return None


def _get_enhanced_ma_facts(
    price_vs_50ma: float,
    price_vs_100ma: float,
    price_vs_200ma: float,
    price_vs_50ma_prev: float,
    price_vs_100ma_prev: float,
    price_vs_200ma_prev: float
) -> List[str]:
    """
    Extract enhanced MA facts with specific terminology.

    Only includes the MOST RELEVANT MAs to avoid clutter.
    """
    facts = []

    # Always include 50d MA fact (most important short-term)
    fact_50 = _get_ma_fact(50, price_vs_50ma, price_vs_50ma_prev)
    if fact_50:
        facts.append(fact_50)

    # Include 100d MA only if:
    # - Price is between 50d and 100d (sandwich)
    # - OR there's significant action at 100d (testing/crossing)
    dist_100 = _classify_ma_distance(price_vs_100ma)
    dir_100 = _classify_ma_direction(price_vs_100ma, price_vs_100ma_prev)

    between_50_100 = price_vs_50ma < -0.5 and price_vs_100ma > 0.5
    action_at_100 = dist_100 in ["AT", "TESTING_FROM_ABOVE", "TESTING_FROM_BELOW"] or \
                    dir_100 in ["BREAKING_ABOVE", "BREAKING_BELOW", "BOUNCING_UP", "REJECTED_DOWN"]

    if between_50_100:
        facts.append("Caught between 50d MA resistance and 100d MA support")
    elif action_at_100:
        fact_100 = _get_ma_fact(100, price_vs_100ma, price_vs_100ma_prev)
        if fact_100:
            facts.append(fact_100)

    # Include 200d MA only if:
    # - Price is near it (testing)
    # - OR price is below it (critical support lost)
    # - OR there's crossing action
    dist_200 = _classify_ma_distance(price_vs_200ma)
    dir_200 = _classify_ma_direction(price_vs_200ma, price_vs_200ma_prev)

    critical_200 = dist_200 in ["AT", "TESTING_FROM_ABOVE", "TESTING_FROM_BELOW", "BELOW", "FAR_BELOW"] or \
                   dir_200 in ["BREAKING_ABOVE", "BREAKING_BELOW"]

    if critical_200:
        fact_200 = _get_ma_fact(200, price_vs_200ma, price_vs_200ma_prev)
        if fact_200:
            facts.append(fact_200)

    # Add overall MA alignment summary if relevant
    all_above = price_vs_50ma > 2 and price_vs_100ma > 2 and price_vs_200ma > 2
    all_below = price_vs_50ma < -2 and price_vs_100ma < -2 and price_vs_200ma < -2

    if all_above and not any("well above" in f.lower() for f in facts):
        facts.append("Positioned above all major moving averages")
    elif all_below and not any("well below" in f.lower() for f in facts):
        facts.append("Trading below all major moving averages")

    return facts


def _get_ma_structure_fact(
    price_vs_50ma: float,
    price_vs_100ma: float,
    price_vs_200ma: float
) -> Optional[str]:
    """Describe the overall MA structure."""

    # Bullish alignment: price > 50 > 100 > 200
    if price_vs_50ma > 2 and price_vs_100ma > price_vs_200ma > 0:
        return "Bullish MA alignment supports the trend"

    # Bearish alignment: price < 50 < 100 < 200
    if price_vs_50ma < -2 and price_vs_100ma < price_vs_200ma < 0:
        return "Bearish MA alignment confirms the downtrend"

    # Compression: all MAs close together
    spread = max(price_vs_50ma, price_vs_100ma, price_vs_200ma) - \
             min(price_vs_50ma, price_vs_100ma, price_vs_200ma)
    if spread < 5:
        return "Moving averages are compressed, suggesting potential breakout"

    return None


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
    price_vs_200ma_prev = asset_data.get("price_vs_200ma_pct_prev", price_vs_200ma)

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

    # ENHANCED MA FACTS (v3.2 - replaces old MA functions)
    ma_facts = _get_enhanced_ma_facts(
        price_vs_50ma, price_vs_100ma, price_vs_200ma,
        price_vs_50ma_prev, price_vs_100ma_prev, price_vs_200ma_prev
    )
    facts.extend(ma_facts)

    # MA structure (optional)
    structure_fact = _get_ma_structure_fact(price_vs_50ma, price_vs_100ma, price_vs_200ma)
    if structure_fact:
        facts.append(structure_fact)

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
