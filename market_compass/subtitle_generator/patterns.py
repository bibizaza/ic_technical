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
# IMPORTANT: 10+ patterns per category minimum for variety

PATTERNS = {
    # =========================================================================
    # POSITIVE Rating Patterns (DMAS >= 70)
    # Language allowed: bullish, strong, excellent
    # =========================================================================
    "positive_strong": [
        "The picture remains bullish with strong momentum",
        "Momentum remains strong as well as technical",
        "{asset} has all the technical elements to go higher in the coming weeks",
        "No cloud on the technical horizon",
        "Strong technical and momentum keep DMAS high and {asset} stretched over its MA",
        "All the stars seem aligned to propel {asset} even higher",
        "Technical and momentum indicators confirm the uptrend",
        "The bullish setup remains intact with solid momentum support",
        "{asset} continues to exhibit strong technical characteristics",
        "Positive momentum underpins the strong technical picture",
        "The technical backdrop supports further gains",
        "Strong DMAS reflects aligned technical and momentum signals",
    ],

    "positive_ath": [
        "New 52-week highs are supported by strong technical and momentum",
        "At 52-week highs with the technical picture calling for more gains",
        "Trading near 52-week highs with strong momentum behind it",
        "52-week highs confirmed by robust technical indicators",
        "{asset} pushes to new highs backed by strong momentum",
    ],

    "positive_target": [
        "The bull trend is intact, {asset} could go to {target} in the coming weeks",
        "We are getting close to the first target of {target}",
        "Strong momentum propels {asset} toward the {target} target",
    ],

    "positive_continuation": [
        "As expected last week, {asset} continues its ascent",
        "Last week's call confirmed: {asset} propels higher and has still fuel",
        "The positive medium-term trend continues with strong technical setup",
        "The price action should continue its positive ascent",
        "{asset} extends its winning streak with momentum intact",
        "The upward trajectory remains well-supported technically",
        "Trend continuation confirmed by sustained momentum",
        "Technical strength persists, supporting the ongoing rally",
        "{asset} keeps pushing higher with solid technical backing",
        "The rally maintains its bullish character",
    ],

    "positive_rebound": [
        "Successful rebound on the {ma}d MA",
        "Revived momentum pushes DMAS higher and keeps {asset} marching above key averages",
        "Strong bounce off the {ma}d MA confirms the bullish picture",
        "{asset} rebounded sharply with momentum surging higher",
    ],

    # =========================================================================
    # CONSTRUCTIVE Rating Patterns (DMAS 55-69)
    # Language allowed: constructive, favorable, positive, encouraging
    # DO NOT USE: bullish
    # =========================================================================
    "constructive_caution": [
        "While the momentum is still high, the technical score calls for some caution",
        "The momentum has now switched to 100 but the technical is still calling for caution",
        "Strong momentum offset by more cautious technical readings",
        "Momentum leads but technicals suggest measured optimism",
        "High momentum warrants attention despite moderate technical score",
        "The setup is constructive but technical indicators advise patience",
        "Momentum remains elevated though technicals urge prudence",
        "Strong momentum masks some technical concerns",
    ],

    "constructive_correction": [
        "The picture remains constructive despite the current correction",
        "Despite the correction, the picture is still positive for {asset}",
        "The correction may stabilize with momentum still strong",
        "The correction has been contained thanks to a high momentum",
        "{asset} holds constructive ground despite recent pullback",
        "Correction absorbed while maintaining positive technical structure",
        "The dip has not damaged the underlying constructive setup",
        "Pullback within a constructive trend, momentum still supportive",
        "The constructive picture persists through the correction",
        "Healthy consolidation within the constructive framework",
    ],

    "constructive_improving": [
        "The technical signal is slightly lower but the picture remains constructive",
        "The global technical picture is unchanged and calls for more gains",
        "Constructive outlook maintained with room for improvement",
        "Technical picture holds steady in favorable territory",
        "The setup remains constructive with balanced indicators",
        "{asset} stays in constructive territory despite minor fluctuations",
        "Favorable technical backdrop persists",
        "The constructive trend maintains its character",
        "Positive momentum supports the constructive outlook",
        "Technical indicators remain in favorable territory",
    ],

    "constructive_near_ma": [
        "{asset} consolidates near the {ma}d MA, awaiting direction",
        "Trading near the {ma}d MA reinforces a cautiously constructive stance",
        "Proximity to the {ma}d MA suggests consolidation before next move",
        "Hovering around the {ma}d MA with constructive momentum signals",
        "The {ma}d MA acts as pivot point in this constructive setup",
        "Constructive picture intact as {asset} tests the {ma}d MA",
    ],

    # =========================================================================
    # NEUTRAL Rating Patterns (DMAS 45-54)
    # Language allowed: balanced, mixed, uncertain, consolidating
    # DO NOT USE: bullish, bearish
    # =========================================================================
    "neutral_tech_offset": [
        "The technical score is still supporting the movement despite the poor momentum",
        "High technical score is offset by weak momentum",
        "The technical score has improved but it is offset by poor momentum",
        "Technical strength contrasts with lackluster momentum",
        "Solid technicals undermined by momentum weakness",
        "Technical indicators positive but momentum fails to confirm",
        "The technical-momentum divergence keeps the picture neutral",
        "Good technicals cannot overcome weak momentum",
        "Technical support present but momentum is missing",
        "Technicals suggest potential but momentum lags",
    ],

    "neutral_mom_offset": [
        "Despite strong momentum, the technical picture has now shifted to neutral territory",
        "High momentum score may offset the current technical weakness",
        "Momentum leads while technicals lag behind",
        "Strong momentum not yet reflected in technical indicators",
        "Momentum suggests potential but technicals need confirmation",
        "The momentum-technical gap creates an uncertain outlook",
        "Momentum pushes higher but technicals remain unconvinced",
        "Strong momentum yet technical indicators stay neutral",
    ],

    "neutral_consolidation": [
        "The technical score has been stable over the past weeks",
        "Still hovering above its two moving averages in a tight channel",
        "Range-bound trading reflects the balanced technical picture",
        "Consolidation continues with neither bulls nor bears in control",
        "Technical indicators suggest a wait-and-see approach",
        "The sideways movement reflects neutral momentum and technicals",
        "Consolidation phase persists with mixed signals",
        "{asset} trades sideways awaiting a catalyst",
        "The narrow range reflects balanced forces",
        "Neutral stance maintained as consolidation extends",
    ],

    "neutral_turning": [
        "{asset} is at a turning point: technical suggest uptrend but momentum calls caution",
        "{asset} is now hovering on its support, waiting for a clear new trend",
        "No clear trend despite a robust technical score",
        "Mixed signals keep the outlook uncertain for {asset}",
        "The picture could tip either way as signals diverge",
        "A pivotal moment for {asset} with conflicting indicators",
        "Technical and momentum at odds create uncertainty",
        "Waiting for resolution of the mixed technical picture",
    ],

    "neutral_breakout_potential": [
        "{asset} could turn constructive if it manages to break above the {ma}d MA",
        "A break above the {ma}d MA would shift the picture to constructive",
        "Watch for a breakout above the {ma}d MA to confirm direction",
        "The {ma}d MA remains the key level to reclaim for a positive shift",
        "A close above the {ma}d MA would improve the technical outlook",
        "Neutral but with upside potential on a {ma}d MA breakout",
    ],

    # =========================================================================
    # CAUTIOUS Rating Patterns (DMAS 30-44)
    # Language allowed: cautious, weak, challenging, concerning
    # DO NOT USE: bullish
    # =========================================================================
    "cautious_weakening": [
        "The technicals are weakening fast and open the door to a test of the {ma}d MA",
        "The technical setup is weakening even more",
        "Technical deterioration accelerates, caution warranted",
        "Weakening technicals suggest defensive positioning",
        "The technical picture continues to erode",
        "Declining DMAS reflects ongoing technical weakness",
        "The weakening trend shows no signs of stabilizing",
        "Technicals continue to slide lower",
        "Further technical erosion cannot be ruled out",
        "The technical picture remains under pressure",
    ],

    "cautious_stuck": [
        "Still below the {ma}-day MA that seems to act as a strong resistance",
        "After the macro rebound, {asset} remains stuck below the {ma}d MA",
        "{asset} struggles to reclaim the {ma}d MA resistance",
        "The {ma}d MA caps upside attempts",
        "Failed attempts to breach the {ma}d MA keep outlook cautious",
        "Resistance at the {ma}d MA proves stubborn",
        "The {ma}d MA remains an insurmountable barrier",
        "Repeated rejections at the {ma}d MA dampen optimism",
        "Unable to break above the critical {ma}d MA",
        "{asset} stuck in a rut below the {ma}d MA",
    ],

    "cautious_silver": [
        "The picture is weak but {asset} managed to break above its 2 MA",
        "Very weak picture but {asset} has so far managed to stay above the {ma}d MA",
        "Despite a small bump in DMAS, the picture remains very negative",
        "Minor improvement but not enough to change the cautious view",
        "Small gains insufficient to shift the cautious outlook",
        "Slight improvement though challenges remain",
    ],

    "cautious_rebound": [
        "Small rebound but still not enough to trigger a new positive trend",
        "A short-lived rebound does not change the cautious picture",
        "Minor bounce fails to alter the technical weakness",
        "The tentative rebound lacks conviction",
        "Rebound attempt lacks the momentum to sustain",
        "Brief respite but the cautious view persists",
    ],

    "cautious_no_catalyst": [
        "{asset} lacks a clear catalyst to reverse the cautious setup",
        "No technical trigger visible to shift the cautious outlook",
        "The cautious picture persists without improvement signals",
        "Technical indicators remain subdued with no reversal signs",
        "Absent a catalyst, the cautious stance holds",
        "No sign of a meaningful reversal on the horizon",
    ],

    "cautious_near_52w_low": [
        "{asset} is testing 52-week lows with weak momentum",
        "Trading near 52-week lows, technical setup remains fragile",
        "Close to 52-week lows with no clear reversal signal yet",
        "Proximity to 52-week lows adds to the cautious outlook",
        "52-week lows loom as technicals remain weak",
    ],

    # =========================================================================
    # NEGATIVE Rating Patterns (DMAS < 30)
    # Language allowed: bearish, negative, weak, poor
    # DO NOT USE: bullish, constructive
    # =========================================================================
    "negative_deep": [
        "{asset} is deeply engulfed in negative territory",
        "The choppy trading range remains deeply negative",
        "Strong headwinds for {asset}",
        "{asset} remains mired in negative technical territory",
        "The bearish technical picture shows no signs of improvement",
        "Deep negative readings persist across indicators",
        "Technical weakness dominates the picture for {asset}",
        "The negative trend shows no signs of abating",
        "{asset} struggles in deeply negative conditions",
        "Bearish momentum keeps {asset} under pressure",
    ],

    "negative_no_hope": [
        "No short term hopes of a sustained rebound with challenging technical and momentum",
        "Weak technical, poor momentum and the cross below the {ma}d MA does not bode well",
        "A weak technical setup and momentum are no help for a rebound",
        "Both technical and momentum fail to offer support",
        "The dual weakness in technical and momentum is concerning",
        "Neither technicals nor momentum provide hope for recovery",
        "Challenging setup with both indicators in negative territory",
        "Weak technicals and poor momentum paint a bleak picture",
        "No technical support visible to stem the decline",
        "The negative picture persists with no catalyst for reversal",
    ],

    "negative_dramatic": [
        "A dramatic picture for {asset} that crossed all its MA in a week",
        "No respite for {asset} that crossed down all the MA",
        "Severe technical damage as all MAs have been breached",
        "The collapse through all MAs signals deep weakness",
        "Dramatic breakdown through multiple support levels",
    ],

    "negative_near_52w_low": [
        "{asset} is at 52-week lows with no sign of a bottom",
        "Trading at 52-week lows amid deep negative momentum",
        "At 52-week lows with both technical and momentum severely impaired",
        "52-week lows reached as the negative trend intensifies",
        "The slide to 52-week lows reflects severe weakness",
    ],

    # =========================================================================
    # Special Event Patterns
    # =========================================================================
    "ma_cross_up": [
        "Successful rebound on the {ma}d MA",
        "Breakout confirmed as {asset} closed above the {ma}-day MA",
        "The reclaim of the {ma}d MA is an encouraging sign",
        "Breaking above the {ma}d MA improves the technical outlook",
    ],

    "ma_cross_down": [
        "Dangerous setup: {asset} just closed below the {ma}d MA",
        "The break below the {ma}d MA is a concerning development",
        "Loss of the {ma}d MA darkens the technical picture",
    ],

    "golden_cross": [
        "Golden cross confirms the constructive setup",
        "The golden cross is a positive technical signal",
        "Golden cross formation supports the favorable outlook",
    ],

    "golden_cross_weak": [
        "{asset} shows a golden cross but DMAS remains weak",
        "Golden cross but momentum has yet to confirm",
    ],

    "death_cross": [
        "A short-lived rebound accompanied by a death cross",
        "The death cross adds to the technical concerns",
        "Death cross formation darkens the outlook",
    ],

    "dramatic_surge": [
        "A surge in momentum lifts DMAS and moves {asset} closer to bullish territory",
        "Sharp momentum improvement shifts the technical picture higher",
    ],

    "dramatic_collapse": [
        "The technical picture has worsened significantly this week",
        "A sharp deterioration in technicals this week",
        "The collapse in DMAS reflects severe technical damage",
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


# Contextual suffixes to add variety when conditions warrant
CONTEXTUAL_SUFFIXES = {
    "despite_strong_momentum": " despite strong momentum",
    "despite_weak_momentum": " despite weak momentum",
    "near_ma_breakout": " if it manages to break above the {ma}d MA",
    "ma_resistance": " with the {ma}d MA acting as resistance",
    "ma_support": " with the {ma}d MA providing support",
}


def add_context_if_needed(
    subtitle: str,
    asset_data: dict,
    max_words: int = 15
) -> str:
    """
    Add contextual suffix if conditions warrant and word count allows.

    This adds variety and specificity to subtitles by appending
    context-specific phrases when appropriate.

    Parameters
    ----------
    subtitle : str
        Current subtitle text
    asset_data : dict
        Asset data with dmas, technical_score, momentum_score, etc.
    max_words : int, default=15
        Maximum words allowed (per directive)

    Returns
    -------
    str
        Subtitle with optional contextual suffix
    """
    dmas = asset_data.get("dmas", 50)
    tech = asset_data.get("technical_score", 50)
    mom = asset_data.get("momentum_score", 50)
    ma_dynamics = get_ma_dynamics(asset_data)

    # Check word count budget
    current_words = len(subtitle.split())
    if current_words >= max_words - 3:
        return subtitle  # No room for suffix

    # Divergence: High momentum, lower DMAS (Neutral/Cautious with strong momentum)
    if mom >= 70 and dmas < 55:
        # Check if not already mentioned in subtitle
        if "momentum" not in subtitle.lower():
            suffix = CONTEXTUAL_SUFFIXES["despite_strong_momentum"]
            if current_words + len(suffix.split()) <= max_words:
                return subtitle.rstrip('.!?') + suffix

    # Divergence: Low momentum, higher tech
    if mom < 40 and tech >= 60 and dmas >= 45:
        if "momentum" not in subtitle.lower():
            suffix = CONTEXTUAL_SUFFIXES["despite_weak_momentum"]
            if current_words + len(suffix.split()) <= max_words:
                return subtitle.rstrip('.!?') + suffix

    return subtitle
