"""
Pattern templates for subtitle generation.

Contains all the template variations organized by category
to ensure variety and avoid repetition.
"""


def get_rating(dmas: int) -> str:
    """
    Convert DMAS score to rating label.

    Parameters
    ----------
    dmas : int
        DMAS score (0-100)

    Returns
    -------
    str
        Rating label
    """
    if dmas <= 20:
        return "Strongly Bearish"
    elif dmas <= 35:
        return "Bearish"
    elif dmas <= 45:
        return "Slightly Bearish"
    elif dmas <= 55:
        return "Neutral"
    elif dmas <= 65:
        return "Slightly Bullish"
    elif dmas <= 80:
        return "Bullish"
    else:
        return "Strongly Bullish"


# Pattern templates organized by category
# Use {asset}, {ma}, {target}, {ordinal} as placeholders
PATTERNS = {
    "bullish_strong": [
        "The picture remains bullish with strong momentum",
        "Momentum remains strong as well as technical. The setup is clean and bullish",
        "{asset} has all the technical elements to go higher in the coming weeks",
        "No cloud on the technical horizon",
        "Strong scores highlight the potential for more gains",
        "The technical setup still calls for more gains"
    ],

    "bullish_ath": [
        "New all-time highs are supported by strong technical and momentum",
        "{asset} is technically set for new all-time high",
        "The global technical picture is unchanged and calls for more gains"
    ],

    "bullish_target": [
        "The bull trend is intact, {asset} could go to {target} in the coming weeks",
        "We are getting close to the {ordinal} target of {target}",
        "Technical suggests {target} may be the next level"
    ],

    "bullish_caution": [
        "While the momentum is still high, the technical score has weakened, calling for some caution",
        "The momentum has now switched to 100 but the technical is still calling for caution",
        "The momentum is still high but the technical is still calling for caution"
    ],

    "bullish_correction": [
        "The picture remains bullish despite the current correction",
        "Despite the correction, the picture is still positive for {asset}",
        "The correction has been contained thanks to a high momentum",
        "Despite the correction, the technical picture remains attractive"
    ],

    "neutral_tech_offset": [
        "The technical score is still supporting the bull movement despite the poor momentum",
        "High technical score is offset by weak momentum",
        "The technical score has improved to bull level but it is offset by poor momentum"
    ],

    "neutral_mom_offset": [
        "Despite a strong momentum, the technical picture has now shifted to neutral territory",
        "High momentum score may offset the current technical weakness",
        "The score remains high thanks to momentum"
    ],

    "neutral_consolidation": [
        "The technical score has been stable over the past weeks. Ongoing consolidation",
        "The market remains stable while the technical score slowly improves",
        "The DMAS is high but {asset} is consolidating at current levels"
    ],

    "neutral_turning": [
        "{asset} is at a turning point: technical suggest uptrend but momentum calls caution",
        "A turning point for {asset} that remains stuck in a tight channel",
        "{asset} is now hovering on its support, waiting for a clear new trend"
    ],

    "neutral_default": [
        "No clear trend despite a robust technical score",
        "The technical picture remains mixed",
        "{asset} continues to trade in a range"
    ],

    "bearish_deep": [
        "{asset} is deeply engulfed in negative territory",
        "The choppy trading range remains deeply negative",
        "Strong headwinds for {asset}"
    ],

    "bearish_no_hope": [
        "No short term hopes of a sustained rebound with challenging technical and momentum",
        "Weak technical, poor momentum and the cross below the {ma}d MA does not bode well",
        "A weak technical setup and momentum are no help for a rebound"
    ],

    "bearish_stuck": [
        "Still below the {ma}-day MA that seems to act as a strong resistance",
        "{asset} is trying to push above its {ma}d MA, unsuccessfully for now",
        "After the macro rebound, {asset} remains stuck below the {ma}d MA"
    ],

    "bearish_silver": [
        "The picture is weak but {asset} managed to break above its 2 MA",
        "Very weak picture but {asset} has so far managed to stay above the {ma}d MA",
        "Despite a small bump in DMAS, the picture remains very negative"
    ],

    "bearish_deteriorating": [
        "The technical setup is weakening even more",
        "The technical picture has worsened even further",
        "The technical are weakening fast and open the door to a test to the {ma}d MA"
    ],

    "bearish_default": [
        "The technical picture remains weak",
        "The setup continues to call for caution",
        "{asset} shows no signs of a sustained rebound"
    ],

    "ma_cross_up": [
        "Successful rebound on the {ma}d MA",
        "Breakout confirmed! Despite a poor momentum, {asset} closed above {ma}-day MA",
        "{asset} crushed its {ma}d MA. A bullish setup is forming"
    ],

    "ma_cross_down": [
        "Dangerous setup for {asset} that just closed below the {ma}MA",
        "{asset} just crossed its {ma}d MA, bringing the DMAS to lower level",
        "No respite for {asset} that crossed down all the MA"
    ],

    "golden_cross": [
        "Golden cross confirms the bullish setup",
        "The golden cross reinforces the positive technical picture",
        "A golden cross has formed, signaling potential strength ahead"
    ],

    "golden_cross_weak": [
        "{asset} has been stuck in the {price_range} range for the past weeks, despite a golden cross",
        "Despite the golden cross, {asset} remains range-bound",
        "The golden cross has failed to trigger a breakout so far"
    ],

    "death_cross": [
        "A short-lived rebound accompanied by a death cross",
        "The death cross confirms the negative technical picture",
        "A death cross has formed, adding pressure to the bearish setup"
    ],

    "dramatic_surge": [
        "A surge in momentum lifts DMAS and moves {asset} closer to bullish territory",
        "A dramatic improvement in technical indicators",
        "Strong rally drives DMAS sharply higher"
    ],

    "dramatic_collapse": [
        "A dramatic picture for {asset} that crossed all its MA in a week",
        "Sharp deterioration across all technical indicators",
        "Severe technical breakdown as {asset} breaches multiple support levels"
    ],

    "extreme_momentum_high": [
        "Hot momentum drives {asset} higher",
        "Exceptional momentum score of 100 signals strong buying pressure",
        "Peak momentum reinforces the bullish trend"
    ],

    "extreme_momentum_low": [
        "Poor momentum weighs on the technical picture",
        "Momentum has completely stalled, limiting upside potential",
        "Zero momentum reading highlights the lack of buying interest"
    ],

    "price_target_reached": [
        "The {target} target has been reached. The technical picture suggests there's more to come",
        "{asset} hit the {target} objective, next resistance at {next_target}",
        "Target achieved at {target}, momentum supports further gains"
    ],

    "rhetorical_turning": [
        "Dead cat bounce or start of a new trend?",
        "Will the cross of the {ma}d MA be the start of a new trend?",
        "Is this the bottom or just another failed rally?"
    ]
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
