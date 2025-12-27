"""
Style examples for Claude API prompt.

These examples teach Claude the Market Compass voice and style.
Collected from actual Market Compass editions.
"""

STYLE_EXAMPLES = [
    # Positive examples
    "The picture remains bullish with strong momentum and technical alignment",
    "New highs are supported by strong technical and momentum",
    "All the technical elements are in place to propel {asset} higher",
    "No cloud on the technical horizon",
    "Strong technical and momentum keep DMAS elevated",

    # Constructive examples
    "The picture remains constructive despite the current correction",
    "Momentum supportive but proximity to 50d MA warrants attention",
    "Strong technicals support a constructive view despite moderate momentum",
    "High momentum offsets moderate technicals in this constructive setup",
    "The break above the 50d MA reinforces the constructive outlook",

    # Neutral examples
    "The 50d MA is the battleground between momentum and technicals",
    "Strong momentum fails to translate into technical improvement",
    "A break above the 50d MA would shift the picture to constructive",
    "Technical support at the 50d MA contrasts with weak momentum",
    "Consolidating between the 50d and 100d MA with mixed signals",

    # Cautious examples
    "Stuck below the 50d MA with weak momentum",
    "The 100d MA is the next line of defense as technicals weaken",
    "The technical picture continues to deteriorate",
    "Slight improvement but still far from a constructive setup",
    "Moderate momentum provides limited support to the weak technical setup",

    # Negative examples
    "Deeply engulfed in negative territory below all moving averages",
    "No respite as {asset} drifts further from its moving averages",
    "The 200d MA is the last major support in this negative picture",
    "A dramatic deterioration across all technical indicators",
    "Below the 200d MA, the technical picture is severely damaged",
]


RATING_SPECIFIC_EXAMPLES = {
    "Bullish": [  # Was "Positive"
        "The picture remains bullish with strong momentum",
        "All the technical elements are in place for further gains",
        "New highs supported by strong technical and momentum",
        "No cloud on the technical horizon",
        "Strong DMAS reflects aligned technical and momentum signals",
    ],
    "Constructive": [
        "The picture remains constructive despite the pullback",
        "High momentum offsets moderate technicals",
        "The constructive setup holds with balanced indicators",
        "Momentum keeps the outlook favorable despite technical caution",
        "The break above the 50d MA reinforces the constructive view",
    ],
    "Neutral": [
        "Mixed signals keep the picture neutral",
        "Strong momentum offset by weak technicals",
        "A break above the 50d MA would turn the picture constructive",
        "Consolidating at key moving average levels",
        "The neutral picture persists with diverging indicators",
    ],
    "Cautious": [
        "Weak momentum and technicals warrant caution",
        "Stuck below key moving averages",
        "The technical picture is weakening",
        "Limited upside potential with current setup",
        "The 100d MA is critical support",
    ],
    "Bearish": [  # Was "Negative"
        "Deeply bearish technical picture",
        "Below all major moving averages",
        "No short-term catalyst for recovery visible",
        "The bearish setup shows no signs of improvement",
        "Technical damage is severe",
    ],
}


def get_examples_for_rating(rating: str, n: int = 3) -> list:
    """Get style examples for a specific rating."""
    return RATING_SPECIFIC_EXAMPLES.get(rating, STYLE_EXAMPLES[:n])[:n]
