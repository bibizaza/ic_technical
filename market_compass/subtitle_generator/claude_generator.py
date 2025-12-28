"""
Claude API integration for Market Compass subtitle generation.
v4: Prompt-focused approach - let Claude identify the story.
"""

import os
from typing import Optional, List

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# =============================================================================
# API KEY CONFIGURATION
# =============================================================================
ANTHROPIC_API_KEY = None  # Or set via environment variable

# Default model
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def get_client(api_key: str = None):
    """Get Anthropic client."""
    if not ANTHROPIC_AVAILABLE:
        raise ImportError(
            "anthropic package not installed. "
            "Install with: pip install anthropic"
        )

    key = api_key or ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("No Anthropic API key found.")
    return anthropic.Anthropic(api_key=key)


SYSTEM_PROMPT = """You are the technical analyst for Herculis Partners' Market Compass weekly report. Your task is to write ONE subtitle (max 15 words) that captures the key story for each asset.

## RATING SCALE
- Bullish: DMAS ≥ 70
- Constructive: DMAS 55-69
- Neutral: DMAS 45-54
- Cautious: DMAS 30-44
- Bearish: DMAS < 30

## YOUR JOB
Identify the PRIMARY STORY from the data. What's the one thing that matters most this week?

Possible stories (pick ONE):
1. **Confirmation** - Tech and momentum aligned, trend intact
2. **Divergence** - Tech and momentum disagree (mention which leads)
3. **Change** - DMAS moved significantly WoW (upgrade/downgrade)
4. **MA Event** - Price crossing, testing, rebounding, or rejected at MA
5. **Correction** - Pullback within a trend
6. **Consolidation** - Range-bound, waiting for direction
7. **Breakdown** - Support lost, picture deteriorating
8. **Recovery** - Improving from weak levels

## CRITICAL RULES
1. ONE sentence, maximum 15 words
2. MA should ONLY be mentioned if there's ACTION (cross, test, rebound, rejection) - roughly 20-30% of subtitles
3. Use rating-appropriate language:
   - Bullish: "bullish", "strong", "positive"
   - Constructive: "constructive", "favorable" (NEVER "bullish")
   - Cautious: "cautious", "weak"
   - Bearish: "bearish", "negative"
4. Focus on what the data suggests for NEXT WEEK
5. NEVER start with the asset name
6. Output ONLY the subtitle, nothing else"""


EXAMPLES = """
## REAL EXAMPLES FROM MARKET COMPASS

### Bullish (DMAS ≥ 70, both scores high)
Data: DMAS=85, Tech=70, Mom=100, above all MAs
→ "The picture remains bullish with strong momentum"

Data: DMAS=78, Tech=65, Mom=91, at ATH
→ "New all-time highs are supported by strong technical and momentum"

Data: DMAS=82, Tech=72, Mom=92, above all MAs, continuing trend
→ "No cloud on the technical horizon"

Data: DMAS=75, Tech=62, Mom=88, DMAS stable WoW
→ "All the technical elements are in place for further gains"

### Bullish with Caution (High mom, moderate tech)
Data: DMAS=72, Tech=52, Mom=92, near resistance
→ "While momentum is strong, the technical score calls for some caution"

Data: DMAS=68, Tech=48, Mom=88, approaching 50d MA
→ "High momentum but technical weakness warrants attention near the 50d MA"

### Constructive (DMAS 55-69)
Data: DMAS=62, Tech=58, Mom=66, above 50d MA
→ "The picture remains constructive with balanced indicators"

Data: DMAS=58, Tech=65, Mom=51, correction from higher levels
→ "Despite the correction, the setup remains constructive"

Data: DMAS=60, Tech=55, Mom=65, improving from last week
→ "The technical picture is gradually improving"

### Neutral - Divergence (High tech, low mom)
Data: DMAS=52, Tech=68, Mom=36, above 50d MA
→ "Strong technical score is offset by weak momentum"

Data: DMAS=48, Tech=62, Mom=34, consolidating
→ "High technical reading but poor momentum keeps the picture neutral"

### Neutral - Divergence (Low tech, high mom)
Data: DMAS=54, Tech=38, Mom=70, below 50d MA
→ "Strong momentum fails to lift the weak technical picture"

Data: DMAS=50, Tech=42, Mom=58, approaching 50d MA from below
→ "Momentum may drive a test of the 50d MA resistance"

### Neutral - Consolidation
Data: DMAS=52, Tech=54, Mom=50, stable WoW, range-bound
→ "Ongoing consolidation with no clear directional signal"

Data: DMAS=48, Tech=50, Mom=46, at 100d MA
→ "Trading at a crossroads, waiting for a catalyst"

### Cautious (DMAS 30-44)
Data: DMAS=38, Tech=42, Mom=34, below 50d MA
→ "Weak momentum and technicals warrant a cautious stance"

Data: DMAS=42, Tech=48, Mom=36, failed to reclaim 50d MA
→ "The 50d MA remains a stubborn resistance"

Data: DMAS=35, Tech=38, Mom=32, deteriorating
→ "The technical picture continues to weaken"

### Bearish (DMAS < 30)
Data: DMAS=22, Tech=28, Mom=16, below all MAs
→ "Deeply engulfed in bearish territory with no catalyst in sight"

Data: DMAS=18, Tech=24, Mom=12, deteriorating further
→ "The bearish setup shows no signs of improvement"

Data: DMAS=25, Tech=32, Mom=18, small bounce
→ "A small bounce but still far from a trend reversal"

### MA Events (only when there's action)
Data: DMAS=58, Tech=52, Mom=64, just crossed above 50d MA
→ "The break above the 50d MA reinforces the constructive outlook"

Data: DMAS=45, Tech=48, Mom=42, rebounded on 100d MA
→ "Successful rebound on the 100d MA, but needs follow-through"

Data: DMAS=35, Tech=40, Mom=30, just broke below 200d MA
→ "The break below the 200d MA is a significant technical blow"

Data: DMAS=62, Tech=58, Mom=66, testing 50d MA from above
→ "Testing the 50d MA support, a key level to hold"

### Change Events (WoW movement)
Data: DMAS=65 (was 48), Tech=60, Mom=70, surged
→ "A surge in momentum lifts the outlook to constructive territory"

Data: DMAS=42 (was 58), Tech=45, Mom=39, dropped sharply
→ "A sharp deterioration in both technical and momentum"

Data: DMAS=70 (was 65), Tech=68, Mom=72, upgraded
→ "Continued improvement pushes the picture firmly into bullish territory"
"""


def build_prompt(asset_data: dict, previous_subtitles: List[str]) -> str:
    """Build the user prompt with raw data."""

    asset = asset_data["asset_name"]
    dmas = asset_data["dmas"]
    tech = asset_data["technical_score"]
    mom = asset_data["momentum_score"]

    dmas_prev = asset_data.get("dmas_prev_week")
    price_vs_50 = asset_data.get("price_vs_50ma_pct", 0)
    price_vs_100 = asset_data.get("price_vs_100ma_pct", 0)
    price_vs_200 = asset_data.get("price_vs_200ma_pct", 0)
    price_vs_50_prev = asset_data.get("price_vs_50ma_pct_prev")

    # Build data block
    lines = [
        f"## Generate subtitle for: {asset}",
        "",
        f"DMAS: {dmas}",
        f"Technical Score: {tech}",
        f"Momentum Score: {mom}",
    ]

    # Add WoW change if available
    if dmas_prev is not None:
        change = dmas - dmas_prev
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        lines.append(f"DMAS Change: {change:+d} ({direction} from {dmas_prev} last week)")

    # Add MA positions
    lines.append("")
    lines.append("MA Positions:")

    # 50d MA with action detection
    ma_50_action = ""
    if price_vs_50_prev is not None:
        if price_vs_50_prev < -1 and price_vs_50 > 1:
            ma_50_action = " [JUST CROSSED ABOVE]"
        elif price_vs_50_prev > 1 and price_vs_50 < -1:
            ma_50_action = " [JUST CROSSED BELOW]"
        elif abs(price_vs_50_prev) > 3 and abs(price_vs_50) <= 2:
            ma_50_action = " [TESTING]"
        elif abs(price_vs_50_prev) <= 2 and price_vs_50 > 3:
            ma_50_action = " [BOUNCED OFF]"
        elif abs(price_vs_50_prev) <= 2 and price_vs_50 < -3:
            ma_50_action = " [REJECTED]"

    lines.append(f"  vs 50d MA: {price_vs_50:+.1f}%{ma_50_action}")
    lines.append(f"  vs 100d MA: {price_vs_100:+.1f}%")
    lines.append(f"  vs 200d MA: {price_vs_200:+.1f}%")

    # Add special conditions
    if asset_data.get("at_ath") or asset_data.get("at_52w_high"):
        lines.append("  [AT 52-WEEK HIGH]")
    if asset_data.get("near_52w_low") or asset_data.get("at_52w_low"):
        lines.append("  [AT 52-WEEK LOW]")

    # Add deduplication context
    if previous_subtitles:
        lines.append("")
        lines.append("ALREADY USED (generate something DIFFERENT):")
        for s in previous_subtitles[-5:]:
            lines.append(f"  - {s}")

    lines.append("")
    lines.append("Output only the subtitle:")

    return "\n".join(lines)


def generate_subtitle(
    asset_data: dict,
    previous_subtitles: List[str] = None,
    client=None,
    model: str = DEFAULT_MODEL,
    api_key: str = None
) -> dict:
    """
    Generate subtitle using Claude API.

    Parameters
    ----------
    asset_data : dict
        Asset data with keys:
        - asset_name (str)
        - dmas (int): 0-100
        - technical_score (int): 0-100
        - momentum_score (int): 0-100
        - dmas_prev_week (int): Previous DMAS (optional)
        - price_vs_50ma_pct (float): % vs 50d MA
        - price_vs_100ma_pct (float): % vs 100d MA
        - price_vs_200ma_pct (float): % vs 200d MA
        - price_vs_50ma_pct_prev (float): Previous week (optional)

    previous_subtitles : list[str], optional
        Already generated subtitles in this batch (for deduplication)

    Returns
    -------
    dict
        Result with subtitle, rating, tokens_used
    """
    if client is None:
        client = get_client(api_key)

    if previous_subtitles is None:
        previous_subtitles = []

    # Build user prompt with raw data
    prompt = build_prompt(asset_data, previous_subtitles)

    # Call Claude API
    message = client.messages.create(
        model=model,
        max_tokens=50,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": EXAMPLES + "\n\n" + prompt}
        ]
    )

    subtitle = message.content[0].text.strip().strip('"\'')
    if subtitle and subtitle[-1] not in '.!?':
        subtitle += '.'

    # Determine rating
    dmas = asset_data["dmas"]
    if dmas >= 70:
        rating = "Bullish"
    elif dmas >= 55:
        rating = "Constructive"
    elif dmas >= 45:
        rating = "Neutral"
    elif dmas >= 30:
        rating = "Cautious"
    else:
        rating = "Bearish"

    return {
        "subtitle": subtitle,
        "rating": rating,
        "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
    }


def generate_batch(
    assets_data: List[dict],
    client=None,
    model: str = DEFAULT_MODEL,
    api_key: str = None
) -> List[dict]:
    """Generate subtitles for multiple assets with deduplication."""

    if client is None:
        client = get_client(api_key)

    results = []
    generated_subtitles = []
    total_tokens = 0

    for asset_data in assets_data:
        try:
            result = generate_subtitle(
                asset_data,
                previous_subtitles=generated_subtitles,
                client=client,
                model=model
            )
            result["asset_name"] = asset_data["asset_name"]
            results.append(result)
            generated_subtitles.append(result["subtitle"])
            total_tokens += result["tokens_used"]
        except Exception as e:
            print(f"Error generating subtitle for {asset_data.get('asset_name', 'Unknown')}: {e}")
            results.append({
                "asset_name": asset_data.get("asset_name", "Unknown"),
                "subtitle": "Technical analysis under review.",
                "rating": "Neutral",
                "error": str(e),
                "tokens_used": 0,
            })

    print(f"Total tokens used: {total_tokens}")
    # Haiku pricing: $0.80/M input, $4/M output
    estimated_cost = (total_tokens * 0.0000008) + (total_tokens * 0.000004)
    print(f"Estimated cost: ${estimated_cost:.4f}")

    return results


def is_claude_available() -> bool:
    """Check if Claude API is available and configured."""
    if not ANTHROPIC_AVAILABLE:
        return False
    key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
    return key is not None


def set_api_key(api_key: str):
    """Set the API key at runtime."""
    global ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY = api_key


# Convenience function for single asset
def quick_generate(
    asset_name: str,
    dmas: int,
    technical_score: int,
    momentum_score: int,
    price_vs_50ma_pct: float = 0,
    price_vs_100ma_pct: float = 0,
    price_vs_200ma_pct: float = 0,
    api_key: str = None,
    **kwargs
) -> str:
    """Quick subtitle generation with minimal parameters."""
    asset_data = {
        "asset_name": asset_name,
        "dmas": dmas,
        "technical_score": technical_score,
        "momentum_score": momentum_score,
        "price_vs_50ma_pct": price_vs_50ma_pct,
        "price_vs_100ma_pct": price_vs_100ma_pct,
        "price_vs_200ma_pct": price_vs_200ma_pct,
        **kwargs
    }

    result = generate_subtitle(asset_data, api_key=api_key)
    return result["subtitle"]
