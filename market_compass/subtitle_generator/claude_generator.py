"""
Claude API integration for Market Compass subtitle generation.
v4.2: Forward-looking prompt - "What should investors expect next week?"
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


SYSTEM_PROMPT = """You are the technical analyst for Herculis Partners' Market Compass weekly report.

## YOUR TASK
Write ONE subtitle (max 15 words) that answers: "What should investors expect for this asset next week?"

## RATING SCALE
- Bullish (DMAS ≥70): Expect continuation or new highs
- Constructive (DMAS 55-69): Positive bias, but watch for confirmation
- Neutral (DMAS 45-54): Could go either way, wait for catalyst
- Cautious (DMAS 30-44): Downside risk, defensive stance
- Bearish (DMAS <30): Expect continued weakness

## FIRST: IDENTIFY THE STORY (do not output this)

Before writing, mentally identify which ONE story fits best:

1. **TREND CONTINUATION** - Strong aligned scores, expect more of the same
2. **DIVERGENCE** - Tech vs Mom disagree, one will prevail
3. **INFLECTION** - Near key level, could break either way
4. **RECOVERY** - Improving from weakness
5. **DETERIORATION** - Worsening from strength
6. **CONSOLIDATION** - Range-bound, waiting for catalyst
7. **BREAKOUT/BREAKDOWN** - Just crossed key level

## THEN: WRITE FORWARD-LOOKING SUBTITLE

Structure options:
- "Expect [outcome] as [reason]"
- "[Condition] suggests [expectation]"
- "The setup points to [direction] if [condition]"
- "Watch for [event] which would [implication]"

## RULES
1. ONE sentence, max 15 words
2. FORWARD-LOOKING - what's next, not just what is
3. MA mentioned ONLY if:
   - Price just crossed it (breakout/breakdown story)
   - Price is testing it (inflection story)
   - Price has been stuck at it for weeks (historical context)
4. Use rating-appropriate language:
   - Bullish: "expect gains", "rally may continue", "upside"
   - Constructive: "positive bias", "favorable setup", "potential upside"
   - Cautious: "downside risk", "defensive", "vulnerable"
   - Bearish: "expect weakness", "further downside", "no floor"
5. NEVER start with asset name
6. Output ONLY the subtitle"""


EXAMPLES = """
## EXAMPLES BY STORY TYPE

### TREND CONTINUATION
When: Strong aligned scores, clear direction, no obstacles

Data: DMAS=85, Tech=70, Mom=100, above all MAs, DMAS stable
→ "The rally has room to run with momentum and technicals aligned"

Data: DMAS=22, Tech=28, Mom=16, below all MAs, DMAS stable
→ "Expect continued weakness with no catalyst for reversal"

Data: DMAS=78, Tech=65, Mom=91, 4 weeks of gains
→ "The bullish trend should persist into the new week"


### DIVERGENCE
When: Tech and Mom disagree significantly (>20 point gap)

Data: DMAS=52, Tech=68, Mom=36, Tech leading for 3 weeks
→ "Strong technicals await momentum confirmation to turn constructive"

Data: DMAS=54, Tech=38, Mom=70, Mom leading for 2 weeks
→ "High momentum may eventually lift the weak technical picture"

Data: DMAS=48, Tech=62, Mom=34, divergence widening
→ "The tech-momentum gap must narrow for a clearer direction"


### INFLECTION (at key level)
When: Price at/near important MA, could go either way

Data: DMAS=58, Tech=55, Mom=61, testing 50d MA from above
→ "The 50d MA test will determine if the constructive bias holds"

Data: DMAS=45, Tech=48, Mom=42, stuck below 50d MA for 4 weeks
→ "A break above the stubborn 50d MA resistance would shift outlook"

Data: DMAS=62, Tech=58, Mom=66, hovering at 50d MA
→ "Watch the 50d MA - holding it keeps the constructive setup intact"


### RECOVERY
When: DMAS improving from low levels, or rebounding from correction

Data: DMAS=55 (was 42), Tech=52, Mom=58, bounced off 100d MA
→ "The rebound suggests a potential shift toward constructive territory"

Data: DMAS=48 (was 35), Tech=50, Mom=46, rising from bearish
→ "Early signs of stabilization, but confirmation needed above 50d MA"

Data: DMAS=68 (was 58), Tech=65, Mom=71, resuming after pullback
→ "The correction appears over, expect resumption of the uptrend"


### DETERIORATION
When: DMAS falling, conditions worsening

Data: DMAS=52 (was 65), Tech=55, Mom=49, slipping
→ "The picture is fading - watch for further weakness if momentum fails"

Data: DMAS=38 (was 55), Tech=42, Mom=34, sharp drop
→ "The swift deterioration may continue until support is found"

Data: DMAS=28 (was 42), Tech=32, Mom=24, accelerating down
→ "No floor in sight as both technicals and momentum collapse"


### CONSOLIDATION
When: Range-bound, low volatility, waiting for direction

Data: DMAS=50, Tech=52, Mom=48, stable for 3 weeks
→ "Sideways consolidation continues - await breakout for direction"

Data: DMAS=55, Tech=54, Mom=56, tight range
→ "Coiling in a tight range suggests a larger move is brewing"

Data: DMAS=48, Tech=50, Mom=46, neutral for 5 weeks
→ "The extended consolidation will resolve, but timing unclear"


### BREAKOUT / BREAKDOWN
When: Just crossed key level, new trend potentially starting

Data: DMAS=58, Tech=52, Mom=64, just broke above 50d MA
→ "The 50d MA breakout, if sustained, opens the door to further gains"

Data: DMAS=42, Tech=45, Mom=39, just broke below 50d MA
→ "The break below 50d MA signals risk of further downside"

Data: DMAS=65, Tech=60, Mom=70, broke above after 4 weeks below
→ "Finally clearing the 50d MA hurdle - bullish momentum may accelerate"


### WITH HISTORICAL CONTEXT

Data: DMAS=45, Tech=50, Mom=40, below 50d MA
Historical: Below 50d MA for 5 weeks; Cautious for 4 weeks
→ "Still stuck below the 50d MA - no change in cautious stance"

Data: DMAS=62, Tech=58, Mom=66
Historical: DMAS improved +15 over past month
→ "The steady improvement may push the outlook toward bullish"

Data: DMAS=38, Tech=42, Mom=34
Historical: Bearish for 6 consecutive weeks
→ "The persistent bearish trend shows no signs of reversal"


## BAD EXAMPLES (what NOT to write)

❌ "DMAS is at 52 with technical at 55" (just restating data)
❌ "The picture is neutral" (no forward look)
❌ "Below the 50d MA at -3.2%" (data dump, no insight)
❌ "Strong momentum and weak technical" (divergence but no implication)
❌ "S&P 500 remains bullish" (starts with asset name)
"""


def build_prompt(
    asset_data: dict,
    previous_subtitles: List[str],
    historical_context: Optional[str] = None
) -> str:
    """Build forward-looking prompt with context."""

    asset = asset_data["asset_name"]
    dmas = asset_data["dmas"]
    tech = asset_data["technical_score"]
    mom = asset_data["momentum_score"]

    dmas_prev = asset_data.get("dmas_prev_week")
    price_vs_50 = asset_data.get("price_vs_50ma_pct", 0)
    price_vs_100 = asset_data.get("price_vs_100ma_pct", 0)
    price_vs_200 = asset_data.get("price_vs_200ma_pct", 0)
    price_vs_50_prev = asset_data.get("price_vs_50ma_pct_prev")

    # Determine rating
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

    lines = [
        f"## Generate forward-looking subtitle for: {asset}",
        f"## Current Rating: {rating}",
        "",
        "### Current Scores",
        f"DMAS: {dmas}",
        f"Technical: {tech}",
        f"Momentum: {mom}",
    ]

    # WoW change with interpretation
    if dmas_prev is not None:
        change = dmas - dmas_prev
        if change >= 10:
            lines.append(f"DMAS Change: +{change} (significant improvement)")
        elif change >= 5:
            lines.append(f"DMAS Change: +{change} (improving)")
        elif change <= -10:
            lines.append(f"DMAS Change: {change} (significant deterioration)")
        elif change <= -5:
            lines.append(f"DMAS Change: {change} (weakening)")
        elif abs(change) <= 2:
            lines.append(f"DMAS Change: {change:+d} (stable)")
        else:
            lines.append(f"DMAS Change: {change:+d}")

    # Divergence detection
    divergence = mom - tech
    if divergence >= 25:
        lines.append(f"DIVERGENCE: Momentum leads technical by {divergence} points")
    elif divergence <= -25:
        lines.append(f"DIVERGENCE: Technical leads momentum by {-divergence} points")

    # MA positions with action flags
    lines.append("")
    lines.append("### MA Positions")

    ma_50_note = ""
    if price_vs_50_prev is not None:
        if price_vs_50_prev < -1 and price_vs_50 > 1:
            ma_50_note = " >>> JUST BROKE ABOVE"
        elif price_vs_50_prev > 1 and price_vs_50 < -1:
            ma_50_note = " >>> JUST BROKE BELOW"
        elif abs(price_vs_50) <= 2:
            ma_50_note = " >>> TESTING"

    lines.append(f"vs 50d MA: {price_vs_50:+.1f}%{ma_50_note}")

    # Only include 100d/200d if relevant
    if abs(price_vs_100) <= 3 or price_vs_50 < -2:
        lines.append(f"vs 100d MA: {price_vs_100:+.1f}%")
    if price_vs_200 < 0 or abs(price_vs_200) <= 3:
        lines.append(f"vs 200d MA: {price_vs_200:+.1f}%")

    # Special conditions
    if asset_data.get("at_ath") or asset_data.get("at_52w_high"):
        lines.append("[AT 52-WEEK HIGH]")
    if asset_data.get("near_52w_low") or asset_data.get("at_52w_low"):
        lines.append("[AT 52-WEEK LOW]")

    # Historical context
    if historical_context:
        lines.append("")
        lines.append("### Historical Context")
        lines.append(historical_context)

    # Deduplication
    if previous_subtitles:
        lines.append("")
        lines.append("### Already used (write something DIFFERENT):")
        for s in previous_subtitles[-5:]:
            lines.append(f"- {s}")

    # Final instruction
    lines.append("")
    lines.append("### Task")
    lines.append(f"Write a forward-looking subtitle for {asset}.")
    lines.append("Answer: What should investors expect next week?")
    lines.append("")
    lines.append("Output only the subtitle:")

    return "\n".join(lines)


def generate_subtitle(
    asset_data: dict,
    previous_subtitles: List[str] = None,
    client=None,
    model: str = DEFAULT_MODEL,
    api_key: str = None,
    historical_context: str = None
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

    historical_context : str, optional
        Context from history tracker (e.g., "Below 50d MA for 4 weeks")

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
    prompt = build_prompt(asset_data, previous_subtitles, historical_context)

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
    api_key: str = None,
    use_history: bool = True
) -> List[dict]:
    """Generate subtitles for multiple assets with deduplication."""

    if client is None:
        client = get_client(api_key)

    # Try to get historical context if enabled
    history_tracker = None
    if use_history:
        try:
            from .history_tracker import get_tracker
            history_tracker = get_tracker()
        except ImportError:
            pass

    results = []
    generated_subtitles = []
    total_tokens = 0

    for asset_data in assets_data:
        try:
            # Get historical context if available
            historical_context = None
            if history_tracker:
                historical_context = history_tracker.get_context_for_subtitle(
                    asset_data["asset_name"]
                )

            result = generate_subtitle(
                asset_data,
                previous_subtitles=generated_subtitles,
                client=client,
                model=model,
                historical_context=historical_context
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

    # Save to history after generation
    if history_tracker and results:
        try:
            history_data = []
            for asset, result in zip(assets_data, results):
                history_data.append({
                    "asset_name": asset["asset_name"],
                    "dmas": asset["dmas"],
                    "technical_score": asset["technical_score"],
                    "momentum_score": asset["momentum_score"],
                    "price_vs_50ma_pct": asset.get("price_vs_50ma_pct", 0),
                    "price_vs_100ma_pct": asset.get("price_vs_100ma_pct", 0),
                    "price_vs_200ma_pct": asset.get("price_vs_200ma_pct", 0),
                    "rating": result.get("rating", "Neutral"),
                })
            history_tracker.record_batch(history_data)
            print(f"Saved {len(history_data)} assets to history")
        except Exception as e:
            print(f"Warning: Could not save to history: {e}")

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
