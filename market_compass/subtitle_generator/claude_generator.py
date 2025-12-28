"""
Claude API integration for Market Compass subtitle generation.
v4.4: Prompt caching - caches system prompt + examples for 83% cost savings.

Uses Anthropic's prompt caching to cache static content (~2800 tokens) and only
send asset-specific data (~200 tokens) per request.
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
Write ONE subtitle (max 15 words) answering: "What's the outlook for this asset next week?"

## RATING SCALE
- Bullish (DMAS ≥70): Strong setup, expect continuation
- Constructive (DMAS 55-69): Positive bias, favorable conditions
- Neutral (DMAS 45-54): Mixed signals, no clear direction
- Cautious (DMAS 30-44): Weak setup, downside risk
- Bearish (DMAS <30): Very weak, expect continued pressure

## STORY PRIORITY (check in this order)

1. **DMAS Level First** - What does the overall score tell us?
   - DMAS ≥70: Lead with strength
   - DMAS <30: Lead with weakness

2. **Significant Change** - Did DMAS move >10 points?
   - Yes: The change IS the story

3. **MA Action** - Is there a cross, test, or rejection happening?
   - Only mention MA if there's ACTION this week
   - If stuck below MA for weeks → that's the story (from history)

4. **Divergence** - ONLY if gap >25 AND neither tech nor mom dominates the story
   - Divergence is usually a SECONDARY detail, not the headline

## CRITICAL TERMINOLOGY RULES

### Moving Averages
- Price ABOVE MA → MA is "support" (it holds price up)
- Price BELOW MA → MA is "resistance" (it blocks price from rising)

❌ WRONG: "Support fails" when price is already 5% below MA
✅ RIGHT: "The 50d MA resistance remains a hurdle"

❌ WRONG: "Testing support at the 50d MA" when price is below it
✅ RIGHT: "Approaching the 50d MA resistance from below"

### Momentum
Momentum is a SCORE (0-100), not a price level.

❌ WRONG: "Momentum support prevents further decline"
✅ RIGHT: "Strong momentum cushions the weak technicals"

❌ WRONG: "Momentum acts as a floor"
✅ RIGHT: "High momentum score offsets technical weakness"

## STRUCTURE TEMPLATES

### For Bullish/Constructive (focus on continuation)
- "The [bullish/constructive] setup points to further gains"
- "Expect the rally to continue with [reason]"
- "All indicators support the positive outlook"

### For Neutral (focus on what needs to happen)
- "Mixed signals suggest waiting for [catalyst]"
- "The [condition] will determine next direction"
- "No clear edge until [condition resolves]"

### For Cautious/Bearish (focus on risks)
- "Weak technicals suggest further downside risk"
- "No catalyst visible for a reversal"
- "The negative setup may persist until [condition]"

### For MA Events (only when there's action)
- "The break above the 50d MA reinforces the bullish case"
- "Struggling below the 50d MA resistance limits upside"
- "A sustained move above the [X]d MA would shift outlook"

### For Divergence (only when pronounced AND central)
- "Strong momentum may eventually lift weak technicals"
- "Technical strength awaits momentum confirmation"

## RULES
1. ONE sentence, max 15 words
2. Lead with the OVERALL PICTURE, not a single detail
3. MA mentioned only if there's action OR historical stuck pattern
4. Divergence mentioned only if gap >25 and it's central to the story
5. Use correct support/resistance terminology
6. Never call momentum "support" or "floor"
7. Match tone to rating (bullish words for bullish rating, etc.)

## OUTPUT
Only the subtitle, nothing else."""


EXAMPLES = """
## EXAMPLES BY RATING

### BULLISH (DMAS ≥70)
Lead with strength, mention specifics only if remarkable.

Data: DMAS=85, Tech=70, Mom=100
→ "The bullish setup remains intact with aligned indicators"

Data: DMAS=78, Tech=65, Mom=91, above all MAs
→ "Expect the rally to extend with technicals and momentum aligned"

Data: DMAS=72, Tech=55, Mom=89 (divergence but bullish)
→ "Strong momentum drives the bullish outlook despite moderate technicals"
NOT: "Divergence between momentum and technicals may cause..."


### CONSTRUCTIVE (DMAS 55-69)
Positive but measured tone.

Data: DMAS=62, Tech=58, Mom=66
→ "The constructive setup suggests continued positive bias"

Data: DMAS=58, Tech=52, Mom=64, just broke above 50d MA
→ "The 50d MA breakout supports the improving outlook"

Data: DMAS=60, Tech=68, Mom=52 (tech leads)
→ "Solid technicals provide foundation for potential gains"
NOT: "Tech-momentum divergence creates uncertainty"


### NEUTRAL (DMAS 45-54)
Focus on what would change the picture.

Data: DMAS=50, Tech=52, Mom=48
→ "Mixed signals warrant patience until a clearer trend emerges"

Data: DMAS=48, Tech=65, Mom=31 (big divergence)
→ "Strong technicals need momentum confirmation to turn constructive"

Data: DMAS=52, Tech=40, Mom=64 (opposite divergence)
→ "Momentum strength may eventually lift the weak technical picture"


### CAUTIOUS (DMAS 30-44)
Highlight risks, what would improve.

Data: DMAS=38, Tech=42, Mom=34, below 50d MA
→ "Weak setup suggests continued pressure below the 50d MA resistance"
NOT: "Support fails as technicals collapse"

Data: DMAS=42, Tech=48, Mom=36
→ "The cautious stance persists with no catalyst for reversal"

Data: DMAS=35, Tech=38, Mom=32, well below 50d MA (-8%)
→ "Far below key moving averages with no floor visible"
NOT: "Testing support at the 50d MA" (it's 8% above, not testing)


### BEARISH (DMAS <30)
Emphasize severity, what would change.

Data: DMAS=22, Tech=28, Mom=16
→ "The bearish picture shows no signs of stabilization"

Data: DMAS=18, Tech=24, Mom=12, below all MAs
→ "Deep in bearish territory with further downside likely"

Data: DMAS=28, Tech=32, Mom=24, small bounce
→ "A modest bounce but far from reversing the negative trend"


### MA-SPECIFIC (only when there's action)

Crossing UP:
Data: DMAS=58, just broke above 50d MA
→ "The 50d MA breakout opens the door to further gains"

Crossing DOWN:
Data: DMAS=42, just broke below 50d MA
→ "Breaking below the 50d MA adds to the cautious outlook"

Testing from BELOW (approaching resistance):
Data: DMAS=48, at -1% vs 50d MA, was -5%
→ "Approaching the 50d MA resistance - a break above would shift tone"

Testing from ABOVE (approaching support):
Data: DMAS=58, at +1% vs 50d MA, was +5%
→ "Testing the 50d MA support - holding it preserves the constructive case"

STUCK below for weeks:
Data: DMAS=42, -6% vs 50d MA
History: Below 50d MA for 5 weeks
→ "Still trapped below the 50d MA after five weeks of struggle"


### WHAT NOT TO WRITE

❌ "The massive tech-momentum divergence must resolve"
→ Divergence is not the main story, DMAS level is

❌ "Momentum support prevents further decline"
→ Momentum is a score, not support

❌ "Support fails as price breaks down"
→ If already below, it's resistance not support

❌ "Divergence will resolve lower if support breaks"
→ Double error: over-emphasis on divergence + wrong terminology

❌ "Despite moderate momentum support"
→ Momentum is not support

❌ "DMAS is at 52 with technical at 55" (just restating data)
❌ "The picture is neutral" (no forward look)
❌ "Below the 50d MA at -3.2%" (data dump, no insight)
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

    # Divergence detection - ONLY when very pronounced
    divergence = mom - tech
    if abs(divergence) >= 30:  # Raised threshold - divergence is secondary
        if divergence > 0:
            lines.append(f"Note: Momentum leads technical by {divergence} points")
        else:
            lines.append(f"Note: Technical leads momentum by {-divergence} points")

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
    Generate subtitle using Claude API with prompt caching.

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
        Result with subtitle, rating, tokens info (including cache stats)
    """
    if client is None:
        client = get_client(api_key)

    if previous_subtitles is None:
        previous_subtitles = []

    # Build user prompt with raw data
    prompt = build_prompt(asset_data, previous_subtitles, historical_context)

    # Call Claude API with prompt caching
    # Cache the static system prompt + examples, only send dynamic prompt each time
    message = client.messages.create(
        model=model,
        max_tokens=50,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT + "\n\n" + EXAMPLES,
                "cache_control": {"type": "ephemeral"}  # Cache this block
            }
        ],
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    subtitle = message.content[0].text.strip().strip('"\'')

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

    # Get cache stats
    usage = message.usage
    cache_read = getattr(usage, 'cache_read_input_tokens', 0)
    cache_create = getattr(usage, 'cache_creation_input_tokens', 0)

    return {
        "subtitle": subtitle,
        "rating": rating,
        "tokens_used": usage.input_tokens + usage.output_tokens,
        "tokens": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "cache_read": cache_read,
            "cache_create": cache_create,
        }
    }


def generate_batch(
    assets_data: List[dict],
    client=None,
    model: str = DEFAULT_MODEL,
    api_key: str = None,
    use_history: bool = True
) -> List[dict]:
    """Generate subtitles for multiple assets with deduplication and caching."""

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

    # Track token usage with cache stats
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_create = 0

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

            # Accumulate token counts
            tokens = result.get("tokens", {})
            total_input += tokens.get("input", result.get("tokens_used", 0))
            total_output += tokens.get("output", 0)
            total_cache_read += tokens.get("cache_read", 0)
            total_cache_create += tokens.get("cache_create", 0)

        except Exception as e:
            print(f"Error generating subtitle for {asset_data.get('asset_name', 'Unknown')}: {e}")
            results.append({
                "asset_name": asset_data.get("asset_name", "Unknown"),
                "subtitle": "Technical analysis under review",
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

    # Cost calculation with caching
    # Haiku 4.5: Input $0.80/1M, Output $4.00/1M
    # Cache read: 90% discount = $0.08/1M
    # Cache write: 25% premium = $1.00/1M
    regular_input = total_input - total_cache_read - total_cache_create
    regular_input_cost = regular_input * 0.80 / 1_000_000
    cache_read_cost = total_cache_read * 0.08 / 1_000_000  # 90% off
    cache_write_cost = total_cache_create * 1.00 / 1_000_000  # 25% premium
    output_cost = total_output * 4.00 / 1_000_000

    total_cost = regular_input_cost + cache_read_cost + cache_write_cost + output_cost

    print(f"\n📊 Token Usage:")
    print(f"   Input: {total_input:,} (Cache read: {total_cache_read:,}, Cache create: {total_cache_create:,})")
    print(f"   Output: {total_output:,}")
    print(f"   Estimated cost: ${total_cost:.4f}")

    if total_cache_read > 0:
        # Savings = what we would have paid at full price - what we paid at cached price
        savings = total_cache_read * (0.80 - 0.08) / 1_000_000
        print(f"   💰 Cache savings: ${savings:.4f}")

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
