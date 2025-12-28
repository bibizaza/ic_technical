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

# Padding content to reach 4,096 token minimum for Haiku 4.5 caching
EXAMPLES_PADDING = """

## Extended Analysis Framework

### Bullish Environment (DMAS ≥70)
When DMAS exceeds 70 with aligned technical and momentum scores, the setup strongly favors continuation.
Strong bullish setups typically show technical scores above 65 combined with momentum readings above 80.
Price action in these conditions tends to extend well above key moving averages.
The rally has room to run when both indicators confirm and reinforce the bullish bias.
New highs become probable when momentum sustains above 85 with technical confirmation.
Pullbacks in bullish environments often find support at the 50-day moving average.

### Constructive Territory (DMAS 55-69)
DMAS readings between 55-69 indicate favorable but measured market conditions.
Technical leadership over momentum in this range suggests solid underlying structural strength.
Momentum leadership within constructive territory may indicate near-term acceleration potential.
Watch for confirmation moves above key moving average levels to upgrade outlook.
The constructive zone often precedes either bullish breakouts or neutral consolidation.
Position sizing should reflect the moderate conviction level of constructive readings.

### Neutral Zone Analysis (DMAS 45-54)
DMAS readings of 45-54 require patience as directional clarity develops.
Divergences between technical and momentum scores often precede significant breakouts.
Range-bound price action in neutral territory suggests accumulation or distribution phases.
Wait for a clear catalyst before committing to a strong directional view.
The neutral zone is transitional - expect resolution toward bullish or cautious territory.
Technical and momentum convergence from neutral often signals the next trend direction.

### Cautious Stance (DMAS 30-44)
DMAS readings of 30-44 warrant a defensive positioning approach.
Below-average scores in this range suggest limited near-term upside potential.
Key support levels become critically important in cautious market conditions.
Recovery from cautious territory requires improvement in both technical and momentum.
Failed rallies are common when DMAS remains stuck in the cautious zone.
Risk management takes priority over return seeking in cautious environments.

### Bearish Conditions (DMAS <30)
DMAS readings below 30 indicate persistent and significant weakness.
Both technical and momentum scores typically confirm the negative bias at these levels.
Price action in bearish conditions usually remains below all key moving averages.
Meaningful reversal requires a significant catalyst combined with score improvement.
Counter-trend rallies in bearish territory are often short-lived and should be faded.
The bearish zone demands defensive positioning until clear improvement emerges.

### Moving Average Framework
The 50-day MA serves as the primary short-term trend indicator and first support/resistance.
The 100-day MA represents the intermediate trend benchmark for medium-term positioning.
The 200-day MA defines the long-term trend and major support/resistance zones.
Crosses between moving averages signal potential trend changes requiring attention.
Price relationship to the MA structure defines the support and resistance framework.
Golden crosses (50 above 200) and death crosses (50 below 200) mark major regime changes.

### Divergence Analysis
Technical-momentum divergence often precedes significant price moves.
When momentum leads technical by 20+ points, expect technical scores to catch up.
When technical leads momentum by 20+ points, watch for momentum confirmation or failure.
Persistent divergence without resolution suggests range-bound conditions.
Divergence resolution direction often determines the next trending move.

### Forward-Looking Considerations
Always frame analysis in terms of expected next-week price action.
Consider both the current score levels and the direction of change.
Weight recent momentum shifts more heavily than static score readings.
Account for proximity to key moving averages when assessing breakout potential.
Factor historical context when available to identify persistent patterns.

### Score Interpretation Guidelines
DMAS represents the average of technical and momentum scores, providing a balanced view.
Technical scores reflect price structure, trend strength, and moving average relationships.
Momentum scores capture rate of change, relative strength, and buying/selling pressure.
The combination of both scores offers more reliable signals than either alone.
Score changes of 5+ points week-over-week deserve attention in the subtitle.
Score changes of 10+ points represent significant shifts requiring headline treatment.

### Subtitle Writing Best Practices
Lead with the most important insight, not background context.
Use active voice and forward-looking language whenever possible.
Avoid technical jargon that requires explanation in a single line.
Match the energy of the subtitle to the conviction level of the scores.
Bullish subtitles should sound confident; bearish subtitles should sound cautious.
Neutral subtitles should emphasize uncertainty and the need for patience.

### Common Pitfalls to Avoid
Never describe momentum as a price level or support/resistance zone.
Never use past tense when the goal is forward-looking analysis.
Never simply restate the numerical scores without adding insight.
Never start the subtitle with the asset name or ticker symbol.
Never use hedging language that undermines the assessment rating.
Never focus on minor details when the overall picture is clear.

### Historical Context Integration
When an asset has been at the same rating for 4+ weeks, note the persistence.
When DMAS has changed significantly over the past month, highlight the trend.
When price has been stuck below/above an MA for weeks, that becomes the story.
Use historical context to add depth, not to replace the forward-looking view.
Balance historical perspective with actionable near-term expectations.

### Rating-Specific Language Patterns
Bullish: "expect", "should continue", "further upside", "strength persists"
Constructive: "positive bias", "favorable", "potential for gains", "improving"
Neutral: "mixed", "unclear", "patience required", "await confirmation"
Cautious: "downside risk", "vulnerable", "limited upside", "defensive"
Bearish: "weakness", "further decline", "no floor", "capitulation risk"

### Market Structure Analysis
Strong uptrends are characterized by higher highs and higher lows with expanding volume.
Downtrends show lower highs and lower lows with capitulation spikes marking potential bottoms.
Range-bound markets oscillate between well-defined support and resistance zones.
Breakouts from consolidation patterns often lead to sustained directional moves.
Failed breakouts can signal exhaustion and potential trend reversals.
Volume confirmation adds conviction to price-based technical signals.

### Momentum Dynamics
Rising momentum with stable technicals suggests potential for acceleration higher.
Falling momentum with stable technicals may indicate weakening buying interest.
Momentum divergences from price action often precede significant trend changes.
Extreme momentum readings above 90 or below 10 suggest potential mean reversion.
Momentum crossovers between short and long lookback periods signal regime changes.
Sustained momentum above 70 supports bullish continuation expectations.

### Technical Score Components
Price position relative to key moving averages drives technical score calculations.
Trend direction and strength contribute to the overall technical assessment.
Recent price action and pattern formations influence short-term technical readings.
Volume patterns and breadth indicators may affect technical score stability.
Higher highs and higher lows support elevated technical scores.
Distribution patterns and lower highs pressure technical readings lower.

### Cross-Asset Considerations
Equity indices often lead cyclical commodities in risk-on/risk-off transitions.
Currency movements can influence commodity prices and equity valuations.
Interest rate expectations affect both equity multiples and bond performance.
Correlation regimes shift during stress periods, affecting diversification benefits.
Global macro themes can override individual asset technical setups temporarily.
Relative strength across asset classes provides context for absolute readings.

### Timing and Catalysts
Economic data releases can trigger short-term volatility and trend changes.
Central bank communications often mark inflection points in market direction.
Earnings seasons bring increased volatility to equity markets.
Geopolitical events can cause sudden shifts in risk appetite across all assets.
Seasonal patterns may influence near-term expectations for specific assets.
Technical levels often coincide with fundamental catalysts for maximum impact.

### Risk Management Framework
Position sizing should reflect conviction level indicated by DMAS readings.
Stop-loss placement near key technical levels protects against adverse moves.
Profit targets based on technical resistance levels optimize risk-reward ratios.
Correlation awareness prevents unintended concentration in similar positions.
Volatility-adjusted sizing ensures consistent risk exposure across assets.
Drawdown limits preserve capital for future opportunities.
"""

# Combine examples with padding for caching (need 4096+ tokens for Haiku 4.5)
EXAMPLES = EXAMPLES + EXAMPLES_PADDING


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
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
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
