"""
Claude API integration for Market Compass subtitle generation.
v5.4: Dynamics vocabulary + truncation fix.

Features:
- Prompt caching for ~80% cost savings
- Uniqueness checking with is_too_similar() and retry logic
- MA asymmetry with 1% threshold
- Vocabulary: "dynamics/trajectory" = movement, "momentum" = score only
- Fixed truncation to not cut off valid subtitles
"""

import os
import re
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
DEFAULT_MODEL = "claude-3-5-haiku-20241022"


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


SYSTEM_PROMPT = """You are a financial analyst writing chart subtitles for Herculis Partners' Market Compass.

ABSOLUTE RULES:

1. MAXIMUM 12 WORDS - Must fit one line

2. NO PERIOD at end

3. RATING WORD MUST APPEAR:
   - Bullish → "bullish"
   - Constructive → "constructive"
   - Neutral → "neutral" or "mixed"
   - Cautious → "cautious"
   - Bearish → "bearish"

4. CRITICAL VOCABULARY DISTINCTION:

   "momentum" = The Momentum Score (a number). Use for:
   - "strong momentum" (the score is high)
   - "weak momentum" (the score is low)
   - "momentum diverges from technical"

   "dynamics/advance/trajectory/thrust" = Overall movement. Use for:
   - "bullish dynamics extend" (the trend is up)
   - "powerful advance continues" (price is rising)
   - "bearish trajectory deepens" (trend is down)

   WRONG: "Bullish momentum drives..." (confusing!)
   RIGHT: "Bullish dynamics extend with strong momentum"

5. TERMINOLOGY:
   - "technical" = Technical Score
   - "momentum" = Momentum Score
   - "setup" = Both combined (overall picture)

6. SETUP LOGIC:
   - If tech and momentum align → use "setup"
   - If they diverge → describe each: "strong technical offset by weak momentum"

7. NO NUMBERS - Use qualitative words

8. NO ASSET NAME in subtitle

9. MA RULES (1% threshold):
   - Above ALL MAs by >1%: DO NOT mention MAs
   - Near MA (within 1%): Can mention test
   - Below ANY MA: MUST mention

10. UNIQUENESS - Different from previous subtitles

11. MAX 1 SUPERLATIVE per sentence

Output ONLY the subtitle. No quotes, no period, no explanation."""


EXAMPLES = """
=== BULLISH (far above MAs) ===
"Bullish dynamics surge with exceptional setup driving gains"
"Powerful advance extends as aligned indicators confirm strength"
"Bullish trajectory accelerates with robust momentum and solid technical"

=== BULLISH (near MA) ===
"Testing 50d MA, bullish setup holds with strong momentum"
"Bullish advance pauses at 50d MA with solid technical support"

=== CONSTRUCTIVE ===
"Constructive dynamics build with improving technical and momentum"
"Constructive outlook firms as setup strengthens near support"

=== NEUTRAL ===
"Neutral stance persists with mixed signals warranting patience"
"Mixed dynamics maintain neutral bias pending clarity"

=== CAUTIOUS ===
"Cautious trajectory emerges as technical weakens despite momentum"
"Cautious setup develops with fragile technical near key levels"

=== BEARISH (below MAs) ===
"Trapped below all averages, bearish dynamics persist"
"Submerged beneath key MAs with weak technical and fragile momentum"
"Bearish trajectory deepens, buried under moving average resistance"
"Languishing below all MAs with deteriorating setup"

=== WHAT NOT TO WRITE ===
❌ "Bullish momentum drives..." → Use "bullish dynamics" for movement
❌ "Oil's bearish trajectory" → Don't repeat asset name
❌ "exceptional with extraordinary" → Max 1 superlative
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
    """Build prompt with dynamics vocabulary and 1% MA threshold."""

    asset_name = asset_data["asset_name"]
    dmas = asset_data["dmas"]
    technical = asset_data["technical_score"]
    momentum = asset_data["momentum_score"]

    # Get rating
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

    # Convert scores to qualitative descriptors
    def score_to_quality(score):
        if score >= 86:
            return "exceptional"
        elif score >= 71:
            return "strong"
        elif score >= 56:
            return "solid"
        elif score >= 41:
            return "mixed"
        elif score >= 26:
            return "fragile"
        else:
            return "weak"

    tech_quality = score_to_quality(technical)
    mom_quality = score_to_quality(momentum)

    # Check alignment
    aligned = abs(technical - momentum) <= 15
    if aligned:
        setup_quality = score_to_quality((technical + momentum) // 2)
        alignment_note = f"Tech and momentum ALIGNED → use 'setup': '{setup_quality} setup'"
    else:
        alignment_note = f"Tech ({tech_quality}) and momentum ({mom_quality}) DIVERGE → describe each separately"

    # Extract MA data
    price_vs_50ma = asset_data.get("price_vs_50ma_pct", 0)
    price_vs_100ma = asset_data.get("price_vs_100ma_pct", 0)
    price_vs_200ma = asset_data.get("price_vs_200ma_pct", 0)

    # Determine MA context with 1% threshold
    above_all = price_vs_50ma > 1 and price_vs_100ma > 1 and price_vs_200ma > 1
    below_all = price_vs_50ma < -1 and price_vs_100ma < -1 and price_vs_200ma < -1
    near_50 = -1 <= price_vs_50ma <= 1

    # Build MA instruction with bearish phrase rotation
    if above_all:
        ma_note = "FAR ABOVE all MAs → DO NOT mention MAs"
    elif below_all:
        # Rotate phrases
        bearish_count = sum(1 for s in previous_subtitles if any(
            w in s.lower() for w in ['trapped', 'submerged', 'buried', 'languishing']
        ))
        phrases = [
            "trapped below all moving averages",
            "submerged beneath key averages",
            "buried under moving average resistance",
            "languishing below all MAs"
        ]
        phrase = phrases[bearish_count % len(phrases)]
        ma_note = f"BELOW ALL MAs → use: '{phrase}'"
    elif near_50:
        ma_note = f"NEAR 50d MA ({price_vs_50ma:+.1f}%) → mention this test"
    else:
        ma_note = "MAs not critical → do not mention"

    # Build uniqueness section
    avoid_section = ""
    if previous_subtitles:
        recent = previous_subtitles[-4:]
        avoid_section = f"""

AVOID (already used):
{chr(10).join(f'✗ "{s}"' for s in recent)}"""

    # Build prompt with vocabulary reminder
    prompt = f"""Asset: {asset_name}
Rating: {rating}
Technical: {tech_quality} | Momentum: {mom_quality}
{alignment_note}
{ma_note}

VOCABULARY REMINDER:
- "momentum" = the score (strong/weak momentum)
- "dynamics/advance/trajectory" = overall movement (bullish dynamics)
- Do NOT write "bullish momentum" - write "bullish dynamics with strong momentum"
{avoid_section}

Generate subtitle (max 12 words, no period):"""

    return prompt


def is_too_similar(new_subtitle: str, previous: List[str], threshold: float = 0.5) -> bool:
    """
    Check if new subtitle is too similar to any previous subtitle.

    Uses word overlap ratio - if more than threshold of words match, it's too similar.
    """
    if not previous:
        return False

    new_words = set(new_subtitle.lower().split())
    # Remove common words that don't indicate similarity
    stop_words = {'the', 'a', 'an', 'with', 'and', 'or', 'but', 'for', 'to', 'in', 'at', 'of', 'on'}
    new_words = new_words - stop_words

    if not new_words:
        return False

    for prev in previous:
        prev_words = set(prev.lower().split()) - stop_words
        if not prev_words:
            continue

        overlap = len(new_words & prev_words)
        ratio = overlap / min(len(new_words), len(prev_words))

        if ratio >= threshold:
            return True

        # Also check if they start with the same word (bad for variety)
        new_first = new_subtitle.split()[0].lower() if new_subtitle else ""
        prev_first = prev.split()[0].lower() if prev else ""
        if new_first == prev_first and new_first not in stop_words:
            return True

    return False


def generate_subtitle(
    asset_data: dict,
    previous_subtitles: List[str] = None,
    client=None,
    model: str = DEFAULT_MODEL,
    api_key: str = None,
    historical_context: str = None,
    max_retries: int = 2
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

    max_retries : int
        Maximum retries if subtitle is too similar to previous ones

    Returns
    -------
    dict
        Result with subtitle, rating, tokens info (including cache stats)
    """
    if client is None:
        client = get_client(api_key)

    if previous_subtitles is None:
        previous_subtitles = []

    # DEBUG: Print on first asset only
    is_first = not previous_subtitles
    if is_first:
        import anthropic as anth
        print(f"\n=== CACHING DEBUG ===")
        print(f"SDK version: {anth.__version__}")
        cached_content = SYSTEM_PROMPT + "\n\n" + EXAMPLES
        print(f"Cached content: {len(cached_content)} chars (~{len(cached_content)//4} tokens)")
        print(f"Model: {model}")

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

    # Track total tokens across retries
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_create = 0

    subtitle = None
    rejected_subtitles = []

    for attempt in range(max_retries + 1):
        # Build prompt with rejected subtitles added to previous
        all_previous = previous_subtitles + rejected_subtitles
        prompt = build_prompt(asset_data, all_previous, historical_context)

        # Call Claude API with prompt caching
        message = client.messages.create(
            model=model,
            max_tokens=50,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT + "\n\n" + EXAMPLES,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # DEBUG: Print raw usage on first asset
        if is_first and attempt == 0:
            print(f"Raw usage object: {message.usage}")
            print(f"Usage attributes: {[x for x in dir(message.usage) if not x.startswith('_')]}")
            print(f"=== END DEBUG ===\n")

        # Accumulate tokens
        usage = message.usage
        total_input += usage.input_tokens
        total_output += usage.output_tokens
        total_cache_read += getattr(usage, 'cache_read_input_tokens', 0)
        total_cache_create += getattr(usage, 'cache_creation_input_tokens', 0)

        # Get raw response
        raw_subtitle = message.content[0].text.strip()

        # First, remove any quotes around the whole thing
        subtitle = raw_subtitle.strip('"\'')

        # FIXED: Only truncate at clear explanation patterns, not mid-sentence
        # The previous regex was too aggressive and cut off valid content
        explanation_patterns = [
            r'\s*This subtitle',
            r'\s*Let me',
            r'\s*I\'ll',
            r'\s*Here\'s',
            r'\s*Note:',
            r'\s*\(this',
            r'\n',  # Newline indicates explanation started
        ]

        for pattern in explanation_patterns:
            match = re.search(pattern, subtitle, re.IGNORECASE)
            if match:
                subtitle = subtitle[:match.start()].strip()

        # Remove trailing period and quotes
        subtitle = subtitle.rstrip('.').rstrip('"\'').strip()

        # Enforce max 12 words
        words = subtitle.split()
        if len(words) > 12:
            subtitle = ' '.join(words[:12])
            # Clean trailing prepositions/conjunctions
            subtitle = re.sub(r'\s+(with|and|the|a|an|to|for|of|as)$', '', subtitle, flags=re.IGNORECASE)

        # Check uniqueness
        if not is_too_similar(subtitle, previous_subtitles):
            break  # Good, unique subtitle

        # Too similar - retry with this added to rejected
        if attempt < max_retries:
            print(f"  ↻ Retry {attempt + 1}: '{subtitle}' too similar, regenerating...")
            rejected_subtitles.append(subtitle)
        else:
            print(f"  ⚠ Max retries reached for {asset_data['asset_name']}, using last attempt")

    return {
        "subtitle": subtitle,
        "rating": rating,
        "tokens_used": total_input + total_output,
        "tokens": {
            "input": total_input,
            "output": total_output,
            "cache_read": total_cache_read,
            "cache_create": total_cache_create,
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
    # Haiku 3.5: Input $0.80/1M, Output $4.00/1M
    # Cache read: 90% discount = $0.08/1M
    # Cache write: 25% premium = $1.00/1M
    # Note: API may report cache tokens separately, so ensure non-negative
    regular_input = max(0, total_input - total_cache_read - total_cache_create)
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
