"""
Claude API integration for Market Compass subtitle generation.
v5.6: Vocabulary rotation + correction detection + dynamic year.

Features:
- Prompt caching for ~80% cost savings
- Measured verbs (continues, extends) instead of dramatic (surge, soar)
- Uncertainty language (potential, likely, suggests)
- MA direction: support (from above) vs resistance (from below)
- YTD recap subtitles for equity, commodities, crypto
- Vocabulary rotation to prevent overuse of "exceptional", "potential gains"
- WoW correction detection for significant price moves
"""

import os
import re
from datetime import datetime
from typing import Optional, List, Dict

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

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODELS = {
    "haiku_45": "claude-haiku-4-5-20251001",
}

DEFAULT_MODEL_KEY = "haiku_45"

# =============================================================================
# VOCABULARY ROTATION (v5.6)
# =============================================================================
EXCEPTIONAL_SYNONYMS = [
    "exceptional",
    "outstanding",
    "remarkable",
    "impressive",
    "excellent",
    "powerful",
    "robust"
]

GAINS_PHRASES = [
    "potential further gains",
    "likely continued advance",
    "suggesting further upside",
    "pointing to additional strength",
    "with room to extend",
    "targeting higher levels"
]


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

4. VOCABULARY - CRITICAL DISTINCTION:

   "momentum" = ONLY the Momentum Score (a number, weak/strong)
   "dynamics/trajectory/advance" = overall movement direction
   "technical" = Technical Score only
   "setup" = both scores combined

   ❌ WRONG: "Bullish momentum continues" (confusing!)
   ✅ RIGHT: "Bullish dynamics continue with strong momentum"

   The word "momentum" should always have a qualifier:
   - "strong momentum" ✓
   - "weak momentum" ✓
   - "solid momentum" ✓
   - "bullish momentum" ✗ (use "bullish dynamics" instead)

5. USE MEASURED VERBS (professional tone):
   GOOD: continues, extends, builds, strengthens, accelerates, advances, holds, confirms
   AVOID: soars, surges, thunders, explodes, rockets, ignites, unleashes

   Example: "Bullish dynamics continue" NOT "Bullish dynamics surge"

6. INCLUDE UNCERTAINTY for future direction:
   Use: "potential", "likely", "points to", "suggests"
   Example: "pointing to potential further gains"

7. MA TEST DIRECTION:
   - Above MA, testing from above = "holding above 50d MA support" (positive)
   - Below MA, testing from below = "facing 50d MA resistance" (needs breakout)

8. MA VISIBILITY (1% threshold):
   - Far above ALL MAs (>1%): DO NOT mention MAs
   - Near MA (within 1%): Mention with correct direction context
   - Below ANY MA: MUST mention

9. NO NUMBERS, NO ASSET NAME

10. UNIQUENESS - Each subtitle must have different structure

Output ONLY the subtitle. No quotes, no period, no explanation."""


EXAMPLES = """
=== BULLISH (far above MAs - don't mention) ===
"Bullish dynamics continue with exceptional setup pointing to further gains"
"Strong momentum confirms bullish trajectory with potential for continuation"
"Bullish advance extends as aligned indicators suggest further strength"

=== BULLISH (testing MA support from above) ===
"Holding above 50d MA support, bullish setup remains intact"
"Bullish dynamics steady above 50d MA with solid momentum"

=== BULLISH (testing MA resistance from below) ===
"Facing 50d MA resistance, bullish momentum builds for potential breakout"
"Approaching 50d MA test with strong momentum suggesting likely break"

=== CONSTRUCTIVE ===
"Constructive dynamics develop with improving technical and momentum"
"Constructive outlook strengthens as setup builds near support"

=== NEUTRAL ===
"Neutral stance holds with mixed signals suggesting patience"
"Mixed dynamics maintain neutral outlook pending clearer direction"

=== CAUTIOUS ===
"Cautious positioning develops as technical weakens near key levels"
"Cautious outlook persists with fragile setup facing resistance"

=== BEARISH (below MAs) ===
"Trapped below all averages, bearish dynamics likely to persist"
"Submerged beneath key MAs with weak setup suggesting further pressure"
"Buried under moving averages, bearish trajectory continues"
"Languishing below key levels with deteriorating momentum"

=== CORRECTION IN BULLISH CONTEXT ===
"Despite the correction, bullish setup remains intact with strong foundation"
"Pullback offers entry point as underlying dynamics stay bullish"
"Weekly retreat doesn't derail the bullish trajectory"

=== REBOUND IN BEARISH CONTEXT ===
"Rebound offers temporary relief but bearish pressure likely to resume"
"Short-term bounce within persistent bearish structure"
"Relief rally but cautious outlook persists amid weak setup"

=== WHAT NOT TO WRITE ===
❌ "Bullish momentum continues" → Use "Bullish dynamics continue with strong momentum"
❌ "Bullish dynamics surge..." → Use measured verbs: "continue", "extend"
❌ "Momentum soars driving gains" → Too dramatic, use "strengthens"
❌ "driving massive gains" → Use uncertainty: "pointing to potential gains"
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

### Asset Class Specific Considerations

#### Equity Indices
S&P 500 serves as the primary benchmark for US large-cap equity performance.
NASDAQ tends to show higher volatility with greater sensitivity to growth expectations.
International indices like Nikkei and DAX may diverge from US markets on local factors.
Emerging market indices carry additional currency and geopolitical risk factors.
Sector rotation within equity markets can create divergent performance patterns.
Earnings seasons bring elevated volatility requiring adjusted position management.

#### Commodities
Gold typically exhibits inverse correlation to real interest rates and USD strength.
Oil prices respond to both supply dynamics and global demand expectations.
Silver combines precious metal and industrial demand characteristics.
Copper often serves as a leading indicator for global economic activity.
Platinum and palladium pricing reflects both investment and industrial demand.
Uranium has unique supply constraints from limited mining capacity.

#### Cryptocurrencies
Bitcoin dominates crypto market cap and sets the tone for the broader sector.
Ethereum represents the leading smart contract platform with distinct use cases.
Crypto markets trade continuously, creating different volatility patterns than traditional assets.
Regulatory developments can cause sharp moves across the entire crypto space.
Institutional adoption trends increasingly influence crypto market dynamics.
Correlation with risk assets tends to increase during market stress periods.

### Additional Subtitle Patterns

#### Strong Momentum Phrases
"Exceptional momentum continues to power the advance"
"Sustained strength builds on solid technical foundation"
"Robust dynamics suggest further extension higher"
"Powerful setup points to continued outperformance"
"Outstanding momentum confirms the bullish trajectory"

#### Consolidation Phrases
"Consolidating gains while maintaining constructive bias"
"Digesting recent advance with setup still favoring upside"
"Sideways action builds base for potential continuation"
"Range-bound near highs as market awaits catalyst"
"Pause in uptrend allows technical indicators to reset"

#### Weakness Phrases
"Deteriorating technicals suggest limited near-term upside"
"Breakdown below support shifts outlook to cautious"
"Persistent weakness undermines any bullish expectations"
"Failure at resistance reinforces the bearish bias"
"Distribution pattern suggests further downside risk"

### Quality Control Guidelines
Every subtitle must pass these checks before acceptance:
- Contains exactly one rating word (Bullish/Constructive/Neutral/Cautious/Bearish)
- Maximum 12 words total including the rating
- No period at the end of the subtitle
- Forward-looking language emphasizing expected moves
- Specific to current setup, not generic market commentary
- Avoids repetition of words used in previous subtitles
- Matches the conviction level implied by the DMAS score

### Vocabulary Refinement
Prefer "dynamics" over "momentum" for overall market description
Use "setup" to describe the technical configuration
Say "pointing to" instead of "will lead to" for uncertainty
Choose "suggesting" over "indicating" for probabilistic language
Write "potential" rather than "likely" for measured expectations
Select "trajectory" instead of "path" for directional commentary
"""

# Asset universe reference for better subtitle context (~600 tokens)
ASSET_UNIVERSE = """

## Asset Universe Reference

### Equity Indices

**S&P 500 (SPX)**: US large-cap benchmark tracking 500 companies. Market-cap weighted, tech-heavy.
Global risk sentiment leader. When SPX leads, risk-on dominates global markets.

**CSI 300 (SHSZ300)**: China A-shares index covering top 300 stocks on Shanghai and Shenzhen exchanges.
Sensitive to domestic policy, property sector, and US-China relations.

**Nikkei 225 (NKY)**: Japan's premier price-weighted index. Export-sensitive, benefits from weak yen.
BOJ policy and USD/JPY are key drivers.

**TASI (SASEIDX)**: Saudi Arabia Tadawul All Share Index. Oil-linked but diversifying.
Vision 2030 reforms driving non-oil sector growth.

**Sensex (SENSEX)**: India BSE 30 blue-chips. Domestic consumption and IT services driven.
Strong demographic tailwind, relatively insulated from global cycles.

**DAX (DAX)**: Germany's 40 largest companies. Export-oriented, auto and industrial heavy.
China demand and energy costs are key sensitivities.

**SMI (SMI)**: Switzerland's 20 largest stocks. Defensive, pharma and consumer staples heavy.
Safe-haven characteristics, CHF strength sensitivity.

**IBOV (IBOV)**: Brazil Bovespa index. Commodity and banking exposure.
Real rates, commodity prices, and political risk drive performance.

**Mexbol (MEXBOL)**: Mexico IPC 35-stock index. US nearshoring beneficiary.
Peso strength and US manufacturing demand are key drivers.

### Commodities

**Gold (GCA)**: Ultimate safe haven and inflation hedge. Central bank reserve asset.
Inverse correlation to real yields and USD. Flight-to-quality beneficiary.

**Silver (SIA)**: Hybrid precious/industrial metal. Solar panel and electronics demand.
Higher beta than gold, more volatile. Green energy transition play.

**Platinum (XPT)**: Auto catalyst demand (diesel) plus hydrogen economy potential.
Supply concentrated in South Africa. Clean energy transition beneficiary.

**Palladium (XPD)**: Gasoline catalytic converter demand. Severe supply constraints.
Russia supplies 40%+. EV adoption is long-term headwind.

**Oil WTI (CL1)**: US crude benchmark, Cushing Oklahoma delivery. Global growth proxy.
OPEC+ policy, US shale production, and China demand are key drivers.

**Uranium (UXA1)**: Nuclear fuel. Supply deficit emerging as mines underinvested.
Nuclear renaissance theme, SMR technology potential catalyst.

**Copper (LP1)**: Dr. Copper - industrial bellwether. Electrification mega-theme.
EV batteries, grid infrastructure, and China construction demand.

### Cryptocurrencies

**Bitcoin (XBTUSD)**: Digital gold narrative. Institutional adoption accelerating.
Halving cycles, ETF flows, and macro liquidity drive price action.

**Ethereum (XETUSD)**: Smart contract platform. DeFi and NFT backbone.
Layer 2 scaling, staking yields, and developer activity are key metrics.

**Solana (XSOUSD)**: High-speed Layer 1 blockchain. Low fees, NFT and DeFi growth.
Network reliability concerns but strong developer momentum.

**Ripple XRP (XRPUSD)**: Cross-border payments focus. Regulatory clarity improving.
Banking partnerships and legal outcomes drive sentiment.

**Binance BNB (XBIUSD)**: Exchange token for world's largest crypto exchange.
Trading volume, regulatory pressure, and ecosystem growth are drivers.

### Cross-Asset Themes to Consider

**Risk-On Environment**: SPX leads, crypto rallies, gold lags, copper outperforms
**Risk-Off Environment**: Gold leads, bonds rally, SPX falls, crypto crashes
**Inflation Theme**: Gold and commodities outperform, bonds suffer, crypto mixed
**USD Strength**: Pressure on EM equities, gold, and commodities
**USD Weakness**: Supports gold, EM, and commodity exporters
**China Recovery**: CSI 300, copper, iron ore, and luxury goods benefit
**Energy Transition**: Uranium, copper, silver, platinum outperform oil long-term

### Correlation Awareness

High correlation pairs (move together):
- Gold ↔ Silver (0.85+)
- SPX ↔ DAX (0.80+)
- Bitcoin ↔ Ethereum (0.90+)
- Oil ↔ Energy equities (0.75+)

Negative correlation pairs (diversifiers):
- Gold ↔ USD (typically negative)
- Gold ↔ Real yields (negative)
- VIX ↔ SPX (strongly negative)

### Subtitle Differentiation by Asset Type

**Equity subtitles** should reference: technicals, momentum, MA levels, regional factors
**Commodity subtitles** should reference: supply/demand, safe-haven flows, industrial demand
**Crypto subtitles** should reference: risk appetite, adoption trends, network metrics

Always ensure subtitles for similar-scoring assets have different structures and vocabulary.
"""

# Combine examples with padding for caching (need 4096+ tokens for Haiku 4.5)
EXAMPLES = EXAMPLES + EXAMPLES_PADDING + ASSET_UNIVERSE


def build_prompt(
    asset_data: dict,
    previous_subtitles: List[str],
    historical_context: Optional[str] = None
) -> str:
    """Build prompt with correction detection, vocabulary rotation, and MA direction."""

    asset_name = asset_data["asset_name"]
    dmas = asset_data["dmas"]
    technical = asset_data["technical_score"]
    momentum = asset_data["momentum_score"]
    price_change_1w = asset_data.get("price_change_1w_pct", 0)

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

    # CORRECTED thresholds
    def score_to_quality(score):
        if score >= 86:
            return "exceptional"
        elif score >= 71:
            return "strong"
        elif score >= 51:
            return "solid"
        elif score >= 31:
            return "mixed"
        else:
            return "weak"

    tech_quality = score_to_quality(technical)
    mom_quality = score_to_quality(momentum)

    # Check alignment
    aligned = abs(technical - momentum) <= 15
    if aligned:
        setup_quality = score_to_quality((technical + momentum) // 2)
        alignment_note = f"ALIGNED → '{setup_quality} setup'"
    else:
        alignment_note = f"DIVERGENT → '{tech_quality} technical' vs '{mom_quality} momentum'"

    # Extract MA data
    price_vs_50ma = asset_data.get("price_vs_50ma_pct", 0)
    price_vs_100ma = asset_data.get("price_vs_100ma_pct", 0)
    price_vs_200ma = asset_data.get("price_vs_200ma_pct", 0)

    # Determine MA context with DIRECTION
    above_all = price_vs_50ma > 1 and price_vs_100ma > 1 and price_vs_200ma > 1
    below_all = price_vs_50ma < -1 and price_vs_100ma < -1 and price_vs_200ma < -1

    # Near 50d MA with direction
    if 0 < price_vs_50ma <= 1:
        ma_note = "ABOVE 50d MA, testing SUPPORT → 'holding above 50d MA support'"
    elif -1 <= price_vs_50ma < 0:
        ma_note = "BELOW 50d MA, testing RESISTANCE → 'facing 50d MA resistance'"
    elif above_all:
        ma_note = "FAR ABOVE all MAs → DO NOT mention MAs"
    elif below_all:
        # Rotate phrases
        bearish_count = sum(1 for s in previous_subtitles if any(
            w in s.lower() for w in ['trapped', 'submerged', 'buried', 'languishing', 'below']
        ))
        phrases = [
            "trapped below all moving averages",
            "submerged beneath key averages",
            "buried under all MAs",
            "languishing below key levels"
        ]
        phrase = phrases[bearish_count % len(phrases)]
        ma_note = f"BELOW ALL MAs → use: '{phrase}'"
    else:
        ma_note = "MAs not critical → omit"

    # CORRECTION DETECTION (v5.6)
    correction_note = ""
    if price_change_1w <= -3 and dmas >= 70:
        # Significant weekly drop but still bullish
        correction_note = f"\n⚠️ CORRECTION: Price dropped {price_change_1w:.1f}% WoW but DMAS still bullish. Consider: 'Despite the correction, setup remains bullish'"
    elif price_change_1w <= -5:
        # Big drop
        correction_note = f"\n⚠️ SIGNIFICANT DROP: {price_change_1w:.1f}% WoW. Consider mentioning the pullback."
    elif price_change_1w >= 5 and dmas < 50:
        # Big bounce in bearish context
        correction_note = f"\n⚠️ STRONG REBOUND: +{price_change_1w:.1f}% WoW in weak context. Consider: 'Rebound offers relief but setup fragile'"

    # VOCABULARY ROTATION (v5.6) - Track ALL superlatives, not just "exceptional"
    vocab_note = ""
    superlatives = ['exceptional', 'outstanding', 'remarkable', 'impressive', 'powerful', 'robust', 'compelling']

    superlative_counts = {}
    for word in superlatives:
        superlative_counts[word] = sum(1 for s in previous_subtitles if word in s.lower())

    # Find least-used superlatives (used < 2 times)
    available = [w for w in superlatives if superlative_counts.get(w, 0) < 2]

    # If some superlatives are overused, suggest fresh ones
    if len(available) < len(superlatives):
        if available:
            vocab_note += f"\n⚠️ Use fresh superlative: {', '.join(available[:3])}"
        else:
            vocab_note += "\n⚠️ All superlatives used 2x+. Vary sentence structure instead."

    # Track "potential gains" phrases
    gains_count = sum(1 for s in previous_subtitles if 'potential' in s.lower() and 'gain' in s.lower())
    if gains_count >= 2:
        alternatives = [p for p in GAINS_PHRASES if 'potential further gains' not in p]
        vocab_note += f"\n⚠️ 'Potential gains' used {gains_count}x. Use instead: '{alternatives[0]}' or '{alternatives[1]}'"

    # Uniqueness - track structures and openings
    avoid_section = ""
    if previous_subtitles:
        recent = previous_subtitles[-4:]
        # Extract first 2 words to ensure different openings
        openings = [' '.join(s.split()[:2]) for s in recent if s]
        avoid_section = f"""

ALREADY USED (use DIFFERENT structure and opening):
{chr(10).join(f'✗ "{s}"' for s in recent)}
Avoid starting with: {', '.join(openings)}"""

    # Build prompt with style rules
    prompt = f"""Asset: {asset_name}
Rating: {rating}
Technical: {tech_quality} ({technical}) | Momentum: {mom_quality} ({momentum})
Weekly price change: {price_change_1w:+.1f}%
{alignment_note}
{ma_note}
{correction_note}
{vocab_note}

STYLE RULES:
- Use "dynamics/trajectory" for overall movement, "momentum" only with qualifier (strong/weak/solid)
- Use MEASURED verbs: continues, extends, builds, strengthens, accelerates
- AVOID dramatic verbs: soars, surges, thunders, explodes
- Include UNCERTAINTY: "potential gains", "likely to continue", "points to"
- Keep it professional and measured, not sensational
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
    model: str = None,
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

    # Use default model if none specified
    if model is None:
        model = MODELS[DEFAULT_MODEL_KEY]

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
            ],
            extra_headers={
                "anthropic-beta": "prompt-caching-2024-07-31"
            }
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
    model_key: str = DEFAULT_MODEL_KEY,
    api_key: str = None,
    use_history: bool = True,
    data_as_of: str = None,
) -> List[dict]:
    """Generate subtitles for multiple assets with deduplication and caching.

    Parameters
    ----------
    model_key : str, optional
        Model key from MODELS dict. Default: haiku_45.
    data_as_of : str, optional
        Date string (YYYY-MM-DD) for history storage. If None, uses today's date.
    """
    # Convert model key to actual model string
    model = MODELS.get(model_key, MODELS[DEFAULT_MODEL_KEY])
    print(f"\n🤖 Using model: {model_key} → {model}")

    if client is None:
        client = get_client(api_key)

    # Try to get historical context if enabled
    history_tracker = None
    if use_history:
        try:
            from .history_tracker import get_tracker
            history_tracker = get_tracker()
            print(f"[Claude] History tracker loaded: {history_tracker.storage_path}")
        except ImportError as e:
            print(f"[Claude] WARNING: Could not load history tracker: {e}")
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

    # Save to history after generation (including subtitles for later export)
    print(f"[Claude] History tracker: {history_tracker}, Results count: {len(results) if results else 0}")
    if history_tracker and results:
        try:
            history_data = []
            for asset, result in zip(assets_data, results):
                subtitle = result.get("subtitle")
                print(f"[Claude] Saving {asset['asset_name']}: subtitle='{subtitle[:50] if subtitle else None}...'")
                history_data.append({
                    "asset_name": asset["asset_name"],
                    "dmas": asset["dmas"],
                    "technical_score": asset["technical_score"],
                    "momentum_score": asset["momentum_score"],
                    "price_vs_50ma_pct": asset.get("price_vs_50ma_pct", 0),
                    "price_vs_100ma_pct": asset.get("price_vs_100ma_pct", 0),
                    "price_vs_200ma_pct": asset.get("price_vs_200ma_pct", 0),
                    "rating": result.get("rating", "Neutral"),
                    "rsi": asset.get("rsi"),
                    "subtitle": subtitle,  # Store Claude-generated subtitle
                })
            history_tracker.record_batch(history_data, date=data_as_of)
            print(f"[Claude] ✅ Saved {len(history_data)} assets to history (date={data_as_of})")
        except Exception as e:
            print(f"Warning: Could not save to history: {e}")

    # Cost calculation with caching
    # Haiku 4.5: Input $0.80/1M, Output $4.00/1M
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


# =============================================================================
# YTD RECAP SUBTITLE GENERATION
# =============================================================================

# Asset class mappings
ASSET_CLASSES = {
    'equity': ['SPX', 'CSI 300', 'Nikkei', 'TASI', 'Sensex', 'DAX', 'SMI', 'IBOV', 'Mexbol'],
    'commodities': ['Gold', 'Silver', 'Platinum', 'Palladium', 'Oil', 'Copper'],
    'crypto': ['Bitcoin', 'Ethereum', 'Ripple', 'Solana', 'Binance']
}

# System prompt for recap subtitles
RECAP_SYSTEM_PROMPT = """You write short, editorial "highlights of the year" subtitles for financial charts.

Style: Narrative, slightly witty, focused on the story not the numbers.
Length: 8-15 words maximum.
Tone: Like a financial journalist's one-liner summary.

STYLE EXAMPLES:
• "Ibov and Sensex have been spared from the global selloff"
• "Despite the risk off, precious metals are still consolidating"
• "The crypto drama continues. Only Binance is now positive this year"

RULES:
- 8-15 words max
- Editorial/narrative tone, not technical
- Can be slightly witty or have personality
- Highlight the most interesting story
- No period at end unless two sentences
- Focus on divergences, leaders, laggards, or trends

Output ONLY the subtitle. No quotes, no explanation."""


def build_recap_prompt(
    asset_class: str,
    perf_data: List[Dict],
) -> str:
    """
    Build prompt for asset class YTD recap subtitle.

    Parameters
    ----------
    asset_class : str
        'equity', 'commodities', or 'crypto'
    perf_data : list of dict
        Performance data with keys: asset, ytd_pct, 1w_pct, 1m_pct
    """
    # Sort by YTD performance
    sorted_data = sorted(perf_data, key=lambda x: x.get('ytd_pct', 0), reverse=True)

    # Identify key facts
    best_ytd = sorted_data[0] if sorted_data else {}
    worst_ytd = sorted_data[-1] if sorted_data else {}

    # Count positive/negative YTD
    positive_count = sum(1 for d in sorted_data if d.get('ytd_pct', 0) > 0)
    negative_count = sum(1 for d in sorted_data if d.get('ytd_pct', 0) < 0)
    total = len(sorted_data)

    # Best/worst weekly performers
    best_week = max(sorted_data, key=lambda x: x.get('1w_pct', 0)) if sorted_data else {}
    worst_week = min(sorted_data, key=lambda x: x.get('1w_pct', 0)) if sorted_data else {}

    # Build performance summary
    perf_lines = []
    for d in sorted_data:
        ytd = d.get('ytd_pct', 0)
        week = d.get('1w_pct', 0)
        perf_lines.append(f"  {d.get('asset', 'Unknown')}: YTD {ytd:+.1f}%, 1W {week:+.1f}%")
    perf_summary = "\n".join(perf_lines)

    # Identify potential narratives
    narratives = []

    if positive_count == total and total > 0:
        narratives.append(f"ALL {total} assets positive YTD")
    elif negative_count == total and total > 0:
        narratives.append(f"ALL {total} assets negative YTD")
    elif positive_count == 1:
        only_positive = [d for d in sorted_data if d.get('ytd_pct', 0) > 0]
        if only_positive:
            narratives.append(f"Only {only_positive[0].get('asset')} is positive YTD")
    elif negative_count == 1:
        only_negative = [d for d in sorted_data if d.get('ytd_pct', 0) < 0]
        if only_negative:
            narratives.append(f"Only {only_negative[0].get('asset')} is negative YTD")

    if best_ytd.get('ytd_pct', 0) > 20:
        narratives.append(f"{best_ytd.get('asset')} leads with exceptional +{best_ytd.get('ytd_pct', 0):.0f}% YTD")

    if worst_ytd.get('ytd_pct', 0) < -10:
        narratives.append(f"{worst_ytd.get('asset')} struggles with {worst_ytd.get('ytd_pct', 0):.0f}% YTD")

    # Big weekly moves
    if abs(best_week.get('1w_pct', 0)) > 5:
        narratives.append(f"{best_week.get('asset')} jumped {best_week.get('1w_pct', 0):+.1f}% this week")
    if abs(worst_week.get('1w_pct', 0)) > 5:
        narratives.append(f"{worst_week.get('asset')} dropped {worst_week.get('1w_pct', 0):.1f}% this week")

    narratives_text = "\n".join(f"  • {n}" for n in narratives) if narratives else "  • Mixed performance across the board"

    # Class-specific tone hints
    tone_hints = {
        'equity': "Focus on regional divergences, index leadership, or market regime",
        'commodities': "Focus on precious metals vs energy, safe-haven flows, or commodity cycles",
        'crypto': "Focus on the drama, volatility, or divergence between majors and altcoins"
    }

    # Get current year dynamically (v5.6)
    current_year = datetime.now().year

    prompt = f"""Generate a short "YTD Highlights" subtitle for the {asset_class.upper()} asset class.

IMPORTANT: The current year is {current_year}. If you mention a year, use "{current_year}" not any other year.

PERFORMANCE DATA (sorted best to worst YTD):
{perf_summary}

KEY OBSERVATIONS:
{narratives_text}

POSITIVE YTD: {positive_count}/{total} | NEGATIVE YTD: {negative_count}/{total}

TONE: {tone_hints.get(asset_class, 'Editorial, narrative style')}

Generate subtitle:"""

    return prompt


def generate_recap_subtitle(
    asset_class: str,
    perf_data: List[Dict],
    client=None,
    model: str = None
) -> str:
    """
    Generate a single recap subtitle for an asset class.

    Parameters
    ----------
    asset_class : str
        'equity', 'commodities', or 'crypto'
    perf_data : list of dict
        Performance data with keys: asset, ytd_pct, 1w_pct, 1m_pct
    """
    if client is None:
        client = get_client()

    if model is None:
        model = MODELS[DEFAULT_MODEL_KEY]

    prompt = build_recap_prompt(asset_class, perf_data)

    message = client.messages.create(
        model=model,
        max_tokens=50,
        system=[{
            "type": "text",
            "text": RECAP_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }],
        messages=[{"role": "user", "content": prompt}],
        extra_headers={
            "anthropic-beta": "prompt-caching-2024-07-31"
        }
    )

    # Clean subtitle
    raw = message.content[0].text.strip()
    subtitle = raw.strip('"\'')

    # Remove any explanation
    for pattern in [r'\s*This ', r'\s*Note:', r'\n']:
        match = re.search(pattern, subtitle, re.IGNORECASE)
        if match:
            subtitle = subtitle[:match.start()].strip()

    # Remove trailing period unless two sentences
    if subtitle.count('.') <= 1:
        subtitle = subtitle.rstrip('.')

    return subtitle


def generate_all_recaps(
    perf_data: Dict[str, List[Dict]],
    client=None,
    model: str = None
) -> Dict[str, str]:
    """
    Generate all 3 recap subtitles.

    Parameters
    ----------
    perf_data : dict
        Keys: 'equity', 'commodities', 'crypto'
        Values: list of dicts with keys: asset, ytd_pct, 1w_pct, 1m_pct

    Returns
    -------
    Dict with keys: 'equity', 'commodities', 'crypto'
    Values are the generated subtitles
    """
    if client is None:
        client = get_client()

    if model is None:
        model = MODELS[DEFAULT_MODEL_KEY]

    results = {}
    print("\n📊 Generating YTD Recap Subtitles...")

    for asset_class in ['equity', 'commodities', 'crypto']:
        if asset_class in perf_data:
            subtitle = generate_recap_subtitle(
                asset_class,
                perf_data[asset_class],
                client,
                model
            )
            results[asset_class] = subtitle
            print(f"   {asset_class.upper()}: {subtitle}")
        else:
            results[asset_class] = f"No data available for {asset_class}"

    return results
