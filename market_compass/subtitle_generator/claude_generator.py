"""
Claude API integration for Market Compass subtitle generation.

Uses Claude Haiku to generate unique, contextual subtitles
based on extracted market facts.

v3.1: Added batch deduplication, Haiku model, prompt caching
"""

import os
from typing import Optional, List

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

from .fact_extractor import extract_facts, format_facts_for_prompt, MarketFacts
from .style_examples import get_examples_for_rating

# =============================================================================
# API KEY CONFIGURATION
# =============================================================================
# Set your Anthropic API key here for use without environment variables.
# Get your API key from: https://console.anthropic.com/
# Cost: ~$0.001 per full Market Compass report (21 assets) with Haiku + caching
ANTHROPIC_API_KEY = None  # Replace None with your API key: "sk-ant-..."

# Default model - Haiku is 10x cheaper and sufficient for this task
DEFAULT_MODEL = "claude-3-5-haiku-20241022"


def get_client(api_key: str = None):
    """
    Get Anthropic client.

    Parameters
    ----------
    api_key : str, optional
        API key to use. If not provided, uses:
        1. The hardcoded ANTHROPIC_API_KEY constant above
        2. The ANTHROPIC_API_KEY environment variable

    Returns
    -------
    anthropic.Anthropic
        Configured Anthropic client
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError(
            "anthropic package not installed. "
            "Install with: pip install anthropic"
        )

    # Priority: parameter > hardcoded constant > environment variable
    key = api_key or ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")

    if not key:
        raise ValueError(
            "No Anthropic API key found. Either:\n"
            "1. Set ANTHROPIC_API_KEY in claude_generator.py, or\n"
            "2. Set ANTHROPIC_API_KEY environment variable"
        )
    return anthropic.Anthropic(api_key=key)


SYSTEM_PROMPT = """You are a financial writer for Herculis Partners' Market Compass weekly report.
Your task is to write concise, professional subtitles for technical analysis charts.

CRITICAL RULES:
1. ONE sentence only, maximum 15 words
2. Use rating-appropriate language:
   - Bullish: "bullish", "strong", "excellent", "positive" (DMAS ≥70)
   - Constructive: "constructive", "favorable", "encouraging" (DMAS 55-69) - NEVER use "bullish"
   - Neutral: "mixed", "balanced", "uncertain", "consolidating" (DMAS 45-54)
   - Cautious: "cautious", "weak", "concerning", "challenging" (DMAS 30-44)
   - Bearish: "bearish", "negative", "poor", "weak" (DMAS <30)
3. ALWAYS mention the most relevant MA level (50d, 100d, or 200d) in relation to price
4. Reference specific facts provided (MA levels, divergences, etc.)
5. Match the professional, measured tone of the examples
6. Output ONLY the subtitle, nothing else"""


def build_prompt(market_facts: MarketFacts) -> str:
    """Build the user prompt for Claude."""

    facts_text = format_facts_for_prompt(market_facts)
    examples = get_examples_for_rating(market_facts.rating.value)

    examples_text = "\n".join(f"  - {ex}" for ex in examples)

    prompt = f"""Generate a subtitle for this asset's technical analysis chart.

{facts_text}

Style examples for {market_facts.rating.value} rating:
{examples_text}

Remember: ONE sentence, maximum 15 words, MUST mention MA position.
Output only the subtitle:"""

    return prompt


def build_prompt_with_context(market_facts: MarketFacts, previous_subtitles: List[str]) -> str:
    """Build prompt including previously generated subtitles to avoid."""

    facts_text = format_facts_for_prompt(market_facts)
    examples = get_examples_for_rating(market_facts.rating.value)
    examples_text = "\n".join(f"  - {ex}" for ex in examples)

    # Add deduplication context if we have previous subtitles
    dedup_text = ""
    if previous_subtitles:
        recent = previous_subtitles[-5:]  # Last 5 to keep prompt short
        dedup_text = "\n\nALREADY USED IN THIS REPORT (generate something DIFFERENT):\n"
        dedup_text += "\n".join(f"  - {s}" for s in recent)

    prompt = f"""Generate a subtitle for this asset's technical analysis chart.

{facts_text}

Style examples for {market_facts.rating.value} rating:
{examples_text}
{dedup_text}

Remember: ONE sentence, maximum 15 words, MUST mention MA position, MUST be unique.
Output only the subtitle:"""

    return prompt


def generate_subtitle_with_context(
    asset_data: dict,
    previous_subtitles: List[str],
    client,
    model: str = DEFAULT_MODEL
) -> dict:
    """Generate subtitle with awareness of previously generated ones."""

    market_facts = extract_facts(asset_data)
    prompt = build_prompt_with_context(market_facts, previous_subtitles)

    message = client.messages.create(
        model=model,
        max_tokens=50,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    subtitle = message.content[0].text.strip().strip('"\'')
    if subtitle and subtitle[-1] not in '.!?':
        subtitle += '.'

    return {
        "subtitle": subtitle,
        "rating": market_facts.rating.value,
        "facts": market_facts.facts,
        "primary_condition": market_facts.primary_condition,
        "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
    }


def generate_subtitle(
    asset_data: dict,
    client=None,
    model: str = DEFAULT_MODEL,
    api_key: str = None
) -> dict:
    """
    Generate subtitle using Claude API.

    Parameters
    ----------
    asset_data : dict
        Asset data dictionary (see fact_extractor.extract_facts)
    client : anthropic.Anthropic, optional
        Anthropic client (creates new one if not provided)
    model : str
        Model to use (default: claude-haiku-4-20250514)
    api_key : str, optional
        API key (uses hardcoded or env var if not provided)

    Returns
    -------
    dict
        Result with keys:
        - subtitle (str): Generated subtitle
        - rating (str): Asset rating
        - facts (list): Extracted facts
        - tokens_used (int): Total tokens used
    """
    if client is None:
        client = get_client(api_key)

    # Extract facts
    market_facts = extract_facts(asset_data)

    # Build prompt
    prompt = build_prompt(market_facts)

    # Call Claude API
    message = client.messages.create(
        model=model,
        max_tokens=50,  # Subtitle is short
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    subtitle = message.content[0].text.strip()

    # Clean up any quotes or extra punctuation
    subtitle = subtitle.strip('"\'')

    # Ensure ends with period if not already punctuated
    if subtitle and subtitle[-1] not in '.!?':
        subtitle += '.'

    return {
        "subtitle": subtitle,
        "rating": market_facts.rating.value,
        "facts": market_facts.facts,
        "primary_condition": market_facts.primary_condition,
        "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
    }


def generate_batch(
    assets_data: List[dict],
    client=None,
    model: str = DEFAULT_MODEL,
    api_key: str = None
) -> List[dict]:
    """
    Generate subtitles for multiple assets with deduplication.

    Parameters
    ----------
    assets_data : list[dict]
        List of asset data dictionaries
    client : anthropic.Anthropic, optional
        Anthropic client (reused for all calls)
    model : str
        Model to use (default: claude-haiku-4-20250514)
    api_key : str, optional
        API key (uses hardcoded or env var if not provided)

    Returns
    -------
    list[dict]
        List of results with subtitle, rating, facts for each asset
    """
    if client is None:
        client = get_client(api_key)

    results = []
    generated_subtitles = []  # Track what's been generated for deduplication
    total_tokens = 0

    for asset_data in assets_data:
        try:
            # Pass previously generated subtitles to avoid repetition
            result = generate_subtitle_with_context(
                asset_data,
                generated_subtitles,
                client,
                model
            )
            result["asset_name"] = asset_data["asset_name"]
            results.append(result)
            generated_subtitles.append(result["subtitle"])  # Track it
            total_tokens += result["tokens_used"]
        except Exception as e:
            print(f"Error generating subtitle for {asset_data.get('asset_name', 'Unknown')}: {e}")
            results.append({
                "asset_name": asset_data.get("asset_name", "Unknown"),
                "subtitle": "Technical analysis under review.",
                "rating": "Neutral",
                "facts": [],
                "error": str(e),
                "tokens_used": 0,
            })

    # Log total usage (Haiku pricing: $0.25/M input, $1.25/M output)
    print(f"Total tokens used: {total_tokens}")
    print(f"Estimated cost: ${total_tokens * 0.00000025 + total_tokens * 0.00000125:.4f}")

    return results


def is_claude_available() -> bool:
    """
    Check if Claude API is available and configured.

    Returns
    -------
    bool
        True if anthropic package is installed and API key is set
    """
    if not ANTHROPIC_AVAILABLE:
        return False

    # Check for API key (hardcoded or environment)
    key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
    return key is not None


def set_api_key(api_key: str):
    """
    Set the API key at runtime.

    Parameters
    ----------
    api_key : str
        Your Anthropic API key (starts with "sk-ant-...")
    """
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
    """
    Quick subtitle generation with minimal parameters.

    Returns just the subtitle string.
    """
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
