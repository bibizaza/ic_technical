"""
Claude API integration for Market Compass subtitle generation.

Uses Claude Sonnet to generate unique, contextual subtitles
based on extracted market facts.
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


def get_client():
    """Get Anthropic client."""
    if not ANTHROPIC_AVAILABLE:
        raise ImportError(
            "anthropic package not installed. "
            "Install with: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
        )
    return anthropic.Anthropic(api_key=api_key)


SYSTEM_PROMPT = """You are a financial writer for Herculis Partners' Market Compass weekly report.
Your task is to write concise, professional subtitles for technical analysis charts.

CRITICAL RULES:
1. ONE sentence only, maximum 15 words
2. Use rating-appropriate language:
   - Positive: "bullish", "strong", "excellent"
   - Constructive: "constructive", "favorable", "positive" (NEVER "bullish")
   - Neutral: "mixed", "balanced", "uncertain"
   - Cautious: "cautious", "weak", "concerning"
   - Negative: "bearish", "negative", "poor"
3. Reference specific facts provided (MA levels, divergences, etc.)
4. Match the professional, measured tone of the examples
5. Output ONLY the subtitle, nothing else"""


def build_prompt(market_facts: MarketFacts) -> str:
    """Build the user prompt for Claude."""

    facts_text = format_facts_for_prompt(market_facts)
    examples = get_examples_for_rating(market_facts.rating.value)

    examples_text = "\n".join(f"  - {ex}" for ex in examples)

    prompt = f"""Generate a subtitle for this asset's technical analysis chart.

{facts_text}

Style examples for {market_facts.rating.value} rating:
{examples_text}

Remember: ONE sentence, maximum 15 words, match the style above.
Output only the subtitle:"""

    return prompt


def generate_subtitle(
    asset_data: dict,
    client=None,
    model: str = "claude-sonnet-4-20250514"
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
        Model to use (default: claude-sonnet-4-20250514)

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
        client = get_client()

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
    model: str = "claude-sonnet-4-20250514"
) -> List[dict]:
    """
    Generate subtitles for multiple assets.

    Parameters
    ----------
    assets_data : list[dict]
        List of asset data dictionaries
    client : anthropic.Anthropic, optional
        Anthropic client (reused for all calls)
    model : str
        Model to use

    Returns
    -------
    list[dict]
        List of results with subtitle, rating, facts for each asset
    """
    if client is None:
        client = get_client()

    results = []
    total_tokens = 0

    for asset_data in assets_data:
        try:
            result = generate_subtitle(asset_data, client, model)
            result["asset_name"] = asset_data["asset_name"]
            results.append(result)
            total_tokens += result["tokens_used"]
        except Exception as e:
            results.append({
                "asset_name": asset_data.get("asset_name", "Unknown"),
                "subtitle": "Technical analysis under review",
                "rating": "Neutral",
                "facts": [],
                "error": str(e),
                "tokens_used": 0,
            })

    # Log total usage
    print(f"Total tokens used: {total_tokens}")
    print(f"Estimated cost: ${total_tokens * 0.000003 + total_tokens * 0.000015:.4f}")

    return results


# Convenience function for single asset
def quick_generate(
    asset_name: str,
    dmas: int,
    technical_score: int,
    momentum_score: int,
    price_vs_50ma_pct: float = 0,
    price_vs_100ma_pct: float = 0,
    price_vs_200ma_pct: float = 0,
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

    result = generate_subtitle(asset_data)
    return result["subtitle"]
