"""Test the Claude API subtitle generator."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_compass.subtitle_generator.fact_extractor import extract_facts, format_facts_for_prompt
from market_compass.subtitle_generator.claude_generator import generate_subtitle, generate_batch


def test_fact_extraction():
    """Test the fact extraction module."""
    print("=" * 60)
    print("TESTING FACT EXTRACTION")
    print("=" * 60)

    test_asset = {
        "asset_name": "CSI 300",
        "dmas": 52,
        "technical_score": 65,
        "momentum_score": 39,
        "dmas_prev_week": 48,
        "price_vs_50ma_pct": -3.2,
        "price_vs_100ma_pct": 1.5,
        "price_vs_200ma_pct": 5.0,
        "price_vs_50ma_pct_prev": -5.0,
        "price_vs_100ma_pct_prev": -0.5,
    }

    facts = extract_facts(test_asset)

    print(f"\nAsset: {facts.asset_name}")
    print(f"Rating: {facts.rating.value}")
    print(f"DMAS: {facts.dmas}")
    print(f"Technical: {facts.technical_score}")
    print(f"Momentum: {facts.momentum_score}")
    print(f"\nExtracted facts:")
    for fact in facts.facts:
        print(f"  - {fact}")
    print(f"\nPrimary condition: {facts.primary_condition}")

    print("\n" + "-" * 40)
    print("Formatted for prompt:")
    print("-" * 40)
    print(format_facts_for_prompt(facts))

    return facts


def test_claude_generation():
    """Test Claude API subtitle generation."""
    print("\n" + "=" * 60)
    print("TESTING CLAUDE API GENERATION")
    print("=" * 60)

    # Check if API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set.")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("Skipping Claude API test.")
        return None

    test_asset = {
        "asset_name": "CSI 300",
        "dmas": 52,
        "technical_score": 65,
        "momentum_score": 39,
        "dmas_prev_week": 48,
        "price_vs_50ma_pct": -3.2,
        "price_vs_100ma_pct": 1.5,
        "price_vs_200ma_pct": 5.0,
        "price_vs_50ma_pct_prev": -5.0,
        "price_vs_100ma_pct_prev": -0.5,
    }

    print("\nGenerating subtitle for CSI 300...")
    try:
        result = generate_subtitle(test_asset)
        print(f"\nAsset: {test_asset['asset_name']}")
        print(f"Rating: {result['rating']}")
        print(f"Facts: {result['facts']}")
        print(f"Primary condition: {result['primary_condition']}")
        print(f"\n✅ SUBTITLE: {result['subtitle']}")
        print(f"\nTokens used: {result['tokens_used']}")
        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def test_batch_generation():
    """Test batch subtitle generation."""
    print("\n" + "=" * 60)
    print("TESTING BATCH GENERATION")
    print("=" * 60)

    # Check if API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set. Skipping batch test.")
        return None

    assets = [
        {
            "asset_name": "S&P 500",
            "dmas": 75,
            "technical_score": 80,
            "momentum_score": 70,
            "price_vs_50ma_pct": 3.5,
            "price_vs_100ma_pct": 6.2,
            "price_vs_200ma_pct": 12.0,
        },
        {
            "asset_name": "DAX",
            "dmas": 62,
            "technical_score": 55,
            "momentum_score": 69,
            "price_vs_50ma_pct": 1.2,
            "price_vs_100ma_pct": 3.5,
            "price_vs_200ma_pct": 8.0,
        },
        {
            "asset_name": "Gold",
            "dmas": 35,
            "technical_score": 40,
            "momentum_score": 30,
            "price_vs_50ma_pct": -2.5,
            "price_vs_100ma_pct": -4.0,
            "price_vs_200ma_pct": 1.5,
        },
    ]

    print(f"\nGenerating subtitles for {len(assets)} assets...")
    try:
        results = generate_batch(assets)

        print("\nResults:")
        print("-" * 60)
        for r in results:
            print(f"\n{r['asset_name']} ({r['rating']}):")
            print(f"  {r['subtitle']}")
            if 'error' in r:
                print(f"  ⚠️  Error: {r['error']}")

        return results
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Always test fact extraction (no API needed)
    test_fact_extraction()

    # Test Claude generation if API key is set
    test_claude_generation()

    # Test batch generation
    test_batch_generation()
