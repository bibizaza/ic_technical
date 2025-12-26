"""
Integration example for Market Compass subtitle generator.

Shows how to integrate with your existing PowerPoint generation workflow.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from subtitle_generator import SubtitleGenerator
import pandas as pd


# ============================================================================
# EXAMPLE 1: Single Asset Subtitle Generation
# ============================================================================
def example_single_asset():
    """Generate subtitle for a single asset."""
    print("=" * 60)
    print("EXAMPLE 1: Single Asset Subtitle")
    print("=" * 60)

    # Sample data - would come from your Excel/DMAS calculation
    asset_data = {
        "asset_name": "S&P 500",
        "asset_class": "equity",
        "dmas": 75,
        "technical_score": 70,
        "momentum_score": 80,
        "rating": "Bullish",
        "price_vs_50ma": "above",
        "price_vs_100ma": "above",
        "price_vs_200ma": "above",
        "dmas_prev_week": 72,
        "rating_prev_week": "Bullish",
        "ma_cross_event": None,
        "channel_color": "green",
        "near_support": False,
        "near_resistance": False,
        "at_ath": False,
        "price_target": 4800
    }

    generator = SubtitleGenerator()
    result = generator.generate(asset_data)

    print(f"\nAsset: {asset_data['asset_name']}")
    print(f"DMAS: {asset_data['dmas']}")
    print(f"Rating: {result['rating']}")
    print(f"\nSubtitle: {result['subtitle']}")
    print(f"Pattern: {result['pattern_used']}")
    print(f"Length: {len(result['subtitle'])} chars")


# ============================================================================
# EXAMPLE 2: Batch Processing for Weekly Report
# ============================================================================
def example_weekly_report():
    """Generate subtitles for full weekly report."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Weekly Report Batch Processing")
    print("=" * 60)

    # Simulated weekly data for multiple assets
    assets = [
        {
            "asset_name": "S&P 500",
            "asset_class": "equity",
            "dmas": 75,
            "technical_score": 70,
            "momentum_score": 80,
            "rating": "Bullish",
            "price_vs_50ma": "above",
            "price_vs_100ma": "above",
            "price_vs_200ma": "above",
            "dmas_prev_week": 72,
            "rating_prev_week": "Bullish",
            "ma_cross_event": None,
            "channel_color": "green",
            "near_support": False,
            "near_resistance": False,
            "at_ath": False,
            "price_target": None
        },
        {
            "asset_name": "Gold",
            "asset_class": "commodity",
            "dmas": 55,
            "technical_score": 60,
            "momentum_score": 50,
            "rating": "Neutral",
            "price_vs_50ma": "at",
            "price_vs_100ma": "above",
            "price_vs_200ma": "above",
            "dmas_prev_week": 54,
            "rating_prev_week": "Neutral",
            "ma_cross_event": None,
            "channel_color": "green",
            "near_support": True,
            "near_resistance": False,
            "at_ath": False,
            "price_target": None
        },
        {
            "asset_name": "Bitcoin",
            "asset_class": "crypto",
            "dmas": 30,
            "technical_score": 35,
            "momentum_score": 25,
            "rating": "Bearish",
            "price_vs_50ma": "below",
            "price_vs_100ma": "below",
            "price_vs_200ma": "above",
            "dmas_prev_week": 33,
            "rating_prev_week": "Bearish",
            "ma_cross_event": None,
            "channel_color": "red",
            "near_support": False,
            "near_resistance": False,
            "at_ath": False,
            "price_target": None
        },
        {
            "asset_name": "EUR/USD",
            "asset_class": "equity",
            "dmas": 65,
            "technical_score": 60,
            "momentum_score": 70,
            "rating": "Slightly Bullish",
            "price_vs_50ma": "above",
            "price_vs_100ma": "above",
            "price_vs_200ma": "above",
            "dmas_prev_week": 55,
            "rating_prev_week": "Neutral",
            "ma_cross_event": "crossed_above_50",
            "channel_color": "green",
            "near_support": False,
            "near_resistance": False,
            "at_ath": False,
            "price_target": None
        }
    ]

    generator = SubtitleGenerator()
    results = generator.generate_batch(assets)

    print("\nGenerated Subtitles:")
    print("-" * 60)
    for result in results:
        print(f"\n{result['asset_name']} ({result['rating']})")
        print(f"  → {result['subtitle']}")
        print(f"  Pattern: {result['pattern_used']}")


# ============================================================================
# EXAMPLE 3: Integration with Existing PowerPoint Workflow
# ============================================================================
def example_ppt_integration():
    """
    Example code showing how to integrate with PowerPoint generation.

    This is NOT executable - it's a template for your app.py
    """
    template = '''
# In your app.py PowerPoint generation section:

from market_compass.subtitle_generator import SubtitleGenerator

def generate_market_compass_ppt(df: pd.DataFrame, output_path: Path):
    """
    Generate Market Compass PowerPoint with automatic subtitles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
        - asset_name
        - dmas, technical_score, momentum_score
        - price_vs_50ma, price_vs_100ma, price_vs_200ma
        - dmas_prev_week, ma_cross_event
        - channel_color, near_support, near_resistance
        - at_ath, price_target
    """
    # Initialize subtitle generator
    generator = SubtitleGenerator()

    # Load PowerPoint template
    prs = Presentation("market_compass_template.pptx")

    # Process each asset
    for idx, row in df.iterrows():
        # Prepare asset data dictionary
        asset_data = {
            "asset_name": row["asset_name"],
            "asset_class": row.get("asset_class", "equity"),
            "dmas": int(row["dmas"]),
            "technical_score": int(row["technical_score"]),
            "momentum_score": int(row["momentum_score"]),
            "rating": row.get("rating", ""),
            "price_vs_50ma": row.get("price_vs_50ma", "above"),
            "price_vs_100ma": row.get("price_vs_100ma", "above"),
            "price_vs_200ma": row.get("price_vs_200ma", "above"),
            "dmas_prev_week": int(row.get("dmas_prev_week", row["dmas"])),
            "rating_prev_week": row.get("rating_prev_week", ""),
            "ma_cross_event": row.get("ma_cross_event"),
            "channel_color": row.get("channel_color", "green"),
            "near_support": bool(row.get("near_support", False)),
            "near_resistance": bool(row.get("near_resistance", False)),
            "at_ath": bool(row.get("at_ath", False)),
            "price_target": row.get("price_target")
        }

        # Generate subtitle
        result = generator.generate(asset_data, max_length=120)

        # Find slide for this asset (adjust based on your template)
        slide = prs.slides[idx + 1]  # Assuming slides start at index 1

        # Insert subtitle into placeholder
        for shape in slide.shapes:
            if shape.has_text_frame and shape.name.lower() == "subtitle":
                shape.text = result["subtitle"]
                break

        # Optional: Add rating color coding
        for shape in slide.shapes:
            if shape.has_text_frame and shape.name.lower() == "rating":
                shape.text = result["rating"]
                # Apply color based on rating
                if "Bullish" in result["rating"]:
                    shape.text_frame.paragraphs[0].runs[0].font.color.rgb = RGBColor(0, 153, 81)
                elif "Bearish" in result["rating"]:
                    shape.text_frame.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 0, 0)
                break

    # Save PowerPoint
    prs.save(output_path)
    print(f"Market Compass saved to {output_path}")

    # Optional: Export subtitles to CSV for review
    export_subtitles_csv(generator, df, output_path.parent / "subtitles.csv")


def export_subtitles_csv(generator: SubtitleGenerator, df: pd.DataFrame, csv_path: Path):
    """Export generated subtitles to CSV for review."""
    results = []

    for idx, row in df.iterrows():
        asset_data = {...}  # Same as above
        result = generator.generate(asset_data)

        results.append({
            "asset_name": row["asset_name"],
            "dmas": row["dmas"],
            "rating": result["rating"],
            "subtitle": result["subtitle"],
            "pattern_used": result["pattern_used"],
            "length": len(result["subtitle"])
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    print(f"Subtitles exported to {csv_path}")


# Usage in Streamlit:
if st.button("Generate Market Compass"):
    with st.spinner("Generating Market Compass..."):
        # Calculate DMAS and technical indicators
        df = calculate_dmas_scores(price_data)

        # Add MA cross events
        df = detect_ma_cross_events(df)

        # Add support/resistance levels
        df = detect_support_resistance(df)

        # Generate PowerPoint with automatic subtitles
        output_path = Path("outputs") / f"market_compass_{date.today()}.pptx"
        generate_market_compass_ppt(df, output_path)

        st.success(f"Market Compass generated: {output_path}")

        # Show preview of subtitles
        with st.expander("Preview Subtitles"):
            csv_path = output_path.parent / "subtitles.csv"
            if csv_path.exists():
                preview_df = pd.read_csv(csv_path)
                st.dataframe(preview_df)
'''

    print("\n" + "=" * 60)
    print("EXAMPLE 3: PowerPoint Integration Template")
    print("=" * 60)
    print(template)


# ============================================================================
# EXAMPLE 4: Multi-Week Consistency (Anti-Repetition)
# ============================================================================
def example_multi_week():
    """Demonstrate anti-repetition across multiple weeks."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Multi-Week Anti-Repetition")
    print("=" * 60)

    # Same asset data for 3 consecutive weeks
    asset_data = {
        "asset_name": "S&P 500",
        "asset_class": "equity",
        "dmas": 75,
        "technical_score": 70,
        "momentum_score": 80,
        "rating": "Bullish",
        "price_vs_50ma": "above",
        "price_vs_100ma": "above",
        "price_vs_200ma": "above",
        "dmas_prev_week": 73,
        "rating_prev_week": "Bullish",
        "ma_cross_event": None,
        "channel_color": "green",
        "near_support": False,
        "near_resistance": False,
        "at_ath": False,
        "price_target": None
    }

    # Use same generator instance across weeks
    generator = SubtitleGenerator()

    print("\nGenerating subtitles for 3 consecutive weeks:")
    print("-" * 60)

    for week in range(1, 4):
        result = generator.generate(asset_data)
        print(f"\nWeek {week}:")
        print(f"  {result['subtitle']}")
        print(f"  Pattern: {result['pattern_used']}")

    # Show usage stats
    stats = generator.get_stats()
    print(f"\nGenerator Stats:")
    print(f"  Assets tracked: {stats['assets_tracked']}")
    print(f"  Category distribution: {stats['category_distribution']}")


# ============================================================================
# EXAMPLE 5: DataFrame Integration
# ============================================================================
def example_dataframe_integration():
    """Integration with pandas DataFrame."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: DataFrame Integration")
    print("=" * 60)

    # Sample DataFrame (would come from your DMAS calculation)
    data = {
        "asset_name": ["S&P 500", "Gold", "Bitcoin"],
        "dmas": [75, 55, 30],
        "technical_score": [70, 60, 35],
        "momentum_score": [80, 50, 25],
        "price_vs_50ma": ["above", "at", "below"],
        "price_vs_100ma": ["above", "above", "below"],
        "price_vs_200ma": ["above", "above", "above"],
        "dmas_prev_week": [72, 54, 33],
        "ma_cross_event": [None, None, None],
        "channel_color": ["green", "green", "red"],
        "near_support": [False, True, False],
        "near_resistance": [False, False, False],
        "at_ath": [False, False, False],
        "price_target": [None, None, None]
    }

    df = pd.DataFrame(data)

    # Add missing fields with defaults
    df["asset_class"] = "equity"
    df["rating"] = df["dmas"].apply(lambda x: "Bullish" if x > 65 else "Neutral" if x >= 45 else "Bearish")
    df["rating_prev_week"] = df["rating"]

    # Generate subtitles
    generator = SubtitleGenerator()

    results = []
    for idx, row in df.iterrows():
        asset_data = row.to_dict()
        result = generator.generate(asset_data)
        results.append(result)

    # Add to DataFrame
    df["subtitle"] = [r["subtitle"] for r in results]
    df["pattern_used"] = [r["pattern_used"] for r in results]
    df["rating_final"] = [r["rating"] for r in results]

    print("\nGenerated Subtitles DataFrame:")
    print("-" * 60)
    print(df[["asset_name", "dmas", "rating_final", "subtitle"]].to_string(index=False))


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    example_single_asset()
    example_weekly_report()
    example_multi_week()
    example_dataframe_integration()
    example_ppt_integration()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
