"""
Example: How to integrate herculis-assessment with your Streamlit app.

This shows how to automatically classify assets into assessment levels
based on DMAS scores and price structure.
"""

from pathlib import Path
import pandas as pd
import sys

# Add module to path
sys.path.insert(0, str(Path(__file__).parent))

from src.classifier import classify, classify_all
from src.config import Assessment, ASSESSMENT_LABELS, ASSESSMENT_COLORS


# ============================================================================
# EXAMPLE 1: Classify a single asset
# ============================================================================
def example_single_asset():
    """Classify SPX Index into assessment level."""
    print("=" * 60)
    print("EXAMPLE 1: Single Asset Classification (SPX Index)")
    print("=" * 60)

    # Sample data: Generate uptrending prices
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    import numpy as np
    np.random.seed(42)
    base = np.linspace(4000, 4500, 300)
    prices = pd.Series(base + np.random.normal(0, 20, 300), index=dates)

    # Current DMAS and 1-week-ago DMAS
    dmas_current = 72.5
    dmas_1w_ago = 68.0

    # Classify
    result = classify(
        ticker='SPX Index',
        prices=prices,
        dmas=dmas_current,
        dmas_1w=dmas_1w_ago
    )

    # Display results
    print(f"\nTicker: {result.ticker}")
    print(f"Assessment: {ASSESSMENT_LABELS[result.assessment]}")
    print(f"  Color: {ASSESSMENT_COLORS[result.assessment]}")
    print(f"\nBase Assessment: {ASSESSMENT_LABELS[result.base_assessment]}")
    print(f"DMAS: {result.dmas:.1f} (WoW change: {result.dmas_wow_change:+.1f})")

    print(f"\nPrice Structure:")
    print(f"  vs 50d MA:  {result.structure.price_vs_50d:+.2f}%")
    print(f"  vs 100d MA: {result.structure.price_vs_100d:+.2f}%")
    print(f"  vs 200d MA: {result.structure.price_vs_200d:+.2f}%")
    print(f"  Perfect structure: {result.structure.perfect_structure}")

    if result.adjustments:
        print(f"\nAdjustments Applied ({len(result.adjustments)}):")
        for i, adj in enumerate(result.adjustments, 1):
            print(f"  {i}. {adj}")
    else:
        print("\nNo adjustments applied")

    print(f"\nSummary: {result.description}")


# ============================================================================
# EXAMPLE 2: Classify multiple assets
# ============================================================================
def example_multiple_assets():
    """Classify multiple assets at once."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multiple Asset Classification")
    print("=" * 60)

    import numpy as np
    np.random.seed(42)

    # Generate sample data for 3 assets
    tickers = ['SPX Index', 'NKY Index', 'CSI 300 Index']

    prices_dict = {}
    dmas_dict = {}
    dmas_1w_dict = {}

    # SPX: Strong uptrend
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    prices_dict['SPX Index'] = pd.Series(
        np.linspace(4000, 4500, 300) + np.random.normal(0, 20, 300),
        index=dates
    )
    dmas_dict['SPX Index'] = 72.5
    dmas_1w_dict['SPX Index'] = 68.0

    # NKY: Downtrend
    prices_dict['NKY Index'] = pd.Series(
        np.linspace(32000, 29000, 300) + np.random.normal(0, 200, 300),
        index=dates
    )
    dmas_dict['NKY Index'] = 28.3
    dmas_1w_dict['NKY Index'] = 35.0

    # CSI: Sideways
    prices_dict['CSI 300 Index'] = pd.Series(
        np.full(300, 3800) + np.random.normal(0, 100, 300),
        index=dates
    )
    dmas_dict['CSI 300 Index'] = 48.5
    dmas_1w_dict['CSI 300 Index'] = 47.0

    # Classify all
    results_df = classify_all(tickers, prices_dict, dmas_dict, dmas_1w_dict)

    # Display results
    print("\nClassification Results:")
    print("-" * 60)
    display_cols = [
        'ticker', 'assessment', 'dmas', 'dmas_wow_change',
        'price_vs_50d', 'price_vs_100d', 'price_vs_200d',
        'adjustments_count'
    ]
    print(results_df[display_cols].to_string(index=False))

    # Export to CSV
    output_path = "assessment_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nExported to: {output_path}")


# ============================================================================
# EXAMPLE 3: Streamlit Integration
# ============================================================================
def example_streamlit_code():
    """
    Show example code for Streamlit integration.

    This is NOT executable - it's a template for your app.py
    """
    template = '''
# In your app.py:

from herculis_assessment import classify, ASSESSMENT_LABELS, ASSESSMENT_COLORS
import pandas as pd

def get_asset_assessment(ticker, prices_df, dmas_current, dmas_1w_ago):
    """
    Get automatic assessment for an asset.

    Parameters
    ----------
    ticker : str
        Asset identifier
    prices_df : pd.DataFrame
        Price data with 'Price' column (needs 200+ days)
    dmas_current : float
        Current DMAS score
    dmas_1w_ago : float
        DMAS from 1 week ago

    Returns
    -------
    ClassificationResult
        Complete assessment with reasoning
    """
    prices = prices_df['Price']

    result = classify(
        ticker=ticker,
        prices=prices,
        dmas=dmas_current,
        dmas_1w=dmas_1w_ago
    )

    return result


# Usage in Streamlit:
result = get_asset_assessment('SPX Index', spx_prices, 72.5, 68.0)

# Display assessment with color
assessment_label = ASSESSMENT_LABELS[result.assessment]
assessment_color = ASSESSMENT_COLORS[result.assessment]

st.markdown(
    f"<h2 style='color: {assessment_color};'>{assessment_label}</h2>",
    unsafe_allow_html=True
)

# Show details
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("DMAS", f"{result.dmas:.1f}", f"{result.dmas_wow_change:+.1f}")
with col2:
    st.metric("vs 50d MA", f"{result.structure.price_vs_50d:+.1f}%")
with col3:
    st.metric("vs 200d MA", f"{result.structure.price_vs_200d:+.1f}%")

# Show adjustments
if result.adjustments:
    with st.expander("Adjustment Details"):
        for adj in result.adjustments:
            st.write(f"• {adj}")

# Batch classification for dashboard
from herculis_assessment import classify_all

tickers = ['SPX Index', 'NKY Index', 'CSI 300 Index', ...]
results_df = classify_all(tickers, prices_dict, dmas_dict, dmas_1w_dict)

# Create color-coded table
def color_assessment(val):
    """Apply background color based on assessment."""
    colors = {
        'Bearish': 'background-color: #FF0000',
        'Cautious': 'background-color: #FF8C00',
        'Neutral': 'background-color: #FFD700',
        'Constructive': 'background-color: #90EE90',
        'Bullish': 'background-color: #009951',
    }
    return colors.get(val, '')

styled_df = results_df.style.applymap(
    color_assessment,
    subset=['assessment']
)

st.dataframe(styled_df)
'''

    print("\n" + "=" * 60)
    print("EXAMPLE 3: Streamlit Integration Template")
    print("=" * 60)
    print(template)


# ============================================================================
# EXAMPLE 4: Understanding the classification logic
# ============================================================================
def example_classification_logic():
    """Demonstrate how classification rules work."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Classification Logic Demonstration")
    print("=" * 60)

    import numpy as np
    np.random.seed(42)

    dates = pd.date_range('2023-01-01', periods=300, freq='D')

    print("\nScenario 1: High DMAS but broken structure")
    print("-" * 40)
    # Price trending down despite high DMAS (unusual but possible)
    prices_down = pd.Series(
        np.linspace(5000, 4000, 300) + np.random.normal(0, 20, 300),
        index=dates
    )
    result1 = classify('TEST1', prices_down, dmas=72, dmas_1w=70)
    print(f"DMAS: {result1.dmas} → Base: {result1.base_assessment.name}")
    print(f"Final: {result1.assessment.name}")
    print(f"Adjustments: {result1.adjustments}")

    print("\n\nScenario 2: Moderate DMAS with perfect structure")
    print("-" * 40)
    # Strong uptrend with moderate DMAS
    prices_up = pd.Series(
        np.linspace(4000, 4800, 300) + np.random.normal(0, 20, 300),
        index=dates
    )
    result2 = classify('TEST2', prices_up, dmas=66, dmas_1w=60)
    print(f"DMAS: {result2.dmas} → Base: {result2.base_assessment.name}")
    print(f"Final: {result2.assessment.name}")
    print(f"Perfect structure: {result2.structure.perfect_structure}")
    print(f"Adjustments: {result2.adjustments}")

    print("\n\nScenario 3: Momentum change impact")
    print("-" * 40)
    # Sideways price with improving momentum
    prices_flat = pd.Series(
        np.full(300, 4200) + np.random.normal(0, 50, 300),
        index=dates
    )
    result3 = classify('TEST3', prices_flat, dmas=60, dmas_1w=48)
    print(f"DMAS: {result3.dmas} (WoW: {result3.dmas_wow_change:+.1f})")
    print(f"Base: {result3.base_assessment.name} → Final: {result3.assessment.name}")
    print(f"Adjustments: {result3.adjustments}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Run all examples
    example_single_asset()
    example_multiple_assets()
    example_classification_logic()
    example_streamlit_code()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
