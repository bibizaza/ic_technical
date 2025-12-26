"""
Example: How to integrate herculis-technical-score with your Streamlit app.

This shows how to replace BQL-based technical scores with locally computed scores.
"""

from pathlib import Path
import pandas as pd
import sys

# Add module to path
sys.path.insert(0, str(Path(__file__).parent))

from src.scoring import compute_technical_score, compute_all_scores, load_price_data_from_excel


# ============================================================================
# EXAMPLE 1: Compute score for a single ticker
# ============================================================================
def example_single_ticker():
    """Compute score for SPX Index."""
    print("=" * 60)
    print("EXAMPLE 1: Single Ticker (SPX Index)")
    print("=" * 60)

    # Path to your Excel file
    excel_path = Path("../input_ic_v2_global_ticker_V3.xlsx")

    if not excel_path.exists():
        print(f"Excel file not found: {excel_path}")
        return

    try:
        # Load price data
        print("\nLoading price data...")
        prices_df = load_price_data_from_excel(excel_path, "SPX Index")
        print(f"Loaded {len(prices_df)} days of data")

        # Compute technical score
        print("\nComputing technical score...")
        result = compute_technical_score(prices_df, "SPX Index", include_components=True)

        # Display results
        print(f"\nTicker: {result['ticker']}")
        print(f"Technical Score: {result['technical_score']:.2f} / 100")

        print("\nComponent Breakdown:")
        for component, score in result['components'].items():
            print(f"  {component:15s}: {score:.4f} ({score*100:.2f}/100)")

        print("\nRaw Indicators:")
        print(f"  RSI: {result['raw_indicators']['rsi']:.2f}")
        print(f"  MACD: {result['raw_indicators']['macd']}")
        print(f"  Stochastics: {result['raw_indicators']['stochastics']}")

    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# EXAMPLE 2: Compute scores for all tickers
# ============================================================================
def example_all_tickers():
    """Compute scores for all tickers in Excel file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: All Tickers")
    print("=" * 60)

    excel_path = Path("../input_ic_v2_global_ticker_V3.xlsx")

    if not excel_path.exists():
        print(f"Excel file not found: {excel_path}")
        return

    try:
        # Specify tickers to process
        tickers = [
            "SPX Index",
            "NKY Index",  # Nikkei
            "SHSZ300 Index",  # CSI 300
        ]

        print(f"\nComputing scores for {len(tickers)} tickers...")
        scores_df = compute_all_scores(str(excel_path), tickers=tickers)

        # Display results
        print("\nResults:")
        print(scores_df.to_string(index=False))

        # Export to CSV
        output_path = "technical_scores.csv"
        scores_df.to_csv(output_path, index=False)
        print(f"\nExported to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# EXAMPLE 3: Compare with BQL scores
# ============================================================================
def example_compare_with_bql():
    """Compare computed scores with existing BQL scores."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Compare with BQL Scores")
    print("=" * 60)

    excel_path = Path("../input_ic_v2_global_ticker_V3.xlsx")

    if not excel_path.exists():
        print(f"Excel file not found: {excel_path}")
        return

    try:
        # Load BQL scores from Excel
        bql_scores = pd.read_excel(excel_path, sheet_name="bql_formula")
        bql_scores = bql_scores[['Ticker', 'Technical_Score']].dropna()

        # Compute our scores
        tickers = bql_scores['Ticker'].tolist()
        our_scores = compute_all_scores(str(excel_path), tickers=tickers)

        # Merge and compare
        comparison = pd.merge(
            bql_scores,
            our_scores[['ticker', 'technical_score']],
            left_on='Ticker',
            right_on='ticker',
            how='inner'
        )

        comparison['difference'] = comparison['technical_score'] - comparison['Technical_Score']
        comparison['abs_diff'] = comparison['difference'].abs()

        print("\nComparison (Python vs BQL):")
        print(comparison[['Ticker', 'Technical_Score', 'technical_score', 'difference']].to_string(index=False))

        print(f"\nMean Absolute Difference: {comparison['abs_diff'].mean():.2f}")
        print(f"Max Difference: {comparison['abs_diff'].max():.2f}")
        print(f"Scores within ±5 points: {(comparison['abs_diff'] <= 5).sum()} / {len(comparison)}")

    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# EXAMPLE 4: Streamlit Integration
# ============================================================================
def example_streamlit_code():
    """
    Show example code for Streamlit integration.

    This is NOT executable - it's a template for your app.py
    """
    template = '''
# In your app.py:

from herculis_technical_score import compute_technical_score, load_price_data_from_excel

# Replace BQL score loading with:
def get_technical_score_local(excel_path, ticker):
    """Get technical score computed locally instead of from BQL."""
    try:
        # Load price data
        prices_df = load_price_data_from_excel(excel_path, ticker)

        # Compute score
        result = compute_technical_score(prices_df, ticker, include_components=False)

        return result['technical_score']

    except Exception as e:
        print(f"Error computing score for {ticker}: {e}")
        return None


# Usage in your Streamlit page:
spx_score = get_technical_score_local(excel_path, "SPX Index")
st.metric("SPX Technical Score", f"{spx_score:.1f}")

# Or for all tickers at once:
from herculis_technical_score import compute_all_scores

scores_df = compute_all_scores(excel_path)
st.dataframe(scores_df)
'''

    print("\n" + "=" * 60)
    print("EXAMPLE 4: Streamlit Integration Template")
    print("=" * 60)
    print(template)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Run all examples
    example_single_ticker()
    example_all_tickers()
    example_compare_with_bql()
    example_streamlit_code()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
