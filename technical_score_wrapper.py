"""
Wrapper to compute DMAS scores.

Computes:
- Technical score using simple moving average-based calculation
- Momentum score from mars_score sheet in Excel
- DMAS as the average of technical and momentum scores
- RSI (14-day) for momentum context
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from technical_analysis.core.indicators import compute_rsi


def compute_technical_score_only(prices: pd.Series, ticker: str = "Unknown") -> float:
    """
    Compute a simple technical score from price series.

    Uses a combination of short-term and long-term moving averages
    to generate a score between 0 and 100.

    This is a simplified version - for full technical score computation,
    the herculis-technical-score package would need to be properly installed.
    """
    if len(prices) < 200:
        # Not enough data for reliable score
        return 50.0  # Neutral

    # Get current price
    current_price = prices.iloc[-1]

    # Calculate moving averages
    sma_50 = prices.iloc[-50:].mean()
    sma_100 = prices.iloc[-100:].mean()
    sma_200 = prices.iloc[-200:].mean()

    # Calculate score based on price position relative to MAs
    score = 0.0
    total_weight = 0.0

    # Price vs 50-day MA (30% weight)
    if current_price > sma_50:
        pct_above = ((current_price - sma_50) / sma_50) * 100
        score += min(30, 15 + pct_above * 3)  # 15-30 range
    else:
        pct_below = ((sma_50 - current_price) / sma_50) * 100
        score += max(0, 15 - pct_below * 3)  # 0-15 range
    total_weight += 30

    # Price vs 100-day MA (30% weight)
    if current_price > sma_100:
        pct_above = ((current_price - sma_100) / sma_100) * 100
        score += min(30, 15 + pct_above * 3)
    else:
        pct_below = ((sma_100 - current_price) / sma_100) * 100
        score += max(0, 15 - pct_below * 3)
    total_weight += 30

    # Price vs 200-day MA (40% weight)
    if current_price > sma_200:
        pct_above = ((current_price - sma_200) / sma_200) * 100
        score += min(40, 20 + pct_above * 4)
    else:
        pct_below = ((sma_200 - current_price) / sma_200) * 100
        score += max(0, 20 - pct_below * 4)
    total_weight += 40

    # Normalize to 0-100
    final_score = (score / total_weight) * 100

    return max(0, min(100, final_score))


def read_momentum_score_from_excel(excel_path: str, ticker: str) -> Optional[float]:
    """
    Read pre-computed momentum score from mars_score sheet.

    Uses the existing _get_momentum_score_generic helper which handles
    ticker format conversion and sheet reading.

    Parameters
    ----------
    excel_path : str
        Path to Excel file
    ticker : str
        Ticker in format: "SPX Index", "GCA COMDTY", "XBTUSD CURNCY"

    Returns
    -------
    float or None
        Momentum score from mars_score sheet
    """
    try:
        # Use existing helper function from common_helpers
        import sys
        from pathlib import Path

        # Add technical_analysis to path temporarily
        orig_path = sys.path.copy()
        sys.path.insert(0, str(Path(__file__).parent))

        try:
            from technical_analysis.common_helpers import _get_momentum_score_generic
            return _get_momentum_score_generic(excel_path, ticker)
        finally:
            sys.path = orig_path

    except Exception as e:
        print(f"Warning: Could not read momentum score for {ticker}: {e}")
        return None


def compute_dmas_scores(prices: pd.Series, ticker: str = "Unknown", excel_path: Optional[str] = None) -> Dict[str, float]:
    """
    Compute DMAS scores from price series and Excel momentum data.

    Parameters
    ----------
    prices : pd.Series
        Price series with datetime index
    ticker : str
        Ticker name (e.g., "SPX Index", "GCA COMDTY")
    excel_path : str, optional
        Path to Excel file to read momentum scores from mars_score sheet

    Returns
    -------
    dict
        Dictionary with keys: technical_score, momentum_score, dmas, rsi,
        price_vs_50ma_pct, price_vs_100ma_pct, price_vs_200ma_pct
    """
    # Compute technical score
    tech_score = compute_technical_score_only(prices, ticker)

    # Read momentum score from Excel if available
    mom_score = None
    if excel_path:
        mom_score = read_momentum_score_from_excel(excel_path, ticker)

    # If momentum not available, use technical as proxy
    if mom_score is None:
        mom_score = tech_score

    # Calculate DMAS as average
    dmas = (tech_score + mom_score) / 2.0

    # Compute RSI(14) for history tracking
    rsi = None
    if len(prices) >= 14:
        rsi_series = compute_rsi(prices, period=14)
        last_rsi = rsi_series.iloc[-1]
        if pd.notna(last_rsi):
            rsi = int(round(last_rsi))

    # Compute MA values for history tracking
    price_vs_50ma_pct = 0.0
    price_vs_100ma_pct = 0.0
    price_vs_200ma_pct = 0.0

    if len(prices) >= 200:
        current_price = prices.iloc[-1]
        sma_50 = prices.iloc[-50:].mean()
        sma_100 = prices.iloc[-100:].mean()
        sma_200 = prices.iloc[-200:].mean()

        if sma_50 > 0:
            price_vs_50ma_pct = ((current_price - sma_50) / sma_50) * 100
        if sma_100 > 0:
            price_vs_100ma_pct = ((current_price - sma_100) / sma_100) * 100
        if sma_200 > 0:
            price_vs_200ma_pct = ((current_price - sma_200) / sma_200) * 100

    return {
        'technical_score': tech_score,
        'momentum_score': mom_score,
        'dmas': dmas,
        'rsi': rsi,
        'price_vs_50ma_pct': round(price_vs_50ma_pct, 2),
        'price_vs_100ma_pct': round(price_vs_100ma_pct, 2),
        'price_vs_200ma_pct': round(price_vs_200ma_pct, 2),
    }


__all__ = ['compute_dmas_scores']
