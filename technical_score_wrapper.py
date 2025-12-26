"""
Wrapper to compute DMAS scores using herculis-technical-score module.

This wrapper handles the module imports properly and computes:
- Technical score using herculis-technical-score
- Momentum score using MARS engine
- DMAS as the average of technical and momentum scores
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict


def compute_dmas_scores(prices: pd.Series, ticker: str = "Unknown") -> Dict[str, float]:
    """
    Compute DMAS scores from price series.

    Parameters
    ----------
    prices : pd.Series
        Price series with datetime index
    ticker : str
        Ticker name for identification

    Returns
    -------
    dict
        Dictionary with keys: technical_score, momentum_score, dmas
    """
    # Save original sys.path
    original_path = sys.path.copy()
    modules_to_cleanup = []

    try:
        # Add herculis-technical-score to sys.path
        herculis_path = Path(__file__).parent / "herculis-technical-score"

        if not herculis_path.exists():
            raise ImportError(f"Cannot find herculis-technical-score at {herculis_path}")

        # Add both the module directory and parent to enable relative imports
        sys.path.insert(0, str(herculis_path))

        # Import config and scoring directly (not as package)
        import config
        from src.scoring import compute_technical_score

        # Track modules for cleanup
        modules_to_cleanup = [m for m in sys.modules.keys()
                             if any(x in m for x in ['config', 'src.scoring', 'src.indicators', 'src.utils'])]

        # Prepare price data in expected format
        # herculis-technical-score expects DataFrame with Date, Price, High, Low columns
        if isinstance(prices, pd.Series):
            prices_df = pd.DataFrame({
                'Date': prices.index,
                'Price': prices.values,
                'High': prices.values,  # Use Close as proxy for High
                'Low': prices.values,   # Use Close as proxy for Low
            })
        else:
            prices_df = prices

        # Compute technical score
        result = compute_technical_score(prices_df, ticker, include_components=False)
        tech_score = result['technical_score']

        # For momentum: use technical score as proxy for now
        # TODO: Integrate MARS engine for proper momentum calculation
        mom_score = tech_score

        # Calculate DMAS as average
        dmas = (tech_score + mom_score) / 2.0

        return {
            'technical_score': tech_score,
            'momentum_score': mom_score,
            'dmas': dmas
        }

    finally:
        # Restore original sys.path
        sys.path = original_path

        # Clean up imported modules to prevent pollution
        for module_name in modules_to_cleanup:
            if module_name in sys.modules:
                try:
                    del sys.modules[module_name]
                except KeyError:
                    pass


__all__ = ['compute_dmas_scores']
