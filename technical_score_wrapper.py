"""
Wrapper to compute DMAS scores using herculis-technical-score module.

This wrapper handles the module imports properly and computes:
- Technical score using herculis-technical-score
- Momentum score from mars_score sheet in Excel
- DMAS as the average of technical and momentum scores
"""

import sys
import pandas as pd
import importlib.util
from pathlib import Path
from typing import Dict, Optional


def _load_module_from_file(module_name: str, file_path: Path):
    """Load a Python module from a file path without package structure."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_technical_score_only(prices: pd.Series, ticker: str = "Unknown") -> float:
    """
    Compute technical score from price series using herculis-technical-score.

    This bypasses the package __init__.py to avoid relative import errors.
    """
    herculis_path = Path(__file__).parent / "herculis-technical-score"

    if not herculis_path.exists():
        raise ImportError(f"Cannot find herculis-technical-score at {herculis_path}")

    # Load config module directly
    config_path = herculis_path / "config.py"
    config = _load_module_from_file("_herculis_config", config_path)

    # Load scoring module directly, injecting config into its namespace
    scoring_path = herculis_path / "src" / "scoring.py"

    # We need to make the config available for scoring.py's "from ..config import" statement
    # We'll load it and manually inject the imports
    with open(scoring_path, 'r') as f:
        scoring_code = f.read()

    # Replace ALL relative imports with direct access
    scoring_code = scoring_code.replace('from ..config import (', 'from _herculis_config import (')
    scoring_code = scoring_code.replace('from .indicators import (', 'from indicators import (')

    # Create a temporary module
    import types
    scoring_module = types.ModuleType("_herculis_scoring")
    scoring_module.__dict__['_herculis_config'] = config

    # Set up config in sys.modules
    sys.modules['_herculis_config'] = config

    # Add paths to allow indicator imports (no relative imports)
    original_path = sys.path.copy()
    sys.path.insert(0, str(herculis_path / "src"))
    sys.path.insert(0, str(herculis_path))

    try:
        # Now compile and execute the modified scoring code
        exec(scoring_code, scoring_module.__dict__)

        # Prepare price data in expected format
        if isinstance(prices, pd.Series):
            prices_df = pd.DataFrame({
                'Date': prices.index,
                'Price': prices.values,
                'High': prices.values,  # Use Close as proxy for High
                'Low': prices.values,   # Use Close as proxy for Low
            })
        else:
            prices_df = prices

        # Call compute_technical_score
        result = scoring_module.compute_technical_score(prices_df, ticker, include_components=False)
        return result['technical_score']

    finally:
        sys.path = original_path
        # Clean up sys.modules
        for key in list(sys.modules.keys()):
            if key.startswith('_herculis') or 'indicators' in key:
                sys.modules.pop(key, None)


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
        Dictionary with keys: technical_score, momentum_score, dmas
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

    return {
        'technical_score': tech_score,
        'momentum_score': mom_score,
        'dmas': dmas
    }


__all__ = ['compute_dmas_scores']
