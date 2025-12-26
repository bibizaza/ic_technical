"""
Wrapper to import herculis-technical-score module with proper path handling.

This wrapper handles the relative imports in the herculis-technical-score module
by using importlib to import from a specific path without polluting sys.path.
"""

import importlib.util
from pathlib import Path


def compute_dmas_scores(prices):
    """
    Compute DMAS scores from price series.

    This function imports and calls the compute_dmas_scores function from
    herculis-technical-score module without polluting sys.path.

    Parameters
    ----------
    prices : pd.Series
        Price series

    Returns
    -------
    dict
        Dictionary with keys: technical_score, momentum_score, dmas
    """
    # Get path to scoring module
    scoring_path = Path(__file__).parent / "herculis-technical-score" / "src" / "scoring.py"

    if not scoring_path.exists():
        raise ImportError(f"Cannot find scoring.py at {scoring_path}")

    # Import the module using importlib
    spec = importlib.util.spec_from_file_location("herculis_scoring", scoring_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {scoring_path}")

    scoring_module = importlib.util.module_from_spec(spec)

    # Temporarily add parent paths for relative imports
    import sys
    original_path = sys.path.copy()
    try:
        # Add necessary paths for the module's imports
        herculis_path = Path(__file__).parent / "herculis-technical-score"
        sys.path.insert(0, str(herculis_path))

        # Execute the module
        spec.loader.exec_module(scoring_module)

        # Call the function
        result = scoring_module.compute_dmas_scores(prices)

        return result
    finally:
        # Restore original sys.path
        sys.path = original_path


__all__ = ['compute_dmas_scores']
