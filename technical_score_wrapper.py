"""
Wrapper to import herculis-technical-score module with proper path handling.

This wrapper handles the relative imports in the herculis-technical-score module
by temporarily setting up the correct package structure.
"""

import sys
from pathlib import Path


def compute_dmas_scores(prices):
    """
    Compute DMAS scores from price series.

    This function imports and calls the compute_dmas_scores function from
    herculis-technical-score module without polluting sys.path permanently.

    Parameters
    ----------
    prices : pd.Series
        Price series

    Returns
    -------
    dict
        Dictionary with keys: technical_score, momentum_score, dmas
    """
    # Save original sys.path and sys.modules state
    original_path = sys.path.copy()
    modules_to_cleanup = []

    try:
        # Add herculis-technical-score to path to enable package imports
        herculis_path = Path(__file__).parent / "herculis-technical-score"

        if not herculis_path.exists():
            raise ImportError(f"Cannot find herculis-technical-score at {herculis_path}")

        # Add to path
        sys.path.insert(0, str(herculis_path))

        # Now import using package syntax
        from src.scoring import compute_dmas_scores as _compute_dmas_scores

        # Track what we imported for cleanup
        modules_to_cleanup = [m for m in sys.modules.keys() if 'herculis' in m or (m.startswith('src.') and 'scoring' in m)]

        # Call the function
        result = _compute_dmas_scores(prices)

        return result

    finally:
        # Restore original sys.path
        sys.path = original_path

        # Clean up imported modules to prevent pollution
        for module_name in modules_to_cleanup:
            if module_name in sys.modules:
                del sys.modules[module_name]


__all__ = ['compute_dmas_scores']
