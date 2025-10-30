# -*- coding: utf-8 -*-
"""
CSI-specific MARS configuration using dynamic LASSO weighting.

This module demonstrates how to compute CSI momentum scores with
dynamically learned component weights via walk-forward LASSO validation.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
from .mars_lite_scorer import generate_csi_score_history
from .lasso_weighting import perform_walk_forward_validation
from .data_loader import load_prices_for_mars


def generate_csi_lasso_weights(
    excel_path_or_df,
    training_window_years: int = 5,
    testing_window_years: int = 1,
    forward_return_days: int = 63,
) -> pd.DataFrame:
    """
    Generate time-varying LASSO weights for CSI using walk-forward validation.

    This function trains LASSO models on rolling windows to learn which
    of the 5 MARS components (pure momentum, smoothness, Sharpe, idio, ADX)
    are most predictive of future returns for CSI.

    Parameters
    ----------
    excel_path_or_df : str or pd.DataFrame
        Path to Excel file or pre-loaded price DataFrame
    training_window_years : int
        Training window length (default 5 years)
    testing_window_years : int
        Step size between folds (default 1 year)
    forward_return_days : int
        Forward return window for training target (default 63 ≈ 3 months)

    Returns
    -------
    pd.DataFrame
        Time series of learned weights indexed by fold end date
        Columns: ['pure', 'smooth', 'sharpe', 'idio', 'adx', 'alpha']

    Example
    -------
    >>> weights_history = generate_csi_lasso_weights("data.xlsx")
    >>> print(weights_history.tail())
    >>> # Use latest weights:
    >>> latest_weights = weights_history.iloc[-1][['pure', 'smooth', 'sharpe', 'idio', 'adx']].to_dict()
    >>> prices_df = load_prices_for_mars("data.xlsx")
    >>> csi_score = generate_csi_score_history(prices_df, agg_method="weighted", weights=latest_weights)
    """
    # Load prices if needed
    if isinstance(excel_path_or_df, str):
        prices_df = load_prices_for_mars(excel_path_or_df)
    else:
        prices_df = excel_path_or_df

    # We need to compute raw component history first
    # For now, this is a placeholder - in production you'd call the raw component functions
    # from mars_lite_scorer and build a historical DataFrame

    # TODO: Extract raw component calculation into reusable functions
    # For demonstration, return empty DataFrame with guidance
    print("⚠️  Raw component extraction not yet implemented.")
    print("    This requires refactoring mars_lite_scorer to expose raw component calculation.")
    print("    For now, use agg_method='top2' or 'weighted' with DEFAULT_WEIGHTS.")

    return pd.DataFrame(columns=['pure', 'smooth', 'sharpe', 'idio', 'adx', 'alpha'])


def compute_csi_score_with_lasso(
    excel_path_or_df,
    use_latest_weights: bool = True,
) -> pd.Series:
    """
    Compute CSI MARS score using dynamic LASSO weighting.

    This is a convenience function that:
    1. Generates LASSO weights via walk-forward validation
    2. Uses the latest (or average) weights to compute the final score

    Parameters
    ----------
    excel_path_or_df : str or pd.DataFrame
        Path to Excel file or pre-loaded price DataFrame
    use_latest_weights : bool
        If True, use only the most recent fold's weights.
        If False, use average weights across all folds.

    Returns
    -------
    pd.Series
        CSI MARS momentum score (0-100)

    Example
    -------
    >>> csi_score = compute_csi_score_with_lasso("data.xlsx")
    >>> print(f"Latest CSI MARS score: {csi_score.iloc[-1]:.1f}")
    """
    # Load prices
    if isinstance(excel_path_or_df, str):
        prices_df = load_prices_for_mars(excel_path_or_df)
    else:
        prices_df = excel_path_or_df

    # Generate LASSO weights
    weights_history = generate_csi_lasso_weights(excel_path_or_df)

    if weights_history.empty:
        print("⚠️  LASSO weights not available. Falling back to DEFAULT_WEIGHTS.")
        return generate_csi_score_history(prices_df, agg_method="weighted")

    # Extract weights
    if use_latest_weights:
        weights_row = weights_history.iloc[-1]
    else:
        weights_row = weights_history.mean()

    weights_dict = weights_row[['pure', 'smooth', 'sharpe', 'idio', 'adx']].to_dict()

    # Compute score with learned weights
    return generate_csi_score_history(prices_df, agg_method="weighted", weights=weights_dict)


# Recommended aggregation methods for each asset
RECOMMENDED_AGG_METHODS = {
    "SPX": "top2",      # SPX works well with top 2 components
    "CSI": "lasso",     # CSI uses dynamic LASSO (adaptive to regime changes)
}
