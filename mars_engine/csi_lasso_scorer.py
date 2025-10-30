# -*- coding: utf-8 -*-
"""
CSI-specific MARS momentum scoring with dynamic LASSO weighting.

This module provides end-to-end CSI scoring using learned component weights
from walk-forward LASSO validation.
"""

from __future__ import annotations
from typing import Optional, Dict
import pandas as pd
import numpy as np
import streamlit as st

from .raw_components import compute_raw_components
from .lasso_weighting import perform_walk_forward_validation
from .mars_lite_scorer import (
    _winsorize,
    _rolling_percentile_of_last,
    _safe_clip_0_100,
    _ema,
    _normalize_weights,
    calculate_relative_score,
    PEER_GROUP_CSI,
    DEFAULT_WEIGHTS,
)


@st.cache_data(show_spinner=False)
def _train_csi_lasso_weights_cached(
    excel_path: str,
    training_window_years: int = 5,
    testing_window_years: int = 1,
) -> pd.DataFrame:
    """
    Train LASSO weights for CSI using walk-forward validation (cached).

    Parameters
    ----------
    excel_path : str
        Path to Excel file (must be string for Streamlit caching)
    training_window_years : int
        LASSO training window length
    testing_window_years : int
        Step size between folds

    Returns
    -------
    pd.DataFrame
        Time series of learned weights indexed by fold end date
        Columns: ['pure', 'smooth', 'sharpe', 'idio', 'adx', 'alpha']
    """
    from .data_loader import load_prices_for_mars

    # Load prices
    prices_df = load_prices_for_mars(excel_path)

    # Extract CSI columns
    if "CSI" not in prices_df.columns:
        print("Warning: CSI column not found in data")
        return pd.DataFrame()

    # Ensure high/low columns exist
    if "CSI_high" not in prices_df.columns:
        prices_df["CSI_high"] = prices_df["CSI"] * 1.01
    if "CSI_low" not in prices_df.columns:
        prices_df["CSI_low"] = prices_df["CSI"] * 0.99

    # Select benchmark (prioritize MXWO, then fallback)
    bench_candidates = ["MXWO Index", "MXWO", "SPX Index", "SPX", "CSI"]
    bench_col = next((c for c in bench_candidates if c in prices_df.columns), "CSI")

    # Compute raw components for full history
    raw_components = compute_raw_components(
        df=prices_df,
        target_col="CSI",
        hi_col="CSI_high",
        lo_col="CSI_low",
        bench_col=bench_col,
    )

    # Perform walk-forward validation
    weights_history = perform_walk_forward_validation(
        prices_df=prices_df,
        raw_components_history=raw_components,
        target_col="CSI",
        training_window_years=training_window_years,
        testing_window_years=testing_window_years,
        forward_return_days=63,
    )

    return weights_history


def compute_csi_score_with_lasso(
    prices_df: pd.DataFrame,
    lasso_weights_df: pd.DataFrame,
    use_latest_weights: bool = True,
) -> pd.Series:
    """
    Compute CSI MARS score using learned LASSO weights.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Full price DataFrame with CSI columns
    lasso_weights_df : pd.DataFrame
        Learned weights from walk-forward validation
        Columns: ['pure', 'smooth', 'sharpe', 'idio', 'adx', 'alpha']
    use_latest_weights : bool
        If True, use only most recent fold's weights
        If False, use average weights across all folds

    Returns
    -------
    pd.Series
        CSI MARS momentum score (0-100), last ~2 years
    """
    if lasso_weights_df.empty:
        # Fallback to default weights
        print("⚠️  LASSO weights not available. Using DEFAULT_WEIGHTS.")
        from .mars_lite_scorer import generate_csi_score_history
        return generate_csi_score_history(prices_df, agg_method="weighted", weights=DEFAULT_WEIGHTS)

    # Extract weights
    if use_latest_weights:
        weights_row = lasso_weights_df.iloc[-1]
    else:
        weights_row = lasso_weights_df.mean()

    learned_weights = {
        "pure": float(weights_row.get("pure", 0.0)),
        "smooth": float(weights_row.get("smooth", 0.0)),
        "sharpe": float(weights_row.get("sharpe", 0.0)),
        "idio": float(weights_row.get("idio", 0.0)),
        "adx": float(weights_row.get("adx", 0.0)),
    }

    # Normalize weights
    learned_weights = _normalize_weights(learned_weights)

    print(f"📊 Using CSI LASSO weights: {learned_weights}")

    # Now compute the score using these weights
    return _compute_csi_score_with_weights(prices_df, learned_weights)


def _simple_percentile_rank(value: float, historical_series: pd.Series) -> float:
    """
    Compute global percentile rank: where does value rank in the historical series?
    This matches the percentile calculation used in LASSO training.

    Returns value between 0-100.
    """
    hist = historical_series.dropna()
    if hist.empty or pd.isna(value):
        return np.nan
    rank = (hist < value).sum() / len(hist)
    return rank * 100.0


def _compute_csi_score_with_weights(
    df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Internal function to compute CSI score with given LASSO weights.

    CRITICAL: Uses simple global percentile ranking (not rolling window percentile)
    to match the feature transformation used in LASSO training.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df = df.sort_index().copy()

    # Ensure CSI columns exist
    target_col = "CSI"
    hi_col = "CSI_high"
    lo_col = "CSI_low"

    if target_col not in df.columns:
        return pd.Series(dtype=float)

    # Ensure hi/lo exist
    if hi_col not in df.columns:
        df[hi_col] = df[target_col] * 1.01
    if lo_col not in df.columns:
        df[lo_col] = df[target_col] * 0.99

    # Build peer universe
    peers_avail = [c for c in PEER_GROUP_CSI if c in df.columns]

    # Select benchmark
    bench_candidates = ["MXWO Index", "MXWO", "CSI", "SPX Index", "SPX"]
    bench_col = next((c for c in bench_candidates if c in df.columns), target_col)

    # Compute raw components for full history
    raw_components = compute_raw_components(
        df=df,
        target_col=target_col,
        hi_col=hi_col,
        lo_col=lo_col,
        bench_col=bench_col,
    )

    # Convert to percentiles using GLOBAL RANK (same as LASSO training)
    # For each date, rank the raw component value against its full history
    abs_percentiles = pd.DataFrame(index=raw_components.index)

    # Limit to last 5 years for percentile calculation (standard practice)
    lookback_days = 252 * 5
    start_date = raw_components.index.max() - pd.Timedelta(days=lookback_days * 2)  # Allow buffer
    historical_window = raw_components.loc[raw_components.index >= start_date]

    for col in historical_window.columns:
        # For each date, compute: where does this value rank in full history?
        percentile_series = pd.Series(index=historical_window.index, dtype=float)

        # Get full historical series for this component (for ranking)
        full_historical = historical_window[col].dropna()

        # Winsorize the historical series at 2-98% (like reference code)
        hist_lo = full_historical.quantile(0.02)
        hist_hi = full_historical.quantile(0.98)
        winsorized_historical = full_historical.clip(lower=hist_lo, upper=hist_hi)

        for date in historical_window.index:
            raw_value = historical_window.loc[date, col]
            # Rank this value against winsorized history up to this date
            hist_up_to_date = winsorized_historical.loc[:date]
            percentile_series.loc[date] = _simple_percentile_rank(raw_value, hist_up_to_date)

        abs_percentiles[col] = percentile_series

    abs_percentiles = abs_percentiles.apply(_safe_clip_0_100)

    # Weighted aggregation using learned weights
    W = _normalize_weights(weights)

    def _weighted_avg(row: pd.Series) -> float:
        valid = {k: v for k, v in row.items() if not pd.isna(v) and W.get(k, 0) > 0}
        if not valid:
            return np.nan
        total_weight = sum(W[k] for k in valid)
        weighted_sum = sum(valid[k] * W[k] for k in valid)
        return float(weighted_sum / total_weight) if total_weight > 0 else np.nan

    absolute_score = abs_percentiles.apply(_weighted_avg, axis=1)

    # Relative score
    rel_universe = df[[target_col] + peers_avail].copy()
    relative_score = calculate_relative_score(rel_universe, window=126, target_col=target_col)
    relative_score = _safe_clip_0_100(relative_score)

    # Hybrid + smoothing
    hybrid = 0.80 * absolute_score + 0.20 * relative_score
    smoothed = _ema(hybrid, span=5)

    out = _safe_clip_0_100(smoothed).dropna().iloc[-504:]
    return out


def get_csi_lasso_score(excel_path_or_df, use_cached_weights: bool = True) -> Optional[float]:
    """
    Get latest CSI MARS score using dynamic LASSO weighting.

    This is the main entry point for CSI scoring with LASSO.

    Parameters
    ----------
    excel_path_or_df : str or pd.DataFrame
        Path to Excel file or pre-loaded DataFrame
    use_cached_weights : bool
        Whether to use cached LASSO weights (faster)

    Returns
    -------
    float or None
        Latest CSI MARS score (0-100) or None if computation fails

    Example
    -------
    >>> score = get_csi_lasso_score("data.xlsx")
    >>> print(f"CSI MARS (LASSO): {score:.1f}")
    """
    try:
        # Load prices
        if isinstance(excel_path_or_df, str):
            from .data_loader import load_prices_for_mars
            prices_df = load_prices_for_mars(excel_path_or_df)
            excel_path = excel_path_or_df
        else:
            prices_df = excel_path_or_df
            excel_path = None

        # Train/load LASSO weights
        if excel_path and use_cached_weights:
            lasso_weights = _train_csi_lasso_weights_cached(excel_path)
        else:
            # Compute without caching (for DataFrame input)
            print("⚠️  Cannot cache LASSO weights for DataFrame input. Using DEFAULT_WEIGHTS.")
            from .mars_lite_scorer import generate_csi_score_history
            score_series = generate_csi_score_history(prices_df, agg_method="weighted")
            return float(score_series.iloc[-1]) if not score_series.empty else None

        # Compute score with LASSO weights
        score_series = compute_csi_score_with_lasso(prices_df, lasso_weights, use_latest_weights=True)

        if score_series is not None and not score_series.empty:
            return float(score_series.iloc[-1])
        return None

    except Exception as e:
        print(f"Warning: Could not compute CSI LASSO score: {e}")
        return None
