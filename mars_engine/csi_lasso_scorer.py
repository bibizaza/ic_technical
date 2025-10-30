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

    print(f"🔍 Raw LASSO weights (before normalization): {learned_weights}")

    # Check if LASSO produced meaningful weights (not all zeros)
    total_abs_weight = sum(abs(w) for w in learned_weights.values())
    if total_abs_weight < 0.001:
        print("⚠️  LASSO produced all-zero weights. Falling back to top2 aggregation.")
        from .mars_lite_scorer import generate_csi_score_history
        return generate_csi_score_history(prices_df, agg_method="top2")

    # Normalize weights
    learned_weights = _normalize_weights(learned_weights)

    print(f"📊 Normalized LASSO weights: {learned_weights}")

    # Now compute the score using these weights
    return _compute_csi_score_with_weights(prices_df, learned_weights)


def _winsorize_series(s: pd.Series, lower_q: float = 0.02, upper_q: float = 0.98) -> pd.Series:
    """Winsorize series to lower and upper quantiles."""
    s_clean = s.dropna()
    if s_clean.empty:
        return s
    lo = s_clean.quantile(lower_q)
    hi = s_clean.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def _rolling_percentile_rank(s: pd.Series, window: int = 252 * 5) -> pd.Series:
    """
    Compute rolling percentile rank matching the top2 method.

    For each date, look at the last `window` days, winsorize that window to 2-98%,
    and return the percentile of the current value within that window.

    This EXACTLY matches _rolling_percentile_of_last used in top2 scoring.
    """
    def _fn(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        v = pd.Series(x)
        # Winsorize the window
        v_w = _winsorize_series(v, lower_q=0.02, upper_q=0.98)
        # Get the last value
        last = v_w.iloc[-1]
        # Return percentile rank of last value within window
        return float((v_w <= last).mean() * 100.0)

    return s.rolling(window, min_periods=window).apply(_fn, raw=True)


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

    # Convert to percentiles using ROLLING 5-year window (matches top2 method)
    # This is CRITICAL: must match the percentile calculation method exactly
    lookback = 252 * 5  # 5-year rolling window

    abs_percentiles = pd.DataFrame(index=raw_components.index)
    for col in raw_components.columns:
        # Use rolling percentile rank - same as top2 method
        abs_percentiles[col] = _rolling_percentile_rank(raw_components[col], window=lookback)

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
