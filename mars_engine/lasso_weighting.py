# -*- coding: utf-8 -*-
"""
Dynamic LASSO weighting for MARS momentum scoring.

Provides walk-forward validation framework to learn optimal component weights
using LassoCV regression on forward returns.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


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
    Compute rolling percentile rank using 5-year lookback window.

    This MUST match the percentile calculation used in scoring to ensure
    LASSO training features match the features it will see at inference time.

    For each date, looks at the last `window` days, winsorizes that window to 2-98%,
    and returns the percentile of the current value within that window.
    """
    def _fn(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        v = pd.Series(x)
        # Winsorize the window (2nd-98th percentile)
        v_w = _winsorize_series(v, lower_q=0.02, upper_q=0.98)
        # Get the last value
        last = v_w.iloc[-1]
        # Return percentile rank of last value within window
        return float((v_w <= last).mean() * 100.0)

    return s.rolling(window, min_periods=window).apply(_fn, raw=True)


def prepare_training_data(
    raw_components_df: pd.DataFrame,  # DataFrame with 5 raw component columns
    target_price_series: pd.Series,  # Price series for computing forward returns
    forward_return_days: int = 63,
) -> pd.DataFrame:
    """
    Build ML-ready dataset for LASSO training.

    CRITICAL: Uses rolling 5-year percentile ranking to match the MARS framework's
    "Historical Percentile Ranking" methodology (Section 3.1). This ensures training
    features exactly match the features used during scoring/inference.

    Parameters
    ----------
    raw_components_df : pd.DataFrame
        DataFrame with columns: ['pure', 'smooth', 'sharpe', 'idio', 'adx']
        containing RAW component values (not percentiles)
    target_price_series : pd.Series
        Target asset price series for computing forward returns
    forward_return_days : int
        Forward return window for target variable

    Returns
    -------
    pd.DataFrame
        Training data with percentile-ranked features and forward returns
    """
    if raw_components_df.empty or target_price_series.empty:
        return pd.DataFrame()

    # Convert raw components to percentile ranks (0-100) using ROLLING 5-year window
    # This matches the MARS framework's historical percentile ranking methodology
    lookback = 252 * 5  # 5-year rolling window

    feature_df = pd.DataFrame(index=raw_components_df.index)
    for col in ['pure', 'smooth', 'sharpe', 'idio', 'adx']:
        if col in raw_components_df.columns:
            # Use rolling percentile rank - MUST match scoring method
            feature_df[col] = _rolling_percentile_rank(raw_components_df[col], window=lookback)

    # Diagnostic: Check if percentiles have variance
    print(f"\n📊 Feature Percentile Stats (after rolling transformation):")
    for col in feature_df.columns:
        valid_data = feature_df[col].dropna()
        if len(valid_data) > 0:
            print(f"  {col}: mean={valid_data.mean():.2f}, std={valid_data.std():.2f}, "
                  f"min={valid_data.min():.2f}, max={valid_data.max():.2f}, "
                  f"count={len(valid_data)}")

    # Compute forward returns
    forward_ret = (target_price_series.shift(-forward_return_days) / target_price_series) - 1.0

    # Combine features and target
    training_df = feature_df.copy()
    training_df['forward_return'] = forward_ret

    result = training_df.dropna()
    print(f"  Final training rows after dropna: {len(result)}")

    return result


def train_lasso_model(
    training_df: pd.DataFrame,
) -> Tuple[Dict[str, float], float]:
    """
    Train LassoCV model to learn optimal component weights.

    Parameters
    ----------
    training_df : pd.DataFrame
        Must have columns: ['pure', 'smooth', 'sharpe', 'idio', 'adx', 'forward_return']

    Returns
    -------
    weights_dict : Dict[str, float]
        Learned weights for each component
    optimal_alpha : float
        Optimal regularization parameter
    """
    if training_df is None or len(training_df) < 5:
        return {}, np.nan

    feature_names = ['pure', 'smooth', 'sharpe', 'idio', 'adx']

    # Ensure all features exist
    missing = [f for f in feature_names if f not in training_df.columns]
    if missing or 'forward_return' not in training_df.columns:
        return {}, np.nan

    X = training_df[feature_names].astype(float).values
    y = training_df['forward_return'].astype(float).values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define alpha grid (logarithmic)
    alphas = np.logspace(-4, -1, 20)

    # Train LassoCV
    model = LassoCV(
        alphas=alphas,
        cv=5,
        random_state=42,
        max_iter=10000,
    )

    try:
        # Diagnostic: Check feature variance and correlations
        print(f"\n🔍 LASSO Training Diagnostics:")
        print(f"  Training samples: {len(X)}")
        print(f"  Feature std devs: {X.std(axis=0)}")
        print(f"  Target (y) std dev: {y.std():.6f}")
        print(f"  Target (y) mean: {y.mean():.6f}")

        # Check correlations with target
        correlations = {}
        for i, name in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations[name] = corr
        print(f"  Feature-target correlations: {correlations}")

        model.fit(X_scaled, y)
        optimal_alpha = float(model.alpha_)
        coefs = model.coef_

        print(f"  Optimal alpha: {optimal_alpha:.6f}")
        print(f"  Raw coefficients: {dict(zip(feature_names, coefs))}")

        weights_dict = dict(zip(feature_names, coefs))
        return weights_dict, optimal_alpha
    except Exception as e:
        print(f"❌ LASSO training failed: {e}")
        return {}, np.nan


def perform_walk_forward_validation(
    prices_df: pd.DataFrame,
    raw_components_history: pd.DataFrame,  # Full history of raw component values
    target_col: str,
    training_window_years: int = 5,
    testing_window_years: int = 1,
    forward_return_days: int = 63,
) -> pd.DataFrame:
    """
    Walk-forward validation to generate time-varying LASSO weights.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Full price history
    raw_components_history : pd.DataFrame
        Historical raw component values with columns: ['pure', 'smooth', 'sharpe', 'idio', 'adx']
    target_col : str
        Target asset column name
    training_window_years : int
        Training window length in years
    testing_window_years : int
        Step size between folds in years
    forward_return_days : int
        Forward return window for training

    Returns
    -------
    pd.DataFrame
        Time series of learned weights indexed by fold end date
        Columns: ['pure', 'smooth', 'sharpe', 'idio', 'adx', 'alpha']
    """
    if prices_df.empty or raw_components_history.empty:
        return pd.DataFrame()

    prices_df = prices_df.sort_index()
    raw_components_history = raw_components_history.sort_index()

    min_date = max(prices_df.index.min(), raw_components_history.index.min())
    max_date = min(prices_df.index.max(), raw_components_history.index.max())

    results = []

    # First fold end = training window from start
    first_fold_end = min_date + pd.DateOffset(years=training_window_years)

    # Generate fold dates
    fold_dates = pd.date_range(
        start=first_fold_end,
        end=max_date,
        freq=pd.DateOffset(years=testing_window_years)
    )

    for fold_end_date in fold_dates:
        training_start = fold_end_date - pd.DateOffset(years=training_window_years)

        # Slice raw components for this fold
        components_slice = raw_components_history.loc[training_start:fold_end_date]

        # Prepare training data
        price_slice = prices_df[target_col].loc[training_start:fold_end_date]
        training_df = prepare_training_data(
            raw_components_df=components_slice,
            target_price_series=price_slice,
            forward_return_days=forward_return_days,
        )

        if not training_df.empty:
            weights, alpha = train_lasso_model(training_df)
        else:
            weights, alpha = {}, np.nan

        # Store results
        row = {'Fold_End_Date': fold_end_date, 'alpha': float(alpha)}
        row.update({k: float(v) for k, v in weights.items()})
        results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index('Fold_End_Date').sort_index()

    # Ensure all columns exist
    for col in ['pure', 'smooth', 'sharpe', 'idio', 'adx', 'alpha']:
        if col not in df.columns:
            df[col] = 0.0

    return df.fillna(0.0)
