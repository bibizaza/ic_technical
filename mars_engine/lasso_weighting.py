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


def prepare_training_data(
    raw_components_df: pd.DataFrame,  # DataFrame with 5 raw component columns
    target_price_series: pd.Series,  # Price series for computing forward returns
    forward_return_days: int = 63,
) -> pd.DataFrame:
    """
    Build ML-ready dataset for LASSO training.

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

    # Convert raw components to percentile ranks (0-100)
    feature_df = pd.DataFrame(index=raw_components_df.index)
    for col in ['pure', 'smooth', 'sharpe', 'idio', 'adx']:
        if col in raw_components_df.columns:
            feature_df[col] = raw_components_df[col].rank(pct=True) * 100.0

    # Compute forward returns
    forward_ret = (target_price_series.shift(-forward_return_days) / target_price_series) - 1.0

    # Combine features and target
    training_df = feature_df.copy()
    training_df['forward_return'] = forward_ret

    return training_df.dropna()


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
        model.fit(X_scaled, y)
        optimal_alpha = float(model.alpha_)
        coefs = model.coef_

        weights_dict = dict(zip(feature_names, coefs))
        return weights_dict, optimal_alpha
    except Exception:
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
