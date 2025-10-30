# -*- coding: utf-8 -*-
"""
Raw momentum component calculations for MARS scoring.

This module extracts the 5 absolute momentum components as reusable functions
for use in both scoring and LASSO training.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np


def calculate_raw_pure_momentum(close: pd.Series) -> pd.Series:
    """
    Pure momentum: weighted blend of 12m, 6m, 3m returns.
    Weights: 12m=0.4, 6m=0.4, 3m=0.2 (matches reference implementation)

    Parameters
    ----------
    close : pd.Series
        Price series indexed by date

    Returns
    -------
    pd.Series
        Raw pure momentum values
    """
    r12m = close.pct_change(252)
    r6m = close.pct_change(126)
    r3m = close.pct_change(63)
    # Reference weights: 12m=0.4, 6m=0.4, 3m=0.2
    return 0.4 * r12m + 0.4 * r6m + 0.2 * r3m


def calculate_raw_trend_smoothness(close: pd.Series) -> pd.Series:
    """
    Trend smoothness: fraction of positive days over 126-day window.

    Parameters
    ----------
    close : pd.Series
        Price series indexed by date

    Returns
    -------
    pd.Series
        Raw smoothness values (0-1)
    """
    daily = close.pct_change()
    return daily.gt(0).rolling(126, min_periods=126).mean()


def calculate_raw_sharpe_ratio(close: pd.Series) -> pd.Series:
    """
    Sharpe ratio: annualized simple-return Sharpe over 126-day window.
    Matches reference implementation using simple (not log) returns.

    Parameters
    ----------
    close : pd.Series
        Price series indexed by date

    Returns
    -------
    pd.Series
        Raw Sharpe ratio values
    """
    # Use simple returns (not log returns) to match reference
    returns = close.pct_change()
    win = 126
    mean = returns.rolling(win, min_periods=win).mean()
    std = returns.rolling(win, min_periods=win).std()
    # Annualize: mean * 252 / (std * sqrt(252))
    return (mean * 252) / (std * np.sqrt(252))


def calculate_raw_idiosyncratic_momentum(close: pd.Series, benchmark: pd.Series) -> pd.Series:
    """
    Idiosyncratic momentum: 6-month cumulative return of residuals from 12-month OLS.

    Reference implementation:
    - Regress asset returns vs benchmark over 12 months (252 days)
    - Take residuals from last 6 months (126 days)
    - Compute cumulative return of those residuals

    Parameters
    ----------
    close : pd.Series
        Asset price series indexed by date
    benchmark : pd.Series
        Benchmark price series indexed by date

    Returns
    -------
    pd.Series
        Raw idiosyncratic momentum values (6m cumulative residual return)
    """
    # Use simple returns to match reference
    ra = close.pct_change()
    rb = benchmark.pct_change()

    # 12-month regression window
    reg_win = 252
    # 6-month residual window
    resid_win = 126

    # Rolling OLS via rolling moments
    ma = ra.rolling(reg_win, min_periods=reg_win).mean()
    mb = rb.rolling(reg_win, min_periods=reg_win).mean()

    e_ra_rb = (ra * rb).rolling(reg_win, min_periods=reg_win).mean()
    e_rb2 = (rb * rb).rolling(reg_win, min_periods=reg_win).mean()

    cov = e_ra_rb - ma * mb
    var_b = e_rb2 - mb * mb
    beta = cov / var_b.replace(0.0, np.nan)
    alpha = ma - beta * mb

    # Compute residuals
    resid = ra - (alpha + beta * rb)
    resid = resid.fillna(0.0)

    # Cumulative return of last 6 months of residuals
    # For each date, take last 126 days of residuals and compute cumulative return
    def _cumulative_resid_return(x):
        if len(x) < resid_win:
            return np.nan
        # Take last 126 days
        resid_window = x[-resid_win:]
        # Cumulative return: (1 + r1) * (1 + r2) * ... - 1
        return (1 + resid_window).prod() - 1

    idio = resid.rolling(resid_win, min_periods=resid_win).apply(_cumulative_resid_return, raw=True)
    return idio


def calculate_raw_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX) using Wilder's smoothing (RMA).

    Matches reference implementation using EWM with alpha=1/length for RMA.

    Parameters
    ----------
    high : pd.Series
        High prices indexed by date
    low : pd.Series
        Low prices indexed by date
    close : pd.Series
        Close prices indexed by date
    period : int, default 14
        ADX period length

    Returns
    -------
    pd.Series
        Raw ADX values
    """
    def _rma(s: pd.Series, length: int) -> pd.Series:
        """Wilder's smoothing (RMA) via EMA with alpha = 1/length"""
        return s.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    up_move = high.diff()
    down_move = -low.diff()  # positive when low is falling

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index
    )

    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)

    # Use RMA (Wilder smoothing) instead of simple rolling
    atr = _rma(tr, period)
    plus_di = 100.0 * _rma(plus_dm, period) / atr
    minus_di = 100.0 * _rma(minus_dm, period) / atr

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _rma(dx, period)
    return adx


def compute_raw_components(
    df: pd.DataFrame,
    target_col: str,
    hi_col: str,
    lo_col: str,
    bench_col: str,
) -> pd.DataFrame:
    """
    Compute all 5 raw momentum components for a target asset.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame indexed by date
    target_col : str
        Column name for target asset close prices
    hi_col : str
        Column name for target asset high prices
    lo_col : str
        Column name for target asset low prices
    bench_col : str
        Column name for benchmark prices

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['pure', 'smooth', 'sharpe', 'idio', 'adx']
        indexed by date, containing raw component values
    """
    px = df[target_col]
    hi = df[hi_col]
    lo = df[lo_col]
    bench = df[bench_col]

    return pd.DataFrame({
        "pure": calculate_raw_pure_momentum(px),
        "smooth": calculate_raw_trend_smoothness(px),
        "sharpe": calculate_raw_sharpe_ratio(px),
        "idio": calculate_raw_idiosyncratic_momentum(px, bench),
        "adx": calculate_raw_adx(hi, lo, px),
    })
