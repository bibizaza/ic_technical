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
    Pure momentum: 12m + 0.5*6m + 0.25*3m returns.

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
    return r12m + 0.5 * r6m + 0.25 * r3m


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
    Sharpe ratio: annualized log-return Sharpe over 126-day window.

    Parameters
    ----------
    close : pd.Series
        Price series indexed by date

    Returns
    -------
    pd.Series
        Raw Sharpe ratio values
    """
    lr = np.log(close / close.shift(1))
    win = 126
    mean = lr.rolling(win, min_periods=win).mean()
    std = lr.rolling(win, min_periods=win).std()
    return (mean / std) * np.sqrt(252.0)


def calculate_raw_idiosyncratic_momentum(close: pd.Series, benchmark: pd.Series) -> pd.Series:
    """
    Idiosyncratic momentum: cumulative residual returns from rolling OLS.

    Rolling-OLS, vectorized via rolling moments:
        beta = Cov(Ra,Rb)/Var(Rb), alpha = E[Ra]-beta*E[Rb],
        resid_t = Ra_t - (alpha + beta*Rb_t);
    accumulate simple residual returns.

    Parameters
    ----------
    close : pd.Series
        Asset price series indexed by date
    benchmark : pd.Series
        Benchmark price series indexed by date

    Returns
    -------
    pd.Series
        Raw idiosyncratic momentum values
    """
    win = 126
    ra = np.log(close / close.shift(1))
    rb = np.log(benchmark / benchmark.shift(1))

    ma = ra.rolling(win, min_periods=win).mean()
    mb = rb.rolling(win, min_periods=win).mean()

    e_ra_rb = (ra * rb).rolling(win, min_periods=win).mean()
    e_rb2 = (rb * rb).rolling(win, min_periods=win).mean()

    cov = e_ra_rb - ma * mb
    var_b = e_rb2 - mb * mb
    beta = cov / var_b.replace(0.0, np.nan)
    alpha = ma - beta * mb

    resid_lr = ra - (alpha + beta * rb)
    resid_lr = resid_lr.fillna(0.0)
    resid_simple = np.expm1(resid_lr)

    idio = (1.0 + resid_simple).cumprod() - 1.0
    return idio


def calculate_raw_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).

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
    up_move = high.diff()
    down_move = low.shift(1) - low  # positive when new low < prior low

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

    atr = tr.rolling(period, min_periods=period).sum()  # Wilder approx
    plus_di = 100.0 * plus_dm.rolling(period, min_periods=period).sum() / atr
    minus_di = 100.0 * minus_dm.rolling(period, min_periods=period).sum() / atr

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period, min_periods=period).mean()
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
