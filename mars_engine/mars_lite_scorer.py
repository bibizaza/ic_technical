# -*- coding: utf-8 -*-
"""
Lightweight MARS scorer (generic) + SPX & CSI wrappers

This module provides a compact, vectorised momentum engine used by the
Investment Committee app. It implements five absolute components and a
cross-sectional relative component, then combines them (80/20) and
applies a 5-day EMA. Two convenience wrappers are provided:

- generate_spx_score_history(prices_df)
- generate_csi_score_history(prices_df)

For CSI, the peer group equals the SPX peer group but with
"SHSZ300 Index" replaced by "SPX Index".

Expected columns (per wrapper):
- SPX:  'SPX', 'SPX_high', 'SPX_low', plus as many peers as available
- CSI:  'CSI', 'CSI_high', 'CSI_low', plus as many peers as available

Returned series is clipped to [0,100] and limited to ~2y (504 trading days).
"""

from __future__ import annotations

from typing import Iterable, List, Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Base peer universe (used for SPX)
# ---------------------------------------------------------------------------
PEER_GROUP_SPX: List[str] = [
    "CCMP Index",
    "IBOV Index",
    "MEXBOL Index",
    "SXXP Index",
    "UKX Index",
    "SMI Index",
    "HSI Index",
    "SHSZ300 Index",
    "NKY Index",
    "SENSEX Index",
    "DAX Index",
    "MXWO Index",
    "USGG10YR Index",
    "GECU10YR Index",
    "CL1 Comdty",
    "GCA Comdty",
    "DXY Curncy",
    "XBTUSD Curncy",
]

# For CSI, replace "SHSZ300 Index" with "SPX Index"
PEER_GROUP_CSI: List[str] = [("SPX Index" if t == "SHSZ300 Index" else t) for t in PEER_GROUP_SPX]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _winsorize(s: pd.Series, lower_percent: float = 0.02, upper_percent: float = 0.98) -> pd.Series:
    if s.size == 0:
        return s
    lo = s.quantile(lower_percent)
    hi = s.quantile(upper_percent)
    return s.clip(lo, hi)


def _rolling_percentile_of_last(
    s: pd.Series, window: int, *, lower_percent: float = 0.02, upper_percent: float = 0.98
) -> pd.Series:
    """
    For each window, winsorize values to [p2,p98] and return the percentile rank
    (0–100) of the LAST value within that winsorized window.
    """
    def _fn(x: np.ndarray) -> float:
        v = pd.Series(x)
        v_w = _winsorize(v, lower_percent, upper_percent)
        last = v_w.iloc[-1]
        return float((v_w <= last).mean() * 100.0)
    return s.rolling(window, min_periods=window).apply(_fn, raw=True)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _safe_clip_0_100(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=100.0)


# ---------------------------------------------------------------------------
# Raw absolute components (vectorised)
# ---------------------------------------------------------------------------
def calculate_raw_pure_momentum(close: pd.Series) -> pd.Series:
    r12m = close.pct_change(252)
    r6m  = close.pct_change(126)
    r3m  = close.pct_change( 63)
    return r12m + 0.5 * r6m + 0.25 * r3m


def calculate_raw_trend_smoothness(close: pd.Series) -> pd.Series:
    daily = close.pct_change()
    return daily.gt(0).rolling(126, min_periods=126).mean()


def calculate_raw_sharpe_ratio(close: pd.Series) -> pd.Series:
    lr = np.log(close / close.shift(1))
    win = 126
    mean = lr.rolling(win, min_periods=win).mean()
    std  = lr.rolling(win, min_periods=win).std()
    return (mean / std) * np.sqrt(252.0)


def calculate_raw_idiosyncratic_momentum(close: pd.Series, benchmark: pd.Series) -> pd.Series:
    win = 126
    ra = np.log(close / close.shift(1))
    rb = np.log(benchmark / benchmark.shift(1))

    ma = ra.rolling(win, min_periods=win).mean()
    mb = rb.rolling(win, min_periods=win).mean()

    e_ra_rb = (ra * rb).rolling(win, min_periods=win).mean()
    e_rb2   = (rb * rb).rolling(win, min_periods=win).mean()

    cov   = e_ra_rb - ma * mb
    var_b = e_rb2 - mb * mb
    beta  = cov / var_b.replace(0.0, np.nan)
    alpha = ma - beta * mb

    resid_lr    = ra - (alpha + beta * rb)
    resid_lr    = resid_lr.fillna(0.0)
    resid_simpl = np.expm1(resid_lr)
    idio        = (1.0 + resid_simpl).cumprod() - 1.0
    return idio


def calculate_raw_adx(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    period  = 14
    up_move   = high.diff()
    down_move = low.shift(1) - low  # positive when new low < prior low
    plus_dm   = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm  = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)

    atr     = tr.rolling(period, min_periods=period).sum()
    plus_di = 100.0 * plus_dm.rolling(period, min_periods=period).sum() / atr
    minus_di= 100.0 * minus_dm.rolling(period, min_periods=period).sum() / atr

    dx  = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period, min_periods=period).mean()
    return adx


# ---------------------------------------------------------------------------
# Relative momentum
# ---------------------------------------------------------------------------
def calculate_relative_score(prices: pd.DataFrame, window: int = 126, target_col: str = "") -> pd.Series:
    """
    Cross-sectional percentile rank of the target's window return vs available peers.
    """
    ret = prices.pct_change(window)
    if target_col and target_col not in ret.columns:
        # Fallback to first column
        target_col = ret.columns[0]
    elif not target_col:
        target_col = ret.columns[0]
    ranks = ret.rank(axis=1, pct=True)
    return ranks[target_col] * 100.0


# ---------------------------------------------------------------------------
# Master (generic) function
# ---------------------------------------------------------------------------
def _generate_score_history(
    df: pd.DataFrame,
    *,
    target_col: str,
    hi_col: str,
    lo_col: str,
    peer_universe: Iterable[str],
    bench_candidates: Iterable[str],
) -> pd.Series:
    """
    Core routine used by SPX & CSI wrappers.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df = df.sort_index().copy()

    # Ensure hi/lo exist
    if hi_col not in df.columns:
        df[hi_col] = df[target_col] * 1.01
    if lo_col not in df.columns:
        df[lo_col] = df[target_col] * 0.99

    px   = df[target_col]
    hi   = df[hi_col]
    lo   = df[lo_col]

    bench_col = next((c for c in bench_candidates if c in df.columns), target_col)
    bench     = df[bench_col]

    raw = pd.DataFrame({
        "pure":   calculate_raw_pure_momentum(px),
        "smooth": calculate_raw_trend_smoothness(px),
        "sharpe": calculate_raw_sharpe_ratio(px),
        "idio":   calculate_raw_idiosyncratic_momentum(px, bench),
        "adx":    calculate_raw_adx(hi, lo, px),
    })

    lookback = 252 * 5
    abs_percentiles = pd.DataFrame(index=raw.index)
    for col in raw.columns:
        abs_percentiles[col] = _rolling_percentile_of_last(raw[col], lookback)

    abs_percentiles = abs_percentiles.apply(_safe_clip_0_100)

    def _avg_top2(row: pd.Series) -> float:
        r = row.dropna()
        if r.size < 2:
            return np.nan
        top2 = np.sort(r.values)[-2:]
        return float(np.mean(top2))

    absolute_score = abs_percentiles.apply(_avg_top2, axis=1)

    # Relative: use whatever peers are available
    cols = [target_col] + [c for c in peer_universe if c in df.columns]
    rel_universe = df[cols].copy()
    relative_score = calculate_relative_score(rel_universe, window=126, target_col=target_col)
    relative_score = _safe_clip_0_100(relative_score)

    hybrid   = 0.80 * absolute_score + 0.20 * relative_score
    smoothed = _ema(hybrid, span=5)
    out      = _safe_clip_0_100(smoothed).dropna()
    return out.iloc[-504:]


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------
def generate_spx_score_history(prices_df: pd.DataFrame) -> pd.Series:
    return _generate_score_history(
        prices_df,
        target_col="SPX",
        hi_col="SPX_high",
        lo_col="SPX_low",
        peer_universe=PEER_GROUP_SPX,
        bench_candidates=["MXWO Index", "MXWO", "SPX"],
    )


def generate_csi_score_history(prices_df: pd.DataFrame) -> pd.Series:
    """
    CSI (SHSZ300) wrapper. Relative peers = PEER_GROUP_SPX with
    'SHSZ300 Index' → 'SPX Index' substitution.
    """
    return _generate_score_history(
        prices_df,
        target_col="CSI",
        hi_col="CSI_high",
        lo_col="CSI_low",
        peer_universe=PEER_GROUP_CSI,
        bench_candidates=["MXWO Index", "MXWO", "CSI", "SPX"],
    )
