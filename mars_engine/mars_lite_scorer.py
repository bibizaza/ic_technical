# -*- coding: utf-8 -*-
"""
Lightweight MARS scorer with data-aware caching.

This module keeps the original "lite" MARS algorithm used in the IC app:
- 5 absolute components computed on full history:
    pure momentum (12m + 0.5*6m + 0.25*3m),
    trend smoothness (fraction of positive days over ~126),
    Sharpe (log-return SR over ~126, annualised),
    idiosyncratic momentum (residual-vs-benchmark, vectorised),
    ADX(14).
- Each absolute component is converted to a rolling 5-year percentile of
  the LAST value within that window, after winsorising the window at 2/98.
- Absolute score = Average of the Top 2 component percentiles.
- Relative score = 6-month return rank vs peer group (cross-sectional pct).
- Hybrid score = 0.80 * absolute + 0.20 * relative, then 5-day EMA.
- Output limited to the last ~2y (≈504 trading days) and clipped to [0,100].

This version adds an in-memory LRU cache keyed by a stable fingerprint of
the *exact columns used by the scorer* (target, high/low, and available peers).
If the fingerprint hasn't changed since the last computation, the cached
Series is returned immediately.

Public:
    - generate_spx_score_history(prices_df: pd.DataFrame) -> pd.Series
    - generate_csi_score_history(prices_df: pd.DataFrame) -> pd.Series
    - clear_mars_cache() -> None
    - mars_cache_info() -> dict

Notes:
- No Streamlit dependency; caching works in any environment.
- Cache is process-local (clears on app restart). For disk persistence,
  you could wrap these in st.cache_data in your app, but this module-level
  cache already prevents recomputation while the app is running.

"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Optional
from collections import OrderedDict
import hashlib

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Peer universes (unchanged relative to the IC app "lite" scorer)
# -----------------------------------------------------------------------------
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

# For CSI we replace "SHSZ300 Index" with "SPX Index" in the peer list
PEER_GROUP_CSI: List[str] = [("SPX Index" if t == "SHSZ300 Index" else t) for t in PEER_GROUP_SPX]

# -----------------------------------------------------------------------------
# Internal cache (in-memory LRU)
# -----------------------------------------------------------------------------
_ENGINE_VERSION = "2025-10-07/top2+rel126+ema5+winsor2-98"
_CACHE_MAXSIZE = 64
_SCORE_CACHE: "OrderedDict[Tuple[str, str], pd.Series]" = OrderedDict()


def _cache_get(key: Tuple[str, str]) -> Optional[pd.Series]:
    s = _SCORE_CACHE.get(key)
    if s is not None:
        # LRU bump
        _SCORE_CACHE.move_to_end(key)
    return s


def _cache_put(key: Tuple[str, str], value: pd.Series) -> None:
    _SCORE_CACHE[key] = value
    _SCORE_CACHE.move_to_end(key)
    if len(_SCORE_CACHE) > _CACHE_MAXSIZE:
        _SCORE_CACHE.popitem(last=False)  # evict LRU


def clear_mars_cache() -> None:
    """Clear all cached momentum series."""
    _SCORE_CACHE.clear()


def mars_cache_info() -> Dict[str, object]:
    """Return cache stats useful for debugging/monitoring."""
    return {
        "engine_version": _ENGINE_VERSION,
        "entries": len(_SCORE_CACHE),
        "keys": list(_SCORE_CACHE.keys())[-8:],  # last few keys
        "maxsize": _CACHE_MAXSIZE,
    }


def _df_fingerprint(df: pd.DataFrame, *, tag: str) -> str:
    """
    Stable fingerprint of the data used by the scorer.

    Uses pandas' stable row-wise hashing + Blake2b for a compact digest.
    Incorporates an engine version tag so cache invalidates if logic changes.
    """
    ser = pd.util.hash_pandas_object(df, index=True)  # int64 vector
    digest = hashlib.blake2b(ser.values.tobytes(), digest_size=16)
    digest.update(_ENGINE_VERSION.encode("utf-8"))
    digest.update(tag.encode("utf-8"))
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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
    For each rolling window, winsorize to [p2,p98] *within the window* and
    return the percentile rank (0–100) of the *last* value in that window.
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


# -----------------------------------------------------------------------------
# Absolute component calculations (vectorised)
# -----------------------------------------------------------------------------
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
    """
    Rolling-OLS, vectorised via rolling moments:
        beta = Cov(Ra,Rb)/Var(Rb), alpha = E[Ra]-beta*E[Rb],
        resid_t = Ra_t - (alpha + beta*Rb_t);
    accumulate simple residual returns.
    """
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

    resid_lr = ra - (alpha + beta * rb)
    resid_lr = resid_lr.fillna(0.0)
    resid_simple = np.expm1(resid_lr)

    idio = (1.0 + resid_simple).cumprod() - 1.0
    return idio


def calculate_raw_adx(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    period  = 14
    up_move   = high.diff()
    down_move = low.shift(1) - low  # positive when new low < prior low

    plus_dm  = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)

    atr     = tr.rolling(period, min_periods=period).sum()  # Wilder approx
    plus_di = 100.0 * plus_dm.rolling(period, min_periods=period).sum() / atr
    minus_di= 100.0 * minus_dm.rolling(period, min_periods=period).sum() / atr

    dx  = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period, min_periods=period).mean()
    return adx


# -----------------------------------------------------------------------------
# Relative momentum
# -----------------------------------------------------------------------------
def calculate_relative_score(prices: pd.DataFrame, window: int = 126, target_col: str = "") -> pd.Series:
    """
    Cross-sectional percentile rank of target's window return vs available peers.
    """
    ret = prices.pct_change(window)
    if target_col and target_col not in ret.columns:
        target_col = ret.columns[0]
    elif not target_col:
        target_col = ret.columns[0]
    ranks = ret.rank(axis=1, pct=True)
    return ranks[target_col] * 100.0


# -----------------------------------------------------------------------------
# Core engine with caching
# -----------------------------------------------------------------------------
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
    Compute momentum score series using the "lite" MARS recipe with caching.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df = df.sort_index().copy()

    # Ensure hi/lo exist
    if hi_col not in df.columns:
        df[hi_col] = df[target_col] * 1.01
    if lo_col not in df.columns:
        df[lo_col] = df[target_col] * 0.99

    # Build the actual input universe: target + available peers
    peers_avail = [c for c in peer_universe if c in df.columns]
    used_cols = [target_col, hi_col, lo_col] + peers_avail
    df_used = df[used_cols].copy()

    # Compose cache key
    tag = f"{target_col}|{hi_col}|{lo_col}|{','.join(peers_avail)}"
    fp = _df_fingerprint(df_used, tag=tag)
    key: Tuple[str, str] = (target_col.upper(), fp)

    cached = _cache_get(key)
    if cached is not None:
        # Return a view (avoid accidental mutation of cached object)
        return cached.copy()

    # Proceed with compute
    px = df[target_col]
    hi = df[hi_col]
    lo = df[lo_col]

    bench_col = next((c for c in bench_candidates if c in df.columns), target_col)
    bench     = df[bench_col]

    raw = pd.DataFrame({
        "pure":   calculate_raw_pure_momentum(px),
        "smooth": calculate_raw_trend_smoothness(px),
        "sharpe": calculate_raw_sharpe_ratio(px),
        "idio":   calculate_raw_idiosyncratic_momentum(px, bench),
        "adx":    calculate_raw_adx(hi, lo, px),
    })

    # Winsorised 5-year percentile of last value in each window
    lookback = 252 * 5
    abs_percentiles = pd.DataFrame(index=raw.index)
    for col in raw.columns:
        abs_percentiles[col] = _rolling_percentile_of_last(raw[col], lookback)
    abs_percentiles = abs_percentiles.apply(_safe_clip_0_100)

    # Absolute score = Average of TOP 2 component percentiles
    def _avg_top2(row: pd.Series) -> float:
        r = row.dropna()
        if r.size < 2:
            return np.nan
        top2 = np.sort(r.values)[-2:]
        return float(np.mean(top2))

    absolute_score = abs_percentiles.apply(_avg_top2, axis=1)

    # Relative score
    rel_universe = df[[target_col] + peers_avail].copy()
    relative_score = calculate_relative_score(rel_universe, window=126, target_col=target_col)
    relative_score = _safe_clip_0_100(relative_score)

    # Hybrid + smoothing
    hybrid   = 0.80 * absolute_score + 0.20 * relative_score
    smoothed = _ema(hybrid, span=5)

    out = _safe_clip_0_100(smoothed).dropna().iloc[-504:]

    # Save to cache and return
    _cache_put(key, out)
    return out.copy()


# -----------------------------------------------------------------------------
# Public wrappers (SPX & CSI)
# -----------------------------------------------------------------------------
def generate_spx_score_history(prices_df: pd.DataFrame) -> pd.Series:
    """
    SPX wrapper. Expects columns: SPX, SPX_high, SPX_low plus as many peers
    as available (matching PEER_GROUP_SPX). Returns last ~2y [0,100].
    """
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
    CSI (SHSZ300) wrapper. Relative peers = SPX peers with 'SHSZ300 Index'
    replaced by 'SPX Index' as requested. Expects columns:
    CSI, CSI_high, CSI_low (+ available peers).
    """
    return _generate_score_history(
        prices_df,
        target_col="CSI",
        hi_col="CSI_high",
        lo_col="CSI_low",
        peer_universe=PEER_GROUP_CSI,
        bench_candidates=["MXWO Index", "MXWO", "CSI", "SPX"],
    )
