# -*- coding: utf-8 -*-
"""
Lightweight MARS scoring engine for the Investment Committee application.

This module implements a streamlined MARS engine to calculate the daily
S&P 500 momentum score. It computes five absolute components, converts
each to a winsorised (2/98) percentile vs its own rolling 5-year
history, then takes the "average of the top 2" as the Absolute Score.
It also computes a Relative Score vs a fixed peer group, combines them
as 0.80*Absolute + 0.20*Relative, and smooths with a 5-day EMA.
The function returns the last ~2 years of daily scores.

Input (prices_df):
    - Index: datetime (daily)
    - Columns (required):
        'SPX'       : S&P 500 close
        'SPX_high'  : S&P 500 high (if unavailable, approximate with close*1.01)
        'SPX_low'   : S&P 500 low  (if unavailable, approximate with close*0.99)
      Plus as many of the PEER_GROUP tickers as are available.

Public:
    generate_spx_score_history(prices_df: pd.DataFrame) -> pd.Series
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Peer group universe for relative score (must match the IC app expectation)
# ---------------------------------------------------------------------------
PEER_GROUP: List[str] = [
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _winsorize(s: pd.Series, lower_percent: float = 0.02, upper_percent: float = 0.98) -> pd.Series:
    """Winsorize series in-place at the given percentiles."""
    if s.size == 0:
        return s
    lo = s.quantile(lower_percent)
    hi = s.quantile(upper_percent)
    return s.clip(lo, hi)


def _rolling_percentile_of_last(
    s: pd.Series, window: int, *, lower_percent: float = 0.02, upper_percent: float = 0.98
) -> pd.Series:
    """
    For each window, clip values to [p2, p98] and return the percentile rank
    (0–100) of the LAST value within that winsorized window.
    """
    def _fn(x: np.ndarray) -> float:
        v = pd.Series(x)
        v_w = _winsorize(v, lower_percent, upper_percent)
        # Percentile of last value among winsorized window
        last = v_w.iloc[-1]
        return float((v_w <= last).mean() * 100.0)

    return s.rolling(window, min_periods=window).apply(_fn, raw=True)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _safe_clip_0_100(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=100.0)


# ---------------------------------------------------------------------------
# Raw component calculations (vectorised)
# ---------------------------------------------------------------------------
def calculate_raw_pure_momentum(close: pd.Series) -> pd.Series:
    """
    Pure momentum: weighted sum of 12m, 6m, 3m simple returns
    (weights 1, 0.5, 0.25).
    """
    r12m = close.pct_change(252)
    r6m = close.pct_change(126)
    r3m = close.pct_change(63)
    return r12m + 0.5 * r6m + 0.25 * r3m


def calculate_raw_trend_smoothness(close: pd.Series) -> pd.Series:
    """Trend smoothness: fraction of positive days over 6 months (~126 sessions)."""
    daily = close.pct_change()
    return daily.gt(0).rolling(126, min_periods=126).mean()


def calculate_raw_sharpe_ratio(close: pd.Series) -> pd.Series:
    """Annualised Sharpe ratio over a 6‑month window using daily log returns."""
    lr = np.log(close / close.shift(1))
    win = 126
    mean = lr.rolling(win, min_periods=win).mean()
    std = lr.rolling(win, min_periods=win).std()
    return (mean / std) * np.sqrt(252.0)


def calculate_raw_idiosyncratic_momentum(close: pd.Series, benchmark: pd.Series) -> pd.Series:
    """
    Idiosyncratic momentum: cumulative compounded *residual* return vs benchmark,
    using vectorised rolling OLS (window=126). We compute rolling beta and alpha:

        beta_t = Cov(Ra,Rb) / Var(Rb)
        alpha_t = mean(Ra) - beta_t * mean(Rb)
        resid_t = Ra_t - (alpha_t + beta_t * Rb_t)

    Residuals are log‑return residuals; we convert to simple residual returns
    via expm1 and compound over time.
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


def calculate_raw_adx(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    ADX (Average Directional Index), period=14, vectorised approximation.

    Wilder's TR and DM are approximated with rolling sums (fast and stable).
    """
    period = 14

    up_move = high.diff()
    down_move = low.shift(1) - low  # positive when new low < prior low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period, min_periods=period).sum()  # Wilder sum approx
    plus_di = 100.0 * plus_dm.rolling(period, min_periods=period).sum() / atr
    minus_di = 100.0 * minus_dm.rolling(period, min_periods=period).sum() / atr

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period, min_periods=period).mean()
    return adx


# ---------------------------------------------------------------------------
# Relative momentum (cross‑sectional rank of 6‑month return)
# ---------------------------------------------------------------------------
def calculate_relative_score(prices: pd.DataFrame, window: int = 126, target_col: str = "SPX") -> pd.Series:
    """
    For each date, compute the window return of each column; return the
    target column's cross‑sectional percentile (0–100) across available peers.
    """
    ret = prices.pct_change(window)
    # Percentile rank across columns (axis=1). Using rank(pct=True) gives 0–1.
    ranks = ret.rank(axis=1, pct=True)
    if target_col not in ranks.columns:
        # Fallback: use the first column as target if 'SPX' missing
        target_col = ranks.columns[0]
    return ranks[target_col] * 100.0


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------
def generate_spx_score_history(prices_df: pd.DataFrame) -> pd.Series:
    """
    Compute the daily hybrid MARS momentum score for S&P 500.

    Steps:
      1) Compute five raw components on full history.
      2) Convert each component to a winsorized (p2/p98) rolling 5-year
         percentile (against its own history).
      3) Absolute Score = average of the TOP 2 percentile values per day.
      4) Relative Score = SPX's 6‑month return percentile vs PEER_GROUP.
      5) Hybrid = 0.80*Absolute + 0.20*Relative; then 5‑day EMA.
      6) Return the last ~2 years (≈504 trading days).

    Returns: pd.Series indexed by date with values in [0,100].
    """
    if prices_df is None or prices_df.empty:
        return pd.Series(dtype=float)

    df = prices_df.sort_index().copy()

    # Ensure SPX hi/lo exist (approximate if missing)
    if "SPX_high" not in df.columns:
        df["SPX_high"] = df["SPX"] * 1.01
    if "SPX_low" not in df.columns:
        df["SPX_low"] = df["SPX"] * 0.99

    spx = df["SPX"]
    hi = df["SPX_high"]
    lo = df["SPX_low"]

    # Benchmark for idiosyncratic momentum: prefer MSCI World if available
    bench_candidates: Iterable[str] = ["MXWO Index", "MXWO", "SPX"]  # last resort: itself
    bench_col = next((c for c in bench_candidates if c in df.columns), "SPX")
    bench = df[bench_col]

    # 1) Raw components
    raw = pd.DataFrame(
        {
            "pure":   calculate_raw_pure_momentum(spx),
            "smooth": calculate_raw_trend_smoothness(spx),
            "sharpe": calculate_raw_sharpe_ratio(spx),
            "idio":   calculate_raw_idiosyncratic_momentum(spx, bench),
            "adx":    calculate_raw_adx(hi, lo, spx),
        }
    )

    # 2) Winsorised 5-year (≈1260 trading days) percentile of LAST value per window
    lookback = 252 * 5
    abs_percentiles = pd.DataFrame(index=raw.index)
    for col in raw.columns:
        abs_percentiles[col] = _rolling_percentile_of_last(raw[col], lookback)

    abs_percentiles = abs_percentiles.apply(_safe_clip_0_100)

    # 3) Absolute score: "Average of Top 2" component percentiles
    def _avg_top2(row: pd.Series) -> float:
        r = row.dropna()
        if r.size < 2:
            return np.nan
        top2 = np.sort(r.values)[-2:]
        return float(np.mean(top2))

    absolute_score = abs_percentiles.apply(_avg_top2, axis=1)

    # 4) Relative score vs peer group (use only the columns we actually have)
    cols = ["SPX"] + [c for c in PEER_GROUP if c in df.columns]
    rel_universe = df[cols].copy()
    relative_score = calculate_relative_score(rel_universe, window=126, target_col="SPX")
    relative_score = _safe_clip_0_100(relative_score)

    # 5) Hybrid + smoothing
    hybrid = 0.80 * absolute_score + 0.20 * relative_score
    smoothed = _ema(hybrid, span=5)

    # 6) Return last ~2 years
    out = _safe_clip_0_100(smoothed).dropna()
    return out.iloc[-504:]
