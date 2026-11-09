"""1-week trading range from a daily HAR volatility model (one method for every IC Technical slide).

Why this method
---------------
Tested on the deck's own data (master_prices.csv, every Wednesday 2016-2026, close 7 days later):
compared with the old range (last 21 closes high/low, symmetric, 3-20% wide), this range scores
6-12% better (interval score for a 68% range) on all 13 equity and commodity slides with enough
history (HAC t -3.9 to -7.1), and is as good as or better than implied vol where implied vol exists.
For crypto it is about equal (BTC better, ETH/SOL equal, XRP slightly worse).
Test code: crypto-vol-model/crypto_vol/range_test_all.py.

Model
-----
Log-HAR (Corsi 2009) on daily Parkinson (high-low) variance, plus the 22-day average of squared
close-to-close returns (captures overnight gaps), yesterday's return and its negative part (leverage
effect). Target: log of the sum of squared daily close-to-close returns over the next 7 calendar days
(calendar, not a row count: the price file has holidays, missing days and crypto weekend rows).
OLS on the last 1000 trading days, refit on every call; forecast = exp(fit + s^2 / 2).
Range = price +/- price * sqrt(weekly variance): about a +/-1 sigma band; 70-76% of weeks close inside.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

WINDOW = 1000  # trading days in the rolling fit
MIN_TRAIN = 300  # minimum training rows; below this the caller falls back to the old range
FEATURES = ["x1", "x5", "x22", "xc22", "xr", "xneg"]


HORIZON = pd.Timedelta(days=7)  # "next week" = the next 7 calendar days (robust to holidays, missing rows, weekends)
MIN_ROWS_IN_WEEK = 3  # a target week with fewer price rows than this is not used for training


def har_frame(price: pd.Series, high: Optional[pd.Series], low: Optional[pd.Series]) -> pd.DataFrame:
    """Daily features (known at the close of day t) and target (squared returns in the next 7 days)."""
    df = pd.DataFrame({"p": price}).dropna()
    df = df[df.p > 0]
    df["hi"] = high.reindex(df.index) if high is not None else np.nan
    df["lo"] = low.reindex(df.index) if low is not None else np.nan
    df["r"] = np.log(df.p).diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        pk = np.log(df.hi / df.lo) ** 2 / (4 * np.log(2))
    ok = (df.hi > df.lo) & np.isfinite(pk)
    pk = pk.where(ok, df.r ** 2)  # no usable high/low that day: use the squared return
    floor = (df.r ** 2).rolling(250, min_periods=20).median() * 0.05
    pk = np.maximum(pk, floor)
    with np.errstate(divide="ignore"):
        df["x1"] = np.log(pk)
        df["x5"] = np.log(pk.rolling(5).mean())
        df["x22"] = np.log(pk.rolling(22).mean())
        df["xc22"] = np.log((df.r ** 2).rolling(22).mean())
        df["xr"], df["xneg"] = df.r, np.minimum(df.r, 0)
    # target: sum of squared returns of the rows dated in (t, t + 7 days]
    cs = np.concatenate([[0.0], np.nancumsum((df.r ** 2).fillna(0).to_numpy())])
    t = df.index.to_numpy()
    j = np.searchsorted(t, t + HORIZON.to_timedelta64(), side="right")  # rows 0..j-1 are dated <= t + 7d
    k = np.arange(len(df)) + 1  # first row after t
    fwd = cs[j] - cs[k]
    rows = j - k
    with np.errstate(divide="ignore"):
        df["y"] = np.where(rows >= MIN_ROWS_IN_WEEK, np.log(fwd), np.nan)
    df["week_end"] = df.index + HORIZON  # the target is fully known once this date has passed
    return df


def har_week_variance(price: pd.Series, high: Optional[pd.Series] = None,
                      low: Optional[pd.Series] = None) -> Optional[float]:
    """Forecast variance of the log return over the next week, made at the last date of ``price``.

    Uses only data up to that date. Returns None if there is not enough history.
    """
    price = pd.to_numeric(price, errors="coerce").dropna().sort_index()
    if len(price) < MIN_TRAIN + 30:
        return None
    df = har_frame(price, high, low)
    i = len(df) - 1
    last = df.index[i]
    # training rows whose next-week target is fully known at the last date
    tr = df.iloc[max(0, i - WINDOW): i]
    tr = tr[tr.week_end <= last][FEATURES + ["y"]].replace([np.inf, -np.inf], np.nan).dropna()
    x = df.iloc[i][FEATURES].to_numpy(float)
    if len(tr) < MIN_TRAIN or not np.isfinite(x).all():
        return None
    X = np.column_stack([np.ones(len(tr)), tr[FEATURES].to_numpy()])
    y = tr["y"].to_numpy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = np.var(y - X @ b, ddof=X.shape[1])
    var = float(np.exp(b[0] + x @ b[1:] + 0.5 * s2))
    return var if np.isfinite(var) and var > 0 else None


def har_trading_range(price: pd.Series, high: Optional[pd.Series] = None, low: Optional[pd.Series] = None,
                      current_price: Optional[float] = None) -> Optional[Tuple[float, float]]:
    """(lower, higher) 1-week range around ``current_price``, or None if HAR cannot be computed."""
    var = har_week_variance(price, high, low)
    if var is None:
        return None
    p = float(current_price) if current_price is not None else float(pd.to_numeric(price, errors="coerce").dropna().iloc[-1])
    half = p * np.sqrt(var)
    return (p - half, p + half)
