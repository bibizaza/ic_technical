"""Pre-registered earnings revisions with fiscal-year roll mapping."""
import numpy as np
import pandas as pd


def detect_rolls(fy1: pd.Series, fy2: pd.Series) -> pd.DatetimeIndex:
    jumps = (fy1.pct_change(fill_method=None).abs() > .03) | (fy2.pct_change(fill_method=None).abs() > .03)
    matches = (fy1 / fy2.shift(1) - 1).abs() <= .05
    candidates = fy1.index[jumps & matches]
    return candidates[~candidates.year.duplicated()]


def revisions(fy1, fy2, rolls, window=63) -> pd.Series:
    roll_count = pd.Series(fy1.index.isin(rolls), index=fy1.index).cumsum()
    crossed = roll_count.sub(roll_count.shift(window)).gt(0)
    c1 = fy1.shift(window).where(~crossed, fy2.shift(window))
    r1 = np.log(fy1.where(fy1 > 0) / c1.where(c1 > 0))
    c2 = fy2.shift(window)
    r2 = np.log(fy2.where(fy2 > 0) / c2.where(c2 > 0)).where(~crossed)
    return pd.concat([r1, r2], axis=1).mean(axis=1).rename('rev')


def rev_zscore(rev: pd.Series, window=1260, min_obs=756) -> pd.Series:
    past = rev.shift(1).rolling(window, min_periods=min_obs)
    return ((rev - past.mean()) / past.std().replace(0, np.nan)).rename('rev_z')
