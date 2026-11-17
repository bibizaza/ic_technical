"""Fixed evaluation metrics and paired moving-block bootstrap."""
import numpy as np
import pandas as pd
from research.us_gauge.gauge import BANDS


def metrics(r: pd.Series, cash: pd.Series) -> dict:
    x = pd.concat([r.rename('r'), cash.rename('cash')], axis=1).dropna()
    excess = x.r - x.cash
    wealth = (1 + x.r).cumprod()
    dd = wealth / wealth.cummax().clip(lower=1.) - 1
    cagr = wealth.iloc[-1] ** (252 / len(x)) - 1
    maxdd = dd.min()
    sd = excess.std()
    return {'sharpe': excess.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            'cagr': cagr, 'maxdd': maxdd,
            'calmar': cagr / abs(maxdd) if maxdd < 0 else np.nan,
            'vol': x.r.std() * np.sqrt(252)}


def bootstrap_not_better(r1, r0, cash, block=63, n=2000, seed=7) -> float:
    x = pd.concat([r1.rename('r1'), r0.rename('r0'), cash.rename('cash')], axis=1).dropna()
    size = len(x)
    if size < block:
        raise ValueError('Bootstrap needs at least one full block')
    excess = x[['r1', 'r0']].to_numpy() - x[['cash']].to_numpy()
    rng = np.random.default_rng(seed)
    not_better = 0
    for _ in range(n):
        starts = rng.integers(0, size - block + 1, size=int(np.ceil(size / block)))
        rows = (starts[:, None] + np.arange(block)).ravel()[:size]
        sample = excess[rows]
        sharpe = sample.mean(axis=0) / sample.std(axis=0, ddof=1)
        not_better += sharpe[0] <= sharpe[1]
    return float(not_better / n)


def switches_per_year(w: pd.Series) -> float:
    w = w.dropna()
    return float(w.diff().abs().gt(0).sum() / (len(w) / 252)) if len(w) else np.nan


def switch_risk_table(gf: pd.DataFrame, state_col: str) -> pd.DataFrame:
    """Exclude terminal weeks lacking a full forward horizon, separately per horizon."""
    state = gf[state_col]
    shares = {}
    for horizon in (4, 13):
        flipped = pd.Series(False, index=gf.index)
        for offset in range(1, horizon + 1):
            flipped |= state.shift(-offset).ne(state)
        observed = pd.Series(np.arange(len(gf)) + horizon < len(gf), index=gf.index)
        shares[horizon] = flipped.astype(float).where(observed)
    rows = []
    for band in BANDS:
        mask = gf.band.eq(band)
        p4, p13 = shares[4][mask].mean(), shares[13][mask].mean()
        grade = ('low' if p4 < .25 else 'medium' if p4 <= .5 else 'high') if pd.notna(p4) else 'unavailable'
        rows.append({'band': band, 'decision_weeks': int(mask.sum()),
                     'observed_4w': int(shares[4][mask].count()),
                     'observed_13w': int(shares[13][mask].count()),
                     'share_4w': p4, 'share_13w': p13, 'grade': grade})
    return pd.DataFrame(rows)


def worst_drawdowns(r: pd.Series, k=5) -> pd.DataFrame:
    """Non-overlapping peak-to-recovery episodes; NaT marks unrecovered episodes.

    A loss from initial capital has an unobserved (NaT) pre-sample peak.
    """
    wealth = (1 + r.dropna()).cumprod()
    peak_value, peak_date = 1., pd.NaT
    episode = None
    episodes = []
    for date, value in wealth.items():
        if value >= peak_value:
            if episode is not None:
                episode['recovery'] = date
                episodes.append(episode)
                episode = None
            peak_value, peak_date = value, date
        else:
            depth = value / peak_value - 1
            if episode is None:
                episode = {'peak': peak_date, 'trough': date, 'recovery': pd.NaT, 'depth': depth}
            elif depth < episode['depth']:
                episode.update(trough=date, depth=depth)
    if episode is not None:
        episodes.append(episode)
    return pd.DataFrame(episodes, columns=['peak', 'trough', 'recovery', 'depth']).sort_values('depth').head(k).reset_index(drop=True)
