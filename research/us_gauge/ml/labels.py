"""Step 7 (spec section 3): labels. Only run_m.py --one-run calls this on real data.

y = 1 if any close in t+1 .. t+126 is below L (stopped), else 0.
R = P(x+1) / P(t+1) - 1, x = first stop day or t+126 (trade in at the close of t+1, out at the close of x+1).
Open (no label): the 126-day window is not complete at the data end, or x+1 is past the data end.
"""
import numpy as np
import pandas as pd

HORIZON = 126


def label_one(p: np.ndarray, i_t: int, L: float) -> dict:
    n = len(p)
    if i_t + HORIZON > n - 1:
        return dict(y=np.nan, i_x=-1, R=np.nan, open=True)
    seg = p[i_t + 1:i_t + HORIZON + 1]
    below = np.flatnonzero(seg < L)
    y, i_x = (1, i_t + 1 + int(below[0])) if len(below) else (0, i_t + HORIZON)
    if i_x + 1 > n - 1:
        return dict(y=np.nan, i_x=-1, R=np.nan, open=True)
    return dict(y=float(y), i_x=i_x, R=float(p[i_x + 1] / p[i_t + 1] - 1), open=False)


def label_events(P: dict, E: pd.DataFrame) -> pd.DataFrame:
    """E: market, i_t, L (as events.csv). Returns y, R, i_x, x (date), open, aligned on E.index."""
    rows = []
    for r in E.itertuples():
        px = P[r.market]
        d = label_one(px.to_numpy(dtype=float), int(r.i_t), float(r.L))
        d['x'] = px.index[d['i_x']] if d['i_x'] >= 0 else pd.NaT
        rows.append(d)
    return pd.DataFrame(rows, index=E.index)
