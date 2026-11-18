"""Step 2 (spec 2.1, 2.3, 2.4): S-type price events in 21 markets, cluster ids, base_cash, product episodes.

Look-ahead discipline. The state machine is run causally, one close at a time. Whether a new trigger at
close k is admitted depends on whether the previous event in that market is still open at k, i.e. on
closes up to k only. The end date or end reason of an event is never stored, returned or printed, and no
return after a trigger is computed here. Labels exist only in labels.py (step 7).
"""
import sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from research.us_gauge.ml.panel import (OUT, MARKETS, SEEN, REGION, DATA_END, load_panel)

DEPTH, SPEED, BOUNCE, HORIZON = .15, 63, 1.10, 126
START = pd.Timestamp('1965-01-01')
HIST = 756                                    # closes needed before h (run-up feature)
EMBARGO = HORIZON + 63                        # spec 6.1: label window + 63 trading days


def high_index(p: np.ndarray) -> np.ndarray:
    """h_t: index of the highest close in the 252 closes ending t (-1 before 252 closes). First max on ties."""
    n = len(p)
    hi = np.full(n, -1)
    if n >= 252:
        hi[251:] = np.arange(n - 251) + sliding_window_view(p, 252).argmax(axis=1)
    return hi


def _scan(p: np.ndarray, base_w=None):
    """Causal scan. base_w None = spec 2.1 events (no base gating; close on first close < L or t + 126).
    base_w given = run_s.shock_episodes entries (trigger only if base weight < 1; also closes when base invested)."""
    n = len(p)
    hi = high_index(p)
    out = []
    last_h, last_L = -1, np.inf
    active = None                              # (i_t, L) of the open event
    for k in range(252, n):
        if active is not None:
            t0, L0 = active
            closed = (base_w is not None and base_w[k] >= 1) or p[k] < L0 or k == t0 + HORIZON
            if closed:
                active = None
            continue                           # no trigger while open, nor on the closing day (as run_s)
        h = hi[k]
        l = h + int(p[h:k + 1].argmin())
        L = p[l]
        if h != last_h:
            last_h, last_L = h, np.inf
        gate = True if base_w is None else (not np.isnan(base_w[k]) and base_w[k] < 1)
        if L <= (1 - DEPTH) * p[h] and l - h <= SPEED and p[k] >= BOUNCE * L and L < last_L and gate:
            out.append((h, l, k))
            last_L = L
            active = (k, L)
    return out


def scan_events(px: pd.Series) -> pd.DataFrame:
    p = px.to_numpy(dtype=float)
    rows = [dict(i_h=h, i_l=l, i_t=k, h=px.index[h], l=px.index[l], t=px.index[k], H=p[h], L=p[l], P_t=p[k])
            for h, l, k in _scan(p)]
    return pd.DataFrame(rows, columns=['i_h', 'i_l', 'i_t', 'h', 'l', 't', 'H', 'L', 'P_t'])


def product_entries(px: pd.Series, base_w: pd.Series) -> pd.DataFrame:
    """Entries of run_s.shock_episodes(px, base_w), computed causally (no end, no reason)."""
    p = px.to_numpy(dtype=float)
    bw = base_w.reindex(px.index).to_numpy(dtype=float)
    rows = [dict(i_h=h, i_l=l, i_t=k, h=px.index[h], l=px.index[l], t=px.index[k], H=p[h], L=p[l], P_t=p[k])
            for h, l, k in _scan(p, bw)]
    return pd.DataFrame(rows, columns=['i_h', 'i_l', 'i_t', 'h', 'l', 't', 'H', 'L', 'P_t'])


def cluster_ids(dates, gap=45) -> np.ndarray:
    """Spec 2.4: sorted by entry date; an event joins the current cluster if its entry is within `gap`
    calendar days of the previous entry in that cluster. Returned in the input order."""
    d = pd.to_datetime(pd.Series(list(dates)))
    order = np.argsort(d.to_numpy(), kind='stable')
    ids = np.empty(len(d), dtype=int)
    c, prev = -1, None
    for i in order:
        if prev is None or (d.iloc[i] - prev).days > gap:
            c += 1
        ids[i] = c
        prev = d.iloc[i]
    return ids


def plain_trend_weight(px: pd.Series, dmas=None) -> pd.Series:
    """Plain trend rule of spec 6L / run_l.run_market: DMAS on Tuesdays from 252 closes, cash below 45, month
    lock, mid-month exit below 30; weight in force each day (decide t, in force t+2). Uses closes up to each
    decision day only."""
    from research.us_gauge.gauge import dmas_on
    from research.us_gauge.run_p import decision_days
    from research.us_gauge.run_fast import states, to_weights
    idx = px.index
    d = decision_days(idx, str(idx[252].date()), 'W-TUE')
    m = dmas_on(px, d) if dmas is None else dmas.reindex(d)
    st = states(d, m < 45, m < 30, m, None)
    return to_weights(idx, st)


def dmas_cached(market, px):
    """DMAS on the Tuesday decision days, cached in ml/out/cache (a pure function of closes up to each day)."""
    from research.us_gauge.gauge import dmas_on
    from research.us_gauge.run_p import decision_days
    f = OUT / 'cache' / f'dmas_{market}_{px.index[-1]:%Y%m%d}.csv'
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
    f.parent.mkdir(parents=True, exist_ok=True)
    m = dmas_on(px, decision_days(px.index, str(px.index[252].date()), 'W-TUE'))
    m.to_csv(f)
    return m


def build(P=None):
    """All events of all markets with flags; product-episode entries. No outcome is computed."""
    if P is None:
        P, _ = load_panel()
    E, PE = [], []
    for m in MARKETS:
        px = P[m]
        ev = scan_events(px)
        wb = plain_trend_weight(px, dmas_cached(m, px))
        ev['base_cash'] = [float(wb.iloc[i] < 1) if not np.isnan(wb.iloc[i]) else np.nan for i in ev.i_t]
        n = len(px)
        # date of the close 189 trading days after t (calendar only, no price): used for the embargo assertion
        ev['t_embargo'] = [px.index[i + EMBARGO] if i + EMBARGO < n else pd.NaT for i in ev.i_t]
        ev['open'] = ev.i_t + HORIZON > n - 1                # 126-day window not complete at the data end
        ev.insert(0, 'market', m)
        E.append(ev)
        pe = product_entries(px, wb)
        pe.insert(0, 'market', m)
        PE.append(pe)
    E = pd.concat(E, ignore_index=True)
    E['region'] = E.market.map(REGION)
    E['seen'] = E.market.isin(SEEN)
    E['from_1965'] = E.t >= START
    E['hist_ok'] = E.i_h >= HIST
    E['study'] = E.from_1965 & E.hist_ok
    E['us_early'] = (E.market == 'GSPC') & (E.t < START)
    E['cluster45_pre'] = -1
    E['cluster21_pre'] = -1
    pre = E.from_1965
    E.loc[pre, 'cluster45_pre'] = cluster_ids(E.loc[pre, 't'], 45)
    E.loc[pre, 'cluster21_pre'] = cluster_ids(E.loc[pre, 't'], 21)
    E['cluster'] = -1
    E['cluster21'] = -1
    st = E.study
    E.loc[st, 'cluster'] = cluster_ids(E.loc[st, 't'], 45)
    E.loc[st, 'cluster21'] = cluster_ids(E.loc[st, 't'], 21)
    E = E.sort_values(['t', 'market']).reset_index(drop=True)
    E.insert(0, 'event_id', [f'{r.market}_{r.t:%Y%m%d}' for r in E.itertuples()])
    PE = pd.concat(PE, ignore_index=True).sort_values(['t', 'market']).reset_index(drop=True)
    PE.insert(0, 'episode_id', [f'{r.market}_{r.t:%Y%m%d}' for r in PE.itertuples()])
    return E, PE


def count_table(E: pd.DataFrame, PE: pd.DataFrame) -> list:
    def row(name, d, c45=None, c21=None):
        c45 = len(set(cluster_ids(d.t, 45))) if c45 is None else c45
        c21 = len(set(cluster_ids(d.t, 21))) if c21 is None else c21
        return f'| {name} | {len(d)} | {c45} | {c21} | {d.t.dt.year.nunique()} |'
    pre = E[E.from_1965]
    st = E[E.study]
    L = ['| Set | Events | Clusters (45 d) | Clusters (21 d) | Calendar years |', '|---|---:|---:|---:|---:|']
    for lab, X in (('before the history filter', pre), ('after the history filter (study set)', st)):
        L += [row(f'All 1965–2026, {lab}', X),
              row(f'Before 2000, {lab}', X[X.t < '2000-01-01']),
              row(f'2000–2026, {lab}', X[X.t >= '2000-01-01']),
              row(f'2016–2026, {lab}', X[X.t >= '2016-01-01']),
              row(f'13 never-opened markets 2000–2026, {lab}', X[~X.seen & (X.t >= '2000-01-01')]),
              row(f'US, {lab}', X[X.market == 'GSPC']), row(f'US 2000+, {lab}', X[(X.market == 'GSPC') & (X.t >= '2000-01-01')])]
    L += [row('Product episodes, all dates', PE), row('Product episodes, from 1965', PE[PE.t >= START]),
          row('Product episodes, from 1970', PE[PE.t >= '1970-01-01']), row('Product episodes, 2000+', PE[PE.t >= '2000-01-01'])]
    return L


def main():
    P, B = load_panel()
    E, PE = build(P)
    OUT.mkdir(exist_ok=True)
    E.to_csv(OUT / 'events.csv', index=False, date_format='%Y-%m-%d')
    PE.to_csv(OUT / 'product_episodes.csv', index=False, date_format='%Y-%m-%d')
    print('\n'.join(count_table(E, PE)))
    print('\nPer market (1965+, before filter):', E[E.from_1965].market.value_counts().to_dict())
    print('Per market (study set):', E[E.study].market.value_counts().to_dict())
    print('Region (1965+):', E[E.from_1965].region.value_counts().to_dict())
    print('US trigger dates 1965+:', [str(d.date()) for d in E[(E.market == 'GSPC') & E.from_1965].t])
    print('US in study set:', [str(d.date()) for d in E[(E.market == 'GSPC') & E.study].t])
    print('Open (no label possible):', E[E.study & E.open].event_id.tolist())
    print('Study: labelled-eligible', int((E.study & ~E.open).sum()), 'clusters', E.loc[E.study & ~E.open, 'cluster'].nunique())


if __name__ == '__main__':
    main()
