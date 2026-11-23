"""T: fast stress de-risking batch — pre-registered in spec 6T (commit 3971da6). One run.

T1 tranched month lock (4 sub-books on trading days 1/6/11/16); T2 bond/EM canary cap (Keller DAA);
T3 credit-confirmed cap (HY OAS 20-day jump above its trailing 90th percentile AND S&P below its 50-day average).
Base = product spec 11d.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_vol
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_c import sharpe, floor_weights
from research.us_gauge.run_p import episodes as f_episodes
from research.us_gauge.evaluate import bootstrap_not_better
from research.us_gauge.run_fast import states, to_weights
from research.us_gauge.run_s import shock_episodes
from research.us_gauge.run_b import breadth, deemer_ratio, thrust_days, thrust_episodes
from research.us_gauge.revisions import detect_rolls, revisions, rev_zscore
from research.us_gauge.trading import net_returns
from research.us_gauge.core_slide_data import layer, desired_from_states
from research.us_gauge.run_pos import fast_act
from research.us_gauge.run_q import stats

HERE = Path(__file__).resolve().parent
START = '1999-02-01'
OFFSETS = (1, 6, 11, 16)
N_PLACEBO, MIN_SHIFT, P_PASS = 1000, 252, .10 / 3
EPISODES = [('Dot-com bear', '2000-03-24', '2002-10-09'), ('2008 crisis', '2007-10-09', '2009-03-09'),
            ('2011 debt ceiling', '2011-04-29', '2011-10-03'), ('2015–16', '2015-07-20', '2016-02-11'),
            ('Q4 2018', '2018-09-20', '2018-12-24'), ('COVID fall', '2020-02-19', '2020-03-23'),
            ('2022', '2022-01-03', '2022-12-30'), ('Liberation Day fall', '2025-02-19', '2025-04-08'),
            ('Nov 2026 to date', '2026-11-01', None)]


# ---------------------------------------------------------------- T1: offset month lock
def trading_day_of_month(dates: pd.DatetimeIndex) -> np.ndarray:
    m = dates.to_period('M')
    return pd.Series(1, index=dates).groupby(m).cumsum().to_numpy()


def states_offset(dates, cash, deep, k: int) -> pd.Series:
    """As run_fast.states, but the locked period starts on the k-th trading day of each month."""
    tdm = trading_day_of_month(dates)
    out, cur, started = [], np.nan, False
    for i, d in enumerate(dates):
        if tdm[i] == k:
            cur, started = (0. if cash.loc[d] else 1.), True
        elif started and cur == 1. and cash.loc[d] and deep.loc[d]:
            cur = 0.
        out.append(cur if started else np.nan)
    return pd.Series(out, index=dates)


# ---------------------------------------------------------------- caps
def m13612(tr: pd.Series) -> pd.Series:
    r = lambda h: tr / tr.shift(h) - 1
    return 12 * r(21) + 4 * r(63) + 2 * r(126) + r(252)


def canary_cap(agg: pd.Series, eem: pd.Series, index) -> pd.Series:
    a, e = m13612(agg).reindex(index, method='ffill'), m13612(eem).reindex(index, method='ffill')
    bad = (a < 0).astype(int) + (e < 0).astype(int)
    return pd.Series(np.select([bad == 0, bad == 1], [1., .5], 0.), index=index).where(a.notna() & e.notna())


def credit_cap(oas: pd.Series, px: pd.Series) -> pd.Series:
    o = oas.reindex(px.index, method='ffill')
    d = o - o.shift(20)
    p90 = d.shift(1).rolling(1260, min_periods=504).quantile(.9)
    stress = (d > p90) & (px < px.rolling(50).mean())
    return pd.Series(np.where(stress, .5, 1.), index=px.index).where(p90.notna() & d.notna())


# ---------------------------------------------------------------- product parts
def product():
    with read_only_external():
        inputs = load_inputs()
    idx = inputs.index
    br = breadth()
    td = thrust_days(deemer_ratio(br.adv, br.dec))
    v = run_vol.target_weight(inputs.eq_ret)
    half = np.maximum(v.shift(2).fillna(0.), .5)
    full = pd.Series(1., index=idx)
    days = idx[idx >= '1993-04-01']
    dm = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0].reindex(days)
    rz = rev_zscore(revisions(inputs.fy1, inputs.fy2, detect_rolls(inputs.fy1, inputs.fy2), window=63)).reindex(days)
    cash_d, deep = (dm < 45) & (rz < 0), dm < 30
    s1 = states(days, cash_d, deep, dm, None)
    w1 = to_weights(idx, s1)
    epf = f_episodes(idx, s1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    wf = floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)
    ws = layer(idx, shock_episodes(inputs.px, w1), half)
    wb = layer(idx, thrust_episodes(inputs.px, w1, td), full)
    lay = pd.concat([wf.shift(-2), ws.shift(-2), wb.shift(-2)], axis=1).max(axis=1)
    st = s1.reindex(idx).ffill()
    base = pd.concat([st * v, lay], axis=1).max(axis=1)
    ref = desired_from_states(inputs, s1, v, half, td, full)
    assert np.allclose(base.fillna(-1), ref.fillna(-1)), 'base desired differs from the product'
    return dict(inputs=inputs, idx=idx, v=v, lay=lay, base=base, days=days, cash_d=cash_d, deep=deep, st=st)


def run(P, desired):
    inp = P['inputs']
    w = fast_act(P['idx'], (desired * 20).round() / 20)
    return net_returns(w, inp.eq_ret, inp.cash_ret).where(w.notna()), w


def capped(P, cap):
    return np.minimum(P['base'], cap.fillna(1.)).where(P['base'].notna())


def rolled(cap: pd.Series, shift: int) -> pd.Series:
    c = cap.dropna()
    return pd.Series(np.roll(c.to_numpy(), shift), index=c.index).reindex(cap.index)


# ---------------------------------------------------------------- main (one run)
def main():
    P = product()
    idx, inp = P['idx'], P['inputs']
    cash = inp.cash_ret
    # T1
    subs = [states_offset(P['days'], P['cash_d'], P['deep'], k) for k in OFFSETS]
    avg = pd.concat([s.reindex(idx).ffill() for s in subs], axis=1).mean(axis=1, skipna=False)
    d1 = pd.concat([avg * P['v'], P['lay']], axis=1).max(axis=1).where(avg.notna())
    # T2, T3
    with read_only_external():
        agg, eem, oas = (load(n, refresh=False).sort_index() for n in ('AGG.US_TR', 'EEM.US_TR', 'BBG_LF98OAS'))
    cap2, cap3 = canary_cap(agg, eem, idx), credit_cap(oas, inp.px)
    start2 = str(cap2.first_valid_index().date())
    cands = {'T1': (d1, START, '2012-12-31', None), 'T2': (capped(P, cap2), start2, '2015-06-30', cap2),
             'T3': (capped(P, cap3), START, '2012-12-31', cap3)}
    names = {'T1': 'T1 tranched month lock', 'T2': 'T2 bond / EM canary cap', 'T3': 'T3 credit-confirmed stress cap'}
    r_base, w_base = run(P, P['base'])
    sw = lambda w, a: float((w.loc[a:].diff().abs() > 1e-9).sum() / (len(w.loc[a:]) / 252))
    rng = np.random.default_rng(7)
    verdicts, body, rets, wts = {}, [], {'Base': r_base}, {'Base': w_base}
    for n, (des, a, split, cap) in cands.items():
        r, w = run(P, des)
        rets[n], wts[n] = r, w
        rb, rx = r_base.loc[a:], r.loc[a:]
        sb, db, cb = stats(rb, cash)
        sx, dx, cx = stats(rx, cash)
        halves = [(f'{a[:7]} → {split[:7]}', a, split), (f'after {split[:7]}', pd.Timestamp(split) + pd.Timedelta(days=1), None)]
        hs = [(h, stats(r_base.loc[x:y], cash)[0], stats(r.loc[x:y], cash)[0]) for h, x, y in halves]
        nb = bootstrap_not_better(rx.dropna(), rb.dropna(), cash)
        gates = [('G1 Sharpe better, full and both halves', f'full {sx:.3f} vs {sb:.3f}; ' + '; '.join(f'{h} {x:.3f} vs {b_:.3f}' for h, b_, x in hs),
                  sx > sb and all(x > b_ for _, b_, x in hs)),
                 ('G2 worst fall not deeper by > 2 pts', f'{dx:.1%} vs {db:.1%}', dx >= db - .02),
                 ('G3 bootstrap not-better share < 25%', f'{nb:.1%}', nb < .25)]
        if cap is not None:
            nd = int(cap.notna().sum())
            pl = np.array([stats(run(P, capped(P, rolled(cap, int(s))))[0].loc[a:], cash)[0]
                           for s in rng.integers(MIN_SHIFT, nd - MIN_SHIFT + 1, size=N_PLACEBO)])
            p = float((pl >= sx).mean())
            gates.append((f'G4 placebo p <= {P_PASS:.3f}', f'p = {p:.3f} (placebo median Sharpe {np.median(pl):.3f})', p <= P_PASS))
        verdicts[n] = 'PASS' if all(g[2] for g in gates) else 'FAIL'
        body += [f'## {names[n]}: **{verdicts[n]}**', '', '| Gate | Evidence | Result |', '|---|---|---|']
        body += [f'| {g} | {e} | {"PASS" if ok else "FAIL"} |' for g, e, ok in gates]
        extra = '' if cap is None else f'| Share of days capped (below 100%) | | {(cap.loc[a:] < 1).mean():.0%} |'
        body += ['', f'| {a} → {idx[-1].date()} | Product | Product + {n} |', '|---|---:|---:|',
                 f'| Sharpe | {sb:.2f} | {sx:.2f} |', f'| Return a year | {cb:.1%} | {cx:.1%} |', f'| Worst fall | {db:.1%} | {dx:.1%} |',
                 f'| Trades a year | {sw(w_base, a):.1f} | {sw(w, a):.1f} |'] + ([extra] if extra else []) + ['']
    L = ['# T: fast stress de-risking batch (pre-registered, spec 6T, one run)', '',
         '**' + '. '.join(f'{n}: {verdicts[n]}' for n in cands) + '.** Consequence (declared before the run): PASS → proposed to the user; FAIL → rejected list, no change.', '',
         f'Base = product spec 11d, inputs to {idx[-1].date()}. 10 bp, cash = T-bill. T2 window starts {start2} (first day both canaries are defined). Code: `run_t.py`.', '']
    L += body
    L += ['## Information only: stress episodes (total return / worst fall inside the window)', '',
          '| Episode | S&P | Product | T1 | T2 | T3 |', '|---|---:|---:|---:|---:|---:|']
    eq = inp.eq_ret
    for name, a, b in EPISODES:
        def cell(r):
            x = r.loc[a:b].dropna()
            if len(x) == 0:
                return 'n/a'
            eqc = (1 + x).cumprod()
            return f'{eqc.iloc[-1] - 1:+.1%} / {(eqc / eqc.cummax() - 1).min():.1%}'
        L.append(f'| {name} | {cell(eq)} | {cell(rets["Base"])} | {cell(rets["T1"])} | {cell(rets["T2"])} | {cell(rets["T3"])} |')
    L += ['', f'Caps on the last input day ({idx[-1].date()}): T2 = {cap2.iloc[-1]:.0%}, T3 = {cap3.iloc[-1]:.0%}.']
    (HERE / 'RESULTS_t.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    if '--one-run' not in sys.argv:
        sys.exit('Pre-registered single run: pass --one-run.')
    main()
