"""Q: real money flows as a key (Fed Z.1) — pre-registered in spec 6Q (commit 91d940d). One run.

Q1 buyback yield (in when z >= 0), Q2 equity fund + ETF flows (in when z <= 0).
Candidate: cash only if DMAS < 45 AND the flow key is out. Base: cash if DMAS < 45.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_vol
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_c import sharpe
from research.us_gauge.evaluate import metrics, bootstrap_not_better
from research.us_gauge.run_fast import states, to_weights
from research.us_gauge.run_s import shock_episodes
from research.us_gauge.run_b import breadth, deemer_ratio, thrust_days
from research.us_gauge.revisions import detect_rolls, revisions, rev_zscore
from research.us_gauge.trading import net_returns
from research.us_gauge.core_slide_data import layer, desired_from_states
from research.us_gauge.run_pos import skeleton_inputs, fast_act

HERE = Path(__file__).resolve().parent
START, SPLIT = '1956-10-29', '1990-12-31'   # spec 6Q pre-run note: first day both keys are usable
LAG_DAYS, ZWIN, ZMIN = 120, 40, 20
N_PLACEBO = 1000
P_PASS = .05


# ---------------------------------------------------------------- signals (spec 6Q, fixed)
def flow_data() -> dict:
    with read_only_external():
        return {k: load(f'Z1_{k}', refresh=False).sort_index() for k in
                ('FA103164105', 'FA653064100', 'FA563064100', 'LM883164115', 'LM893064105')}


def signals(z1: dict) -> pd.DataFrame:
    q = pd.DataFrame(z1)
    m = lambda s: s.rolling(4, min_periods=4).mean()
    q1 = -m(q.FA103164105) / q.LM883164115
    q2 = m(q.FA653064100 + q.FA563064100.fillna(0.)) / q.LM893064105
    return pd.DataFrame({'Q1': q1, 'Q2': q2})


def trailing_z(s: pd.Series, n=ZWIN, n_min=ZMIN) -> pd.Series:
    """z-score against the previous n values (current excluded), at least n_min of them."""
    prev = s.shift(1).rolling(n, min_periods=n_min)
    return (s - prev.mean()) / prev.std(ddof=1)


def key_in(z: pd.Series, name: str) -> pd.Series:
    """Quarterly in/out key (1/0), NaN where z is undefined."""
    k = (z >= 0) if name == 'Q1' else (z <= 0)
    return k.astype(float).where(z.notna())


def to_daily(qkey: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Quarter q usable from the first trading day on or after quarter end + 120 calendar days."""
    k = qkey.dropna()
    pos = index.searchsorted(k.index + pd.Timedelta(days=LAG_DAYS))
    ok = pos < len(index)
    s = pd.Series(np.nan, index=index)
    s.iloc[pos[ok]] = k.to_numpy()[ok]
    return s.ffill()


# ---------------------------------------------------------------- skeleton rule (as 6P), key-generic
def desired_for(df, dm, cash_d):
    idx = df.index
    s1 = states(dm.index, cash_d, dm < 30, dm, None)
    w1 = to_weights(idx, s1)
    v = run_vol.target_weight(df.eq_ret)
    half = np.maximum(v.shift(2).fillna(0.), .5)
    ws = layer(idx, shock_episodes(df.px, w1), half)
    st = s1.reindex(idx).ffill()
    return pd.concat([st * v, ws.shift(-2)], axis=1).max(axis=1).where(st.notna())


def returns_for(df, desired):
    w = fast_act(df.index, (desired * 20).round() / 20)
    return net_returns(w, df.eq_ret, df.cash_ret).where(w.notna()), w


# ---------------------------------------------------------------- placebo worker
G = {}


def init(df, dm, qkeys):
    G.update(df=df, dm=dm, qkeys=qkeys)


def placebo_one(args):
    name, shift = args
    df, dm = G['df'], G['dm']
    k = G['qkeys'][name].dropna()
    fake = pd.Series(np.roll(k.to_numpy(), shift), index=k.index)
    out = to_daily(fake, df.index).reindex(dm.index)
    r, _ = returns_for(df, desired_for(df, dm, (dm < 45) & (out == 0)))
    r = r.loc[START:].dropna()
    return sharpe(r, df.cash_ret.reindex(r.index))


# ---------------------------------------------------------------- product check (G5)
def product_check(qkeys):
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
    sl = slice('1999-02-01', None)

    def run(cash_d):
        s1 = states(days, cash_d, dm < 30, dm, None)
        d = desired_from_states(inputs, s1, v, half, td, full)
        w = fast_act(idx, (d * 20).round() / 20)
        r = net_returns(w, inputs.eq_ret, inputs.cash_ret).loc[sl].dropna()
        return sharpe(r, inputs.cash_ret.reindex(r.index)), metrics(r, inputs.cash_ret.reindex(r.index))['maxdd']

    base = (dm < 45) & (rz < 0)
    out = {'base': run(base)}
    for name, k in qkeys.items():
        out[name] = run(base & (to_daily(k, idx).reindex(days) == 0))
    return out


# ---------------------------------------------------------------- information tables
def quintile_table(df, zd: pd.Series, a=None, b=None):
    tr = (1 + df.eq_ret.fillna(0.)).cumprod()
    T = pd.DataFrame({'z': zd, 'f1': tr.shift(-252) / tr - 1, 'f3': tr.shift(-756) / tr - 1}).loc[START:].loc[a:b]
    T = T.dropna(subset=['z'])
    T['q'] = pd.qcut(T.z, 5, labels=['1 low', '2', '3', '4', '5 high'])
    return T.groupby('q', observed=True).agg(days=('z', 'size'), z_mean=('z', 'mean'), f1=('f1', 'mean'), f3=('f3', 'mean'))


def stats(r, cash):
    r = r.dropna()
    c = cash.reindex(r.index)
    m = metrics(r, c)
    return sharpe(r, c), m['maxdd'], m['cagr']


# ---------------------------------------------------------------- main (one run)
def main():
    df = skeleton_inputs()
    dm = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0]
    dm = dm.loc[START:].reindex(df.index).dropna()
    df = df.loc[:dm.index[-1]]
    sig = signals(flow_data())
    z = sig.apply(trailing_z)
    qkeys = {n: key_in(z[n], n) for n in ('Q1', 'Q2')}
    daily = {n: to_daily(k, df.index) for n, k in qkeys.items()}
    for n, d in daily.items():
        assert d.reindex(dm.index).notna().all(), f'{n} key undefined inside the window'
    cash = df.cash_ret
    halves = [('1955–1990', None, SPLIT), ('1991–end', '1991-01-01', None)]

    r_base, w_base = returns_for(df, desired_for(df, dm, dm < 45))
    r_base = r_base.loc[START:]
    res = {}
    for n in ('Q1', 'Q2'):
        out = daily[n].reindex(dm.index)
        r, w = returns_for(df, desired_for(df, dm, (dm < 45) & (out == 0)))
        res[n] = dict(r=r.loc[START:], w=w, share_in=float(out.mean()))

    rng = np.random.default_rng(7)
    jobs = []
    for n in ('Q1', 'Q2'):
        nq = int(qkeys[n].notna().sum())
        jobs += [(n, int(s)) for s in rng.integers(20, nq - 20 + 1, size=N_PLACEBO)]
    with Pool(10, initializer=init, initargs=(df, dm, qkeys)) as pool:
        pl = pool.map(placebo_one, jobs, chunksize=10)
    pl = {'Q1': np.array(pl[:N_PLACEBO]), 'Q2': np.array(pl[N_PLACEBO:])}

    prod = product_check(qkeys)

    sb, db, cb = stats(r_base, cash)
    sw = lambda w: float((w.loc[START:].diff().abs() > 1e-9).sum() / (len(w.loc[START:]) / 252))
    L = ['# Q: real money flows as a key, Fed Z.1 (pre-registered, spec 6Q, one run)', '']
    verdicts, body = {}, []
    for n, label in (('Q1', 'Q1 buyback yield (follow the companies)'), ('Q2', 'Q2 equity fund + ETF flows (fade the crowd)')):
        r = res[n]['r']
        sx, dx, cx = stats(r, cash)
        hs = [(h, stats(r_base.loc[a:b], cash)[0], stats(r.loc[a:b], cash)[0]) for h, a, b in halves]
        nb = bootstrap_not_better(r.dropna(), r_base.dropna(), cash)
        p = float((pl[n] >= sx).mean())
        g5b, g5x = prod['base'], prod[n]
        gates = [('G1 Sharpe better, full and both halves', f'full {sx:.3f} vs {sb:.3f}; ' + '; '.join(f'{h} {x:.3f} vs {b_:.3f}' for h, b_, x in hs),
                  sx > sb and all(x > b_ for _, b_, x in hs)),
                 ('G2 worst fall not deeper by > 2 pts', f'{dx:.1%} vs {db:.1%}', dx >= db - .02),
                 ('G3 bootstrap not-better share < 25%', f'{nb:.1%}', nb < .25),
                 (f'G4 placebo p <= {P_PASS}', f'p = {p:.3f} (placebo median Sharpe {np.median(pl[n]):.3f})', p <= P_PASS),
                 ('G5 real product (1999-02 → end) not worse', f'{g5x[0]:.3f} vs {g5b[0]:.3f} (worst fall {g5x[1]:.1%} vs {g5b[1]:.1%})', g5x[0] >= g5b[0])]
        verdicts[n] = 'PASS' if all(g[2] for g in gates) else 'FAIL'
        body += [f'## {label}: **{verdicts[n]}**', '', '| Gate | Evidence | Result |', '|---|---|---|']
        body += [f'| {g} | {e} | {"PASS" if ok else "FAIL"} |' for g, e, ok in gates]
        body += ['', f'| {r_base.dropna().index[0].date()} → {r_base.dropna().index[-1].date()} | Base | Base + {n} |', '|---|---:|---:|',
                 f'| Sharpe | {sb:.2f} | {sx:.2f} |', f'| Return a year | {cb:.1%} | {cx:.1%} |', f'| Worst fall | {db:.1%} | {dx:.1%} |',
                 f'| Trades a year | {sw(w_base):.1f} | {sw(res[n]["w"]):.1f} |', f'| Share of days the flow key is "in" | | {res[n]["share_in"]:.0%} |', '']
    L += [f'**Q1: {verdicts["Q1"]}. Q2: {verdicts["Q2"]}.** Consequence (declared before the run): PASS → proposed to the user as a key; FAIL → rejected list, no change.', '',
          f'S&P 500 {START} → {dm.index[-1].date()}. Price skeleton as 6P (DMAS keys daily with month lock, vol size, S at max(live, 50%), 5-point sizes, '
          'trade next day if gap ≥ 10 points), 10 bp, cash = T-bill. Candidate: cash only if DMAS < 45 AND the flow key is out; the DMAS < 30 mid-month exit '
          'uses the same combined condition, as in the product (`run_fast.states`). Flow data: Fed Z.1, current vintage, usable 120 days after quarter end. Code: `run_q.py`.', '']
    L += body
    L += ['## Information only', '', f'Correlation of Q1 and Q2 z-scores (quarterly): {z.Q1.corr(z.Q2):+.2f}.', '']
    for n in ('Q1', 'Q2'):
        zd = to_daily(z[n], df.index)
        for h, a, b in [('full', None, None)] + halves:
            T = quintile_table(df, zd, a, b)
            L += [f'### {n} z quintile → next S&P total return ({h})', '', '| Quintile | Days | Mean z | Next 1 year | Next 3 years |', '|---|---:|---:|---:|---:|']
            L += [f'| {q} | {r.days:,} | {r.z_mean:+.2f} | {r.f1:+.1%} | {r.f3:+.1%} |' for q, r in T.iterrows()]
            L += ['']
    L += ['Forward windows overlap; day counts overstate the independent evidence (about 70 years ≈ 70 independent 1-year periods).']
    (HERE / 'RESULTS_q.md').write_text('\n'.join(L) + '\n')
    pd.concat([sig, z.add_suffix('_z')], axis=1).to_csv(HERE / 'out/q_signals.csv')
    print('\n'.join(L))


if __name__ == '__main__':
    if '--one-run' not in sys.argv:
        sys.exit('Pre-registered single run: pass --one-run.')
    main()
