"""M: macro filter on the trend exit — pre-registered in spec 6M (commit 0a53e1c). One run.

M1 unemployment trend, M2 Baa-Aaa credit trend; each out when above its average of the previous 12 months.
Candidate: cash only if DMAS < 45 AND the macro key is out. Base: cash if DMAS < 45.
"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_c import sharpe
from research.us_gauge.evaluate import bootstrap_not_better
from research.us_gauge.run_fast import states, to_weights
from research.us_gauge.run_b import breadth, deemer_ratio, thrust_days
from research.us_gauge.revisions import detect_rolls, revisions, rev_zscore
from research.us_gauge.trading import net_returns
from research.us_gauge.core_slide_data import desired_from_states
from research.us_gauge.run_pos import skeleton_inputs, fast_act
from research.us_gauge.run_q import desired_for, returns_for, stats
from research.us_gauge import run_vol

HERE = Path(__file__).resolve().parent
START, SPLIT = '1955-01-03', '1990-12-31'
N_PLACEBO, MIN_SHIFT = 1000, 60
P_PASS = .05
NAMES = {'M1': 'M1 unemployment trend', 'M2': 'M2 credit trend (Baa − Aaa)'}


# ---------------------------------------------------------------- keys (spec 6M, fixed)
def macro_data() -> pd.DataFrame:
    with read_only_external():
        u, baa, aaa = (load(n, refresh=False).sort_index() for n in ('FRED_UNRATE', 'FRED_BAA', 'FRED_AAA'))
    return pd.DataFrame({'u': u, 's': baa - aaa})


def out_key(x: pd.Series) -> pd.Series:
    """Monthly: 1 = in, 0 = out (value above the mean of the previous 12 months); NaN while undefined."""
    prev = x.shift(1).rolling(12, min_periods=12).mean()
    return (x <= prev).astype(float).where(prev.notna() & x.notna())


def keys(m: pd.DataFrame) -> dict:
    return {'M1': out_key(m.u.dropna()), 'M2': out_key(m.s.dropna())}


def to_daily(mkey: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Month m (dated on its first day) usable from the first trading day on or after the 10th of month m+1."""
    k = mkey.dropna()
    usable = (k.index.to_period('M') + 1).to_timestamp() + pd.Timedelta(days=9)
    pos = index.searchsorted(usable)
    ok = pos < len(index)
    s = pd.Series(np.nan, index=index)
    s.iloc[pos[ok]] = k.to_numpy()[ok]
    return s.ffill()


# ---------------------------------------------------------------- placebo worker
G = {}


def init(df, dm, mkeys):
    G.update(df=df, dm=dm, mkeys=mkeys)


def placebo_one(args):
    name, shift = args
    df, dm = G['df'], G['dm']
    k = G['mkeys'][name].dropna()
    fake = pd.Series(np.roll(k.to_numpy(), shift), index=k.index)
    out = to_daily(fake, df.index).reindex(dm.index)
    r, _ = returns_for(df, desired_for(df, dm, (dm < 45) & (out == 0)))
    r = r.loc[START:].dropna()
    return sharpe(r, df.cash_ret.reindex(r.index))


# ---------------------------------------------------------------- product check (G5)
def product_check(mkeys):
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
        return stats(r, inputs.cash_ret)

    base = (dm < 45) & (rz < 0)
    out = {'base': run(base)}
    for name, k in mkeys.items():
        out[name] = run(base & (to_daily(k, idx).reindex(days) == 0))
    return out


# ---------------------------------------------------------------- information: base trend exits
def exits_table(df, w_base, daily):
    w = w_base.loc[START:]
    ex = w.index[((w == 0) & (w.shift(1) > 0)).to_numpy()]
    tr = (1 + df.eq_ret.fillna(0.)).cumprod()
    rows = []
    for d in ex:
        i = df.index.get_loc(d)
        f = lambda h: float(tr.iloc[min(i + h, len(tr) - 1)] / tr.iloc[i] - 1)
        rows.append(dict(date=d.date(), M1=daily['M1'].loc[d], M2=daily['M2'].loc[d], f63=f(63), f252=f(252)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main (one run)
def main():
    df = skeleton_inputs()
    dm = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0]
    dm = dm.loc[START:].reindex(df.index).dropna()
    df = df.loc[:dm.index[-1]]
    mkeys = keys(macro_data())
    daily = {n: to_daily(k, df.index) for n, k in mkeys.items()}
    for n, d in daily.items():
        assert d.reindex(dm.index).notna().all(), f'{n} key undefined inside the window'
    cash = df.cash_ret
    halves = [('1955–1990', None, SPLIT), ('1991–end', '1991-01-01', None)]

    r_base, w_base = returns_for(df, desired_for(df, dm, dm < 45))
    r_base = r_base.loc[START:]
    res = {}
    for n in NAMES:
        out = daily[n].reindex(dm.index)
        r, w = returns_for(df, desired_for(df, dm, (dm < 45) & (out == 0)))
        res[n] = dict(r=r.loc[START:], w=w, share_out=float((out == 0).mean()))

    rng = np.random.default_rng(7)
    jobs = []
    for n in NAMES:
        nm = int(mkeys[n].notna().sum())
        jobs += [(n, int(s)) for s in rng.integers(MIN_SHIFT, nm - MIN_SHIFT + 1, size=N_PLACEBO)]
    with Pool(10, initializer=init, initargs=(df, dm, mkeys)) as pool:
        pl = pool.map(placebo_one, jobs, chunksize=10)
    pl = {'M1': np.array(pl[:N_PLACEBO]), 'M2': np.array(pl[N_PLACEBO:])}

    prod = product_check(mkeys)
    sb, db, cb = stats(r_base, cash)
    sw = lambda w: float((w.loc[START:].diff().abs() > 1e-9).sum() / (len(w.loc[START:]) / 252))
    verdicts, body = {}, []
    for n, label in NAMES.items():
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
                 f'| Trades a year | {sw(w_base):.1f} | {sw(res[n]["w"]):.1f} |', f'| Share of days the key is "out" | | {res[n]["share_out"]:.0%} |', '']
    L = ['# M: macro filter on the trend exit (pre-registered, spec 6M, one run)', '',
         f'**M1: {verdicts["M1"]}. M2: {verdicts["M2"]}.** Consequence (declared before the run): PASS → proposed to the user; FAIL → rejected list, no change.', '',
         f'S&P 500 {START} → {dm.index[-1].date()}. Price skeleton as 6P/6Q, 10 bp, cash = T-bill. Candidate: cash only if DMAS < 45 AND the macro key is out. '
         'Monthly data usable from the 10th of the next month. Code: `run_macro.py`.', '']
    L += body
    E = exits_table(df, w_base, daily)
    both = pd.concat([daily['M1'], daily['M2']], axis=1).loc[START:]
    L += ['## Information only', '',
          f'Correlation of M1 and M2 states (daily): {both.corr().iloc[0, 1]:+.2f}.', '',
          '### Base trend exits (day the base goes to cash), by macro key on that day', '',
          '| Key | State | Exits | Mean next 63 days | Mean next 252 days | Share of exits followed by a lower S&P after 252 days |', '|---|---|---:|---:|---:|---:|']
    for n in NAMES:
        for st, lab in ((0., 'out (confirms the exit)'), (1., 'in (would stay invested)')):
            e = E[E[n] == st]
            if len(e):
                L.append(f'| {n} | {lab} | {len(e)} | {e.f63.mean():+.1%} | {e.f252.mean():+.1%} | {(e.f252 < 0).mean():.0%} |')
    L += ['', '| Exit | M1 | M2 | Next 63 days | Next 252 days |', '|---|---|---|---:|---:|']
    L += [f'| {r.date} | {"out" if r.M1 == 0 else "in"} | {"out" if r.M2 == 0 else "in"} | {r.f63:+.1%} | {r.f252:+.1%} |' for r in E.itertuples()]
    (HERE / 'RESULTS_macro.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    if '--one-run' not in sys.argv:
        sys.exit('Pre-registered single run: pass --one-run.')
    main()
