"""Stress test on synthetic US paths — method fixed in spec 6ST (commit 219423a). Information only.

Stationary block bootstrap of S&P daily returns 1928-2026; price-only core rules on each path.
"""
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from research.us_gauge import run_vol
from research.us_gauge.data import load
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.gauge import dmas_on
from research.us_gauge.run_p import decision_days
from research.us_gauge.run_fast import states, to_weights, v2_from_states
from research.us_gauge.run_s import shock_episodes
from research.us_gauge.trading import net_returns

HERE = Path(__file__).resolve().parent
N_DAYS, WARM, MEAN_BLOCK = 7560, 300, 63
CRISIS_YEARS = {1929, 1930, 1931, 1932, 1937, 1938, 1940, 1941, 1942, 1946, 1962, 1966, 1969, 1970,
                1973, 1974, 1987, 2000, 2001, 2002, 2007, 2008, 2009, 2020, 2022}
RULES = ['Buy & hold', 'R0 (trend only)', 'V = R0 x vol size', 'V + S live', 'V + S 50% floor (product)', 'V + S 100%']


def history():
    with read_only_external():
        px = load('GSPC.INDX', refresh=False).sort_index().astype(float)
        div = load('SHILLER_SPX_DIV', refresh=False).sort_index()
    px = px[px.notna() & (px > 0)]
    px = px[~px.index.duplicated()]
    pr = px.pct_change(fill_method=None).iloc[1:]
    dy = (div.reindex(px.index, method='ffill') / px / 252).reindex(pr.index).fillna(0.)
    return pr.to_numpy(), dy.to_numpy(), pr.index.year.to_numpy()


def bootstrap(rng, pr, dy, starts):
    n, out = N_DAYS + WARM, []
    while len(out) < n:
        s = int(rng.choice(starts))
        L = int(rng.geometric(1 / MEAN_BLOCK))
        out.extend(range(s, min(s + L, len(pr))))
    k = np.array(out[:n])
    return pr[k], dy[k]


def stats(r):
    r = r.dropna()
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    eq = (1 + r).cumprod()
    return sh, float((eq / eq.cummax() - 1).min()), float(eq.iloc[-1] ** (252 / len(r)) - 1)


def one_path(args):
    seed, world = args
    pr, dy, yrs = HIST
    starts = np.arange(len(pr)) if world == 'A' else np.flatnonzero(np.isin(yrs, list(CRISIS_YEARS)))
    rng = np.random.default_rng(seed)
    p_ret, d_ret = bootstrap(rng, pr, dy, starts)
    idx = pd.bdate_range('2000-01-03', periods=len(p_ret))
    px = pd.Series(100 * np.cumprod(1 + p_ret), index=idx)
    df = pd.DataFrame({'px': px, 'eq_ret': pd.Series(p_ret + d_ret, index=idx), 'cash_ret': 0.})
    d = decision_days(idx, str(idx[WARM].date()), 'W-TUE')
    m = dmas_on(px, d)
    st = states(d, m < 45, m < 30, m, None)
    wb = to_weights(idx, st)
    wv = v2_from_states(df, st).reindex(idx)
    ep = shock_episodes(px, wb)
    v = run_vol.target_weight(df.eq_ret).reindex(idx).shift(2).fillna(0.)
    vv = v.to_numpy()
    assert len(vv) == len(idx)
    lay = {k: np.zeros(len(idx)) for k in ('live', 'half', 'full')}
    for r in ep.itertuples():
        a, b = int(r.i_entry) + 2, min(int(r.i_end) + 2, len(idx))
        if a >= b:
            continue
        lay['live'][a:b] = vv[a:b]
        lay['half'][a:b] = np.maximum(vv[a:b], .5)
        lay['full'][a:b] = 1.
    lay = {k: pd.Series(x, index=idx) for k, x in lay.items()}
    R = lambda w: net_returns(w, df.eq_ret, df.cash_ret).where(w.notna())
    first = wv.first_valid_index()
    rets = {'Buy & hold': df.eq_ret, 'R0 (trend only)': R(wb), 'V = R0 x vol size': R(wv),
            'V + S live': R(np.maximum(wv, lay['live']).where(wv.notna())),
            'V + S 50% floor (product)': R(np.maximum(wv, lay['half']).where(wv.notna())),
            'V + S 100%': R(np.maximum(wv, lay['full']).where(wv.notna()))}
    out = {'seed': seed, 'world': world}
    for k, r in rets.items():
        out[k] = stats(r.loc[first:])
    ep_ret = [float((1 + df.eq_ret.iloc[r.i_entry + 2:r.i_end + 2]).prod() - 1) for r in ep.itertuples()]
    out['n_ep'], out['n_stop'] = len(ep), int((ep.reason == 'stop').sum()) if len(ep) else 0
    out['worst_ep'] = min(ep_ret) if ep_ret else np.nan
    out['bh_dd'] = out['Buy & hold'][1]
    return out


def safe_path(args):
    try:
        return one_path(args)
    except Exception as e:
        return {'seed': args[0], 'world': args[1], 'error': f'{type(e).__name__}: {e}'}


def init(h):
    global HIST
    HIST = h


def summarise(res, world):
    rows = [r for r in res if r['world'] == world]
    L = [f'## World {world} ({"blocks from any year" if world == "A" else "blocks from crisis years only"}), {len(rows)} paths of 30 years', '',
         '| Rule | Sharpe median | Sharpe bad case (5%) | Worst fall median | Worst fall bad case (5%) | Single worst | Paths < −30% | Paths < −40% | Beats B&H Sharpe |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    bh = np.array([r['Buy & hold'][0] for r in rows])
    for k in RULES:
        sh = np.array([r[k][0] for r in rows]); dd = np.array([r[k][1] for r in rows])
        L.append(f'| {k} | {np.nanmedian(sh):.2f} | {np.nanpercentile(sh, 5):.2f} | {np.median(dd):.1%} | {np.percentile(dd, 5):.1%} | {dd.min():.1%} | '
                 f'{(dd < -.30).mean():.0%} | {(dd < -.40).mean():.0%} | {"" if k == "Buy & hold" else f"{(sh > bh).mean():.0%}"} |')
    d50 = np.array([r['V + S 50% floor (product)'][0] - r['V + S live'][0] for r in rows])
    d100 = np.array([r['V + S 100%'][0] - r['V + S 50% floor (product)'][0] for r in rows])
    dd100 = np.array([r['V + S 100%'][1] - r['V + S 50% floor (product)'][1] for r in rows])
    ne = np.array([r['n_ep'] for r in rows]); ns = np.array([r['n_stop'] for r in rows])
    we = np.array([r['worst_ep'] for r in rows])
    L += ['', f'- S episodes per path: median {np.median(ne):.0f}; share stopped {ns.sum() / max(ne.sum(), 1):.0%}; worst single episode (index) median {np.nanmedian(we):+.1%}, bad case (5%) {np.nanpercentile(we, 5):+.1%}.',
          f'- 50% floor vs live: Sharpe better in {(d50 > 0).mean():.0%} of paths (median {np.median(d50):+.3f}).',
          f'- 100% vs 50% floor: Sharpe better in {(d100 > 0).mean():.0%} of paths (median {np.median(d100):+.3f}); worst fall deeper in {(dd100 < -1e-9).mean():.0%} of paths, bad case (5%) {np.percentile(dd100, 5):+.1%}.', '']
    return L


def main():
    h = history()
    jobs = [(7_000 + i, 'A') for i in range(1000)] + [(9_000 + i, 'B') for i in range(500)]
    with Pool(12, initializer=init, initargs=(h,)) as pool:
        res = pool.map(safe_path, jobs, chunksize=8)
    errs = [r for r in res if 'error' in r]
    res = [r for r in res if 'error' not in r]
    pd.DataFrame([{**{k: v for k, v in r.items() if not isinstance(v, tuple)},
                   **{f'{k}|{m}': r[k][j] for k in RULES for j, m in enumerate(('sharpe', 'maxdd', 'cagr'))}} for r in res]
                 ).to_csv(HERE / 'out/stress_synth_paths.csv', index=False)
    L = ['# Stress test on synthetic US paths (spec 6ST, information only)', '',
         'Stationary block bootstrap (mean block 63 days) of S&P 500 daily returns 1928–2026. Price-only rules: no earnings key, '
         'no VIX floor, no breadth re-entry, Tuesday decisions with the old band rule, cash 0%, 10 bp. So these numbers are for the '
         'price-driven skeleton of the core, not the full product. Code: `stress_synth.py`.', '']
    L.append(f'Paths that failed to compute and were skipped: {len(errs)}' + (f' (first: {errs[0]["error"][:120]})' if errs else '') + '.')
    L.append('')
    for w in ('A', 'B'):
        L += summarise(res, w)
    (HERE / 'RESULTS_stress_synth.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
