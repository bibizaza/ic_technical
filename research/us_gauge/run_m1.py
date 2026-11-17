"""M1 — macro-only sector rank (pre-registered in the sleeve-tilts spec, commit 0346b60)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from research.us_gauge import run_k1, run_s2
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import MACRO, read_only_external, reproduces, monthly_metrics
from research.us_gauge.evaluate import metrics, bootstrap_not_better

HERE = Path(__file__).resolve().parent


def split_pred(X: pd.DataFrame, y: pd.DataFrame, lag: int):
    """sectors.trend_pred, unchanged fit, returned as (intercept part, macro part)."""
    Xv, Y = X.values, y.values
    a = np.full(Y.shape, np.nan)
    m = np.full(Y.shape, np.nan)
    for k in range(len(Y)):
        if np.isnan(Xv[k]).any():
            continue
        rows = np.arange(0, max(k - lag + 1, 0))
        for i in range(Y.shape[1]):
            ok = rows[~np.isnan(Y[rows, i]) & ~np.isnan(Xv[rows]).any(axis=1)]
            if len(ok) < 60:
                continue
            b = np.linalg.lstsq(np.column_stack([np.ones(len(ok)), Xv[ok]]), Y[ok, i], rcond=None)[0]
            a[k, i], m[k, i] = b[0], Xv[k] @ b[1:]
    f = lambda v: pd.DataFrame(v, index=y.index, columns=y.columns)
    return f(a), f(m)


def build():
    with read_only_external():
        sys.path.insert(0, str(MACRO))
        try:
            import tools
            if str(MACRO / 'tools') not in tools.__path__:
                tools.__path__.append(str(MACRO / 'tools'))
            from tools.regime_market import data, sectors, candidates
            from tools.regime_market.evaluate import WINDOWS
            panel = data.load()
            t = sectors.build(panel)
            X = candidates.c5_trends(panel, final=False)
            c5 = sectors.trend_pred(X, t['y1'], 1)
            icpt, macro = split_pred(X, t['y1'], 1)
            hist = sectors.hist_mean(t['y1'], 1)
            ev = {w: (sectors.evaluate(c5, hist, t['mom'], t['y1'], *WINDOWS[w]),
                      sectors.evaluate(macro, hist, t['mom'], t['y1'], *WINDOWS[w])) for w in ('dev', 'test')}
        finally:
            sys.path.remove(str(MACRO))
    return panel, c5, icpt, macro, ev


def main():
    panel, c5, icpt, macro, ev = build()
    if not reproduces(ev['test'][0]):
        raise RuntimeError('Published C5 did not reproduce; no M1 run')
    diff = (icpt + macro - c5).abs().max().max()
    if not diff < 1e-9:
        raise RuntimeError(f'Intercept + macro != C5 (max diff {diff})')
    checks = []
    for w in ('dev', 'test'):
        r = ev[w][1]
        checks.append((f'M1a {w} IC p < 0.10', f'IC {r["ic"]:+.4f}, p {r["ic_p"]:.4f}', r['ic_p'] < .10))
        checks.append((f'M1a {w} alpha p < 0.10', f'alpha {r["alpha_ann"]:+.2%}/yr, p {r["alpha_p"]:.4f}', r['alpha_p'] < .10))

    etf = run_s2.load_etfs()
    m1, held_m1 = run_s2.tilt_returns(macro, etf, panel.r_rf)
    s2, held_s2 = run_s2.tilt_returns(c5, etf, panel.r_rf)
    fixed_pred = pd.DataFrame(0., index=c5.index, columns=c5.columns)
    fixed_pred[['NoDur', 'Utils', 'Enrgy']] = [3., 2., 1.]
    fixed_pred = fixed_pred.where(c5.notna().all(axis=1), np.nan, axis=0)
    fx, _ = run_s2.tilt_returns(fixed_pred, etf, panel.r_rf)
    ret = pd.DataFrame({'SPY': m1.SPY, 'M1b': m1.S2, 'S2': s2.S2.reindex(m1.index),
                        'FIXED': fx.S2.reindex(m1.index), 'cash': m1.cash})
    rows = []
    for period, sub in [('FULL', ret), ('H1', ret.loc[:'2012-12-31']), ('H2', ret.loc['2013-01-01':]),
                        ('P2021', ret.loc['2021-01-01':])]:
        for rule in ('SPY', 'M1b', 'S2', 'FIXED'):
            rows.append({'period': period, 'rule': rule, 'months': len(sub), **monthly_metrics(sub[rule], sub.cash)})
    table = pd.DataFrame(rows)
    lk = table.set_index(['rule', 'period'])
    for p in ('FULL', 'H1', 'H2'):
        a, b = lk.loc[('M1b', p), 'sharpe'], lk.loc[('SPY', p), 'sharpe']
        checks.append((f'M1b {p} Sharpe > SPY', f'{a:.4f} > {b:.4f}', a > b))
    boot = bootstrap_not_better(ret.M1b, ret.SPY, ret.cash, block=3, n=2000, seed=7)
    checks.append(('M1b bootstrap not-better < 0.25', f'{boot:.4f}', boot < .25))

    # K1 with M1 picks (information only, K1 rules unchanged)
    picks = {d.to_period('M'): list(r.index[r.eq(1)]) for d, r in held_m1.iterrows()}
    with read_only_external():
        inputs = load_inputs()
        levels = pd.DataFrame({t: load(t + '.US_TR', refresh=False).sort_index() for t in etf.columns})
    w, _, _ = run_k1.v2_weights(inputs)
    end = held_m1.index[-1]
    idx = inputs.loc['1999-01-01':end].index
    pidx = inputs.index[inputs.index.get_loc(idx[0]) - 1:inputs.index.get_loc(end) + 1]
    k1m, _ = run_k1.combine(run_k1.etf_daily_returns(levels, pidx).reindex(idx), picks, w, inputs.cash_ret)
    k1m = k1m.loc['1999-02-01':end]
    k1 = pd.read_csv(HERE / 'out/k1_metrics.csv')
    krows = []
    for period, sub in [('FULL', k1m), ('H1', k1m.loc[:'2012-12-31']), ('H2', k1m.loc['2013-01-01':]), ('P2021', k1m.loc['2021-01-01':])]:
        mm = metrics(sub.K1, inputs.cash_ret.reindex(sub.index))
        old = k1[(k1.rule == 'K1') & (k1.period == period)].iloc[0]
        bh = k1[(k1.rule == 'BH') & (k1.period == period)].iloc[0]
        krows.append(f'| {period} | {mm["sharpe"]:.3f} | {old.sharpe:.3f} | {bh.sharpe:.3f} | {mm["cagr"]:.2%} | {old.cagr:.2%} | {mm["maxdd"]:.1%} | {old.maxdd:.1%} |')

    m1a = all(c[2] for c in checks[:4]); m1b = all(c[2] for c in checks[4:])
    rep = ['# M1 — macro-only sector rank', '',
           f'Verdict: M1a {"PASS" if m1a else "FAIL"} · M1b {"PASS" if m1b else "FAIL"} · overall {"PASS" if m1a and m1b else "FAIL"}', '',
           f'Checks: C5 reproduces (IC {ev["test"][0]["ic"]:+.3f}, LS {ev["test"][0]["ls_ann"]:+.1%}); intercept + macro = C5 (max diff {diff:.1e}).', '',
           '## Gate', '', '| Gate | Evidence | Result |', '|---|---|---|']
    rep += [f'| {n} | {e} | {"PASS" if p else "FAIL"} |' for n, e, p in checks]
    rep += ['', '## Signal test, Ken French 12 (M1a vs published C5)', '',
            '| Window | Score | IC | IC p | Long-short/yr | Alpha/yr | Alpha p |', '|---|---|---:|---:|---:|---:|---:|']
    for wnd in ('dev', 'test'):
        for name, r in zip(('C5', 'M1'), ev[wnd]):
            rep.append(f'| {wnd} | {name} | {r["ic"]:+.3f} | {r["ic_p"]:.3f} | {r["ls_ann"]:+.1%} | {r["alpha_ann"]:+.1%} | {r["alpha_p"]:.3f} |')
    rep += ['', '## SPDR ETFs, monthly (M1b; S2 and FIXED XLP/XLU/XLE for information)', '',
            '| Period | Rule | Months | Sharpe | CAGR | Max DD |', '|---|---|---:|---:|---:|---:|']
    rep += [f'| {r.period} | {r.rule} | {r.months} | {r.sharpe:.3f} | {r.cagr:.2%} | {r.maxdd:.1%} |' for r in table.itertuples()]
    rep += ['', '## Share of months picked', '', '| ETF | M1 | C5 (S2) |', '|---|---:|---:|']
    for t in sorted(set(etf.columns) - {'SPY'}):
        rep.append(f'| {t} | {held_m1[t].mean():.0%} | {held_s2[t].mean():.0%} |')
    rep += ['', '## K1 with M1 picks (information only; daily, 1999-02 → ' + str(end.date()) + ')', '',
            '| Period | Sharpe K1-M1 | Sharpe K1 | Sharpe BH | CAGR K1-M1 | CAGR K1 | Max DD K1-M1 | Max DD K1 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    rep += krows
    rep += ['', '## Method', '', 'M1 = C5 regression unchanged (expanding OLS per industry on g12, i12; lag 1; cached vintage), score = slope terms only. '
            'M1a uses macro_health sectors.evaluate unchanged. M1b reuses run_s2.tilt_returns unchanged. FIXED ranks NoDur > Utils > Enrgy every month C5 exists, same machinery. '
            'K1-M1 reuses run_k1.v2_weights and combine unchanged. Inputs read-only. Test no. 13 on this history.', '']
    (HERE / 'RESULTS_m1.md').write_text('\n'.join(rep))
    table.to_csv(HERE / 'out/m1_metrics.csv', index=False)
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
