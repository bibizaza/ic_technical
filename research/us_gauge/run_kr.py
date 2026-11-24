"""KR: Korea canary cap — pre-registered in spec 6KR (commit 352f598). One run.

Cap = 0.5 when the 13612W momentum of EWY (MSCI South Korea, total return) is negative, else 1.
Base = product spec 11d (run_t.product). Information only: the same rule on EEM.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge.data import load
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.evaluate import bootstrap_not_better
from research.us_gauge.run_q import stats
from research.us_gauge.run_t import EPISODES, m13612, product, run, capped, rolled

HERE = Path(__file__).resolve().parent
SPLIT = '2013-12-31'
N_PLACEBO, MIN_SHIFT, P_PASS = 1000, 252, .10 / 4


def single_cap(tr: pd.Series, index) -> pd.Series:
    """0.5 when 13612W momentum < 0, else 1; NaN until defined."""
    m = m13612(tr.sort_index()).reindex(index, method='ffill')
    return pd.Series(np.where(m < 0, .5, 1.), index=index).where(m.notna())


def main():
    P = product()
    idx, inp = P['idx'], P['inputs']
    cash = inp.cash_ret
    with read_only_external():
        ewy, eem = (load(n, refresh=False) for n in ('EWY.US_TR', 'EEM.US_TR'))
    cap, cap_eem = single_cap(ewy, idx), single_cap(eem, idx)
    a = str(cap.first_valid_index().date())
    r_base, w_base = run(P, P['base'])
    r, w = run(P, capped(P, cap))
    sw = lambda w_, a_: float((w_.loc[a_:].diff().abs() > 1e-9).sum() / (len(w_.loc[a_:]) / 252))
    rb, rx = r_base.loc[a:], r.loc[a:]
    sb, db, cb = stats(rb, cash)
    sx, dx, cx = stats(rx, cash)
    halves = [(f'{a[:7]} → {SPLIT[:7]}', a, SPLIT), (f'after {SPLIT[:7]}', '2014-01-01', None)]
    hs = [(h, stats(r_base.loc[x:y], cash)[0], stats(r.loc[x:y], cash)[0]) for h, x, y in halves]
    nb = bootstrap_not_better(rx.dropna(), rb.dropna(), cash)
    rng = np.random.default_rng(7)
    nd = int(cap.notna().sum())
    pl = np.array([stats(run(P, capped(P, rolled(cap, int(s))))[0].loc[a:], cash)[0]
                   for s in rng.integers(MIN_SHIFT, nd - MIN_SHIFT + 1, size=N_PLACEBO)])
    p = float((pl >= sx).mean())
    gates = [('G1 Sharpe better, full and both halves', f'full {sx:.3f} vs {sb:.3f}; ' + '; '.join(f'{h} {x:.3f} vs {b_:.3f}' for h, b_, x in hs),
              sx > sb and all(x > b_ for _, b_, x in hs)),
             ('G2 worst fall not deeper by > 2 pts', f'{dx:.1%} vs {db:.1%}', dx >= db - .02),
             ('G3 bootstrap not-better share < 25%', f'{nb:.1%}', nb < .25),
             (f'G4 placebo p <= {P_PASS:.3f}', f'p = {p:.3f} (placebo median Sharpe {np.median(pl):.3f})', p <= P_PASS)]
    verdict = 'PASS' if all(g[2] for g in gates) else 'FAIL'
    # information only: EEM with the same rule, same window
    a_e = max(a, str(cap_eem.first_valid_index().date()))
    r_e, w_e = run(P, capped(P, cap_eem))
    se, de, ce = stats(r_e.loc[a_e:], cash)
    sb_e, db_e, cb_e = stats(r_base.loc[a_e:], cash)
    L = ['# KR: Korea canary cap (pre-registered, spec 6KR, one run)', '',
         f'**KR: {verdict}.** Consequence (declared before the run): PASS → proposed to the user; FAIL → rejected list, no change.', '',
         f'Base = product spec 11d, inputs to {idx[-1].date()}. 10 bp, cash = T-bill. Window starts {a} (first day EWY momentum is defined). Code: `run_kr.py`.', '',
         '| Gate | Evidence | Result |', '|---|---|---|']
    L += [f'| {g} | {e} | {"PASS" if ok else "FAIL"} |' for g, e, ok in gates]
    L += ['', f'| {a} → {idx[-1].date()} | Product | Product + KR |', '|---|---:|---:|',
          f'| Sharpe | {sb:.2f} | {sx:.2f} |', f'| Return a year | {cb:.1%} | {cx:.1%} |', f'| Worst fall | {db:.1%} | {dx:.1%} |',
          f'| Trades a year | {sw(w_base, a):.1f} | {sw(w, a):.1f} |', f'| Share of days capped at 50% | | {(cap.loc[a:] < 1).mean():.0%} |', '',
          '## Information only: same rule on EEM (not a test)', '',
          f'| {a_e} → {idx[-1].date()} | Product | Product + EEM-only cap |', '|---|---:|---:|',
          f'| Sharpe | {sb_e:.2f} | {se:.2f} |', f'| Return a year | {cb_e:.1%} | {ce:.1%} |', f'| Worst fall | {db_e:.1%} | {de:.1%} |',
          f'| Trades a year | {sw(w_base, a_e):.1f} | {sw(w_e, a_e):.1f} |', f'| Share of days capped at 50% | | {(cap_eem.loc[a_e:] < 1).mean():.0%} |', '',
          f'Correlation of the KR and EEM-only cap states (daily, from {a_e}): {pd.concat([cap, cap_eem], axis=1).loc[a_e:].corr().iloc[0, 1]:+.2f}.', '',
          '## Information only: stress episodes (total return / worst fall inside the window)', '',
          '| Episode | S&P | Product | KR | EEM-only |', '|---|---:|---:|---:|---:|']
    for name, x, y in EPISODES:
        def cell(rr):
            z = rr.loc[x:y].dropna()
            if len(z) == 0:
                return 'n/a'
            eqc = (1 + z).cumprod()
            return f'{eqc.iloc[-1] - 1:+.1%} / {(eqc / eqc.cummax() - 1).min():.1%}'
        L.append(f'| {name} | {cell(inp.eq_ret)} | {cell(r_base)} | {cell(r)} | {cell(r_e)} |')
    L += ['', f'Caps on the last input day ({idx[-1].date()}): KR = {cap.iloc[-1]:.0%}, EEM-only = {cap_eem.iloc[-1]:.0%}.']
    (HERE / 'RESULTS_kr.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    if '--one-run' not in sys.argv:
        sys.exit('Pre-registered single run: pass --one-run.')
    main()
