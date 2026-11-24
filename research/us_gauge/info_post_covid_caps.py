import numpy as np, pandas as pd
from research.us_gauge.data import load
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_q import stats
from research.us_gauge.run_t import product, run, capped, canary_cap, rolled
from research.us_gauge.run_kr import single_cap
P = product(); idx, cash = P['idx'], P['inputs'].cash_ret
with read_only_external():
    ewy, eem, agg = (load(n, refresh=False).sort_index() for n in ('EWY.US_TR','EEM.US_TR','AGG.US_TR'))
caps = {'KR': single_cap(ewy, idx), 'T2': canary_cap(agg, eem, idx)}
R = {'Product': run(P, P['base'])}
for k, c in caps.items(): R[k] = run(P, capped(P, c))
sw = lambda w, a: (w.loc[a:].diff().abs() > 1e-9).sum() / (len(w.loc[a:]) / 252)
for a, b in [('2020-01-01', None), ('2020-06-01', None), ('2023-01-01', None)]:
    print(f'--- {a} -> end')
    for k, (r, w) in R.items():
        s, d, c = stats(r.loc[a:b], cash)
        print(f'{k:8s} Sharpe {s:.2f}  return {c:.1%}  worst {d:.1%}  trades/yr {sw(w, a):.1f}')
# placebo on post-2020-06 only, 300 draws, information
rng = np.random.default_rng(7)
for k, c in caps.items():
    real = stats(R[k][0].loc['2020-06-01':], cash)[0]; nd = int(c.notna().sum())
    pl = np.array([stats(run(P, capped(P, rolled(c, int(s))))[0].loc['2020-06-01':], cash)[0] for s in rng.integers(252, nd-252+1, size=300)])
    print(k, 'post-2020-06 placebo p', round(float((pl >= real).mean()), 3), 'median', round(float(np.median(pl)), 2))
