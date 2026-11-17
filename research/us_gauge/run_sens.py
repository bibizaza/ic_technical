"""Robustness checks, information only (asked by the user after the verdicts; nothing here changes a rule).

1. Sleeve split: SPY share 100% / 80% / 70% (product) / 50%, sectors macro-only, on the full product weight
   V2 + F + S-live + B (10-day Deemer).
2. Breadth window: 5-day sums instead of Deemer's 10, same 1.97 threshold, same exits; R0 gates and product.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_k1, run_s2
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_m1 import build as build_m1
from research.us_gauge.run_b import breadth, deemer_ratio, thrust_days, gates, product_weights
from research.us_gauge.trading import net_returns
from research.us_gauge.stress_periods import stats
from research.us_gauge.combo_f import WINDOWS

HERE = Path(__file__).resolve().parent
END = pd.Timestamp('2026-08-31')


def table(L, ret, cash, title):
    order = list(ret.columns)
    rows = [dict(window=n, rule=c, **dict(zip(('total', 'sharpe', 'maxdd'), stats(ret.loc[a:b, c], cash))))
            for n, a, b in WINDOWS for c in ret]
    t = pd.DataFrame(rows)
    for metric, fmt, sub in (('sharpe', '{:.2f}', 'Sharpe'), ('total', '{:+.1%}', 'Total return'), ('maxdd', '{:.1%}', 'Worst fall')):
        piv = t.pivot_table(index='window', columns='rule', values=metric, sort=False)[order]
        L += [f'### {title} — {sub}', '', '| Window | ' + ' | '.join(order) + ' |', '|---|' + '---:|' * len(order)]
        L += [f'| {w} | ' + ' | '.join('' if pd.isna(x) else fmt.format(x) for x in row) + ' |' for w, row in piv.iterrows()]
        L.append('')
    return t


def main():
    panel, c5, _, macro, _ = build_m1()
    with read_only_external():
        inputs = load_inputs()
    br = breadth()
    inputs = inputs.loc[:br.index[-1]]
    etf = run_s2.load_etfs()
    _, held_m1 = run_s2.tilt_returns(macro, etf, panel.r_rf)
    with read_only_external():
        levels = pd.DataFrame({t: load(t + '.US_TR', refresh=False).sort_index() for t in etf.columns})
    idx = inputs.index
    k_idx = inputs.loc['1999-01-01':END].index
    pidx = idx[idx.get_loc(k_idx[0]) - 1:idx.get_loc(END) + 1]
    daily = run_k1.etf_daily_returns(levels, pidx).reindex(k_idx)
    picks = {d.to_period('M'): list(r.index[r.eq(1)]) for d, r in held_m1.iterrows()}
    K = lambda w, s: run_k1.combine(daily, picks, w, inputs.cash_ret, spy_share=s)[0].K1

    t10 = thrust_days(deemer_ratio(br.adv, br.dec, 10))
    t5 = thrust_days(deemer_ratio(br.adv, br.dec, 5))
    base, w10, _ = product_weights(inputs, t10)
    _, w5, ep5 = product_weights(inputs, t5)
    L = ['# Robustness checks (information only)', '',
         'Asked after the verdicts. Nothing here changes a rule. Product weight = V2 + F + S-live + B, sectors macro-only. '
         f'Daily, 10 bp, 1999-02 → {END.date()}. Code: `run_sens.py`.', '']

    ret = pd.DataFrame({'Buy & hold': inputs.eq_ret, **{f'{int(s * 100)}% SPY / {100 - int(s * 100)}% sectors': K(w10, s)
                                                     for s in (1., .8, .7, .5)}}).loc['1999-02-01':END]
    L += ['## 1. Sleeve split (the product today = 70/30)', '']
    table(L, ret, inputs.cash_ret.reindex(ret.index), 'Sleeve split')

    L += ['## 2. Breadth window: 5 days vs Deemer 10 days (threshold 1.97 both)', '',
          f'Thrust days: 10-day {len(t10)}, 5-day {len(t5)}.', '']
    dmas = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0]
    for nm, th in (('10-day (the rule)', t10), ('5-day', t5)):
        g = gates(inputs, dmas, th)
        met = sum(bool(c[2]) for c in g['checks'])
        L += [f'**R0+B, {nm}:** {len(g["ep"])} episodes, {int((g["ep"].pnl_vs_R0 > 0).sum())} positive; gates G1-G6 met {met} of {len(g["checks"])}.', '',
              '| Gate | Evidence | Result |', '|---|---|---|']
        L += [f'| {n} | {e} | {"PASS" if p else "FAIL"} |' for n, e, p in g['checks']]
        L.append('')
    R = lambda w: net_returns(w, inputs.eq_ret, inputs.cash_ret)
    ret2 = pd.DataFrame({'Buy & hold': inputs.eq_ret, 'V2+F+S (no B)': R(base), '+B 10-day': R(w10), '+B 5-day': R(w5),
                         'K1 70/30 +B 10-day': K(w10, .7), 'K1 70/30 +B 5-day': K(w5, .7)}).loc['1999-02-01':END]
    table(L, ret2, inputs.cash_ret.reindex(ret2.index), 'Product, B 10-day vs 5-day')
    L.append('5-day B episodes on the R1 base: ' + ', '.join(f'{r.entry.date()}→{r.end.date()} ({r.reason})' for r in ep5.itertuples()))
    (HERE / 'RESULTS_sens.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
