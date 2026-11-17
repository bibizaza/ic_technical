"""Information only (not a test, no gate): frozen rules by period and in stress windows.

Rules are unchanged: BH, R0, R1 (all-to-cash), V2-all, K1. Same offline inputs as run_k1.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_k1, run_s2
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals, _simulate
from research.us_gauge.trading import net_returns

HERE = Path(__file__).resolve().parent


def build():
    with read_only_external():
        inputs = load_inputs()
        _, panel, pred, _ = run_k1.reproduce_c5()
        monthly = run_s2.load_etfs()
        _, selected = run_s2.tilt_returns(pred, monthly, panel.r_rf)
        picks = {d.to_period('M'): list(r.index[r.eq(1)]) for d, r in selected.iterrows()}
        levels = pd.DataFrame({t: load(t + '.US_TR', refresh=False).sort_index() for t in monthly.columns})
    gf = build_signals(inputs)
    w0, r0 = _simulate(inputs, gf, 'R0', 0.)
    w1, r1 = _simulate(inputs, gf, 'R1', 0.)
    w2, _, _ = run_k1.v2_weights(inputs)
    r2 = net_returns(w2, inputs.eq_ret, inputs.cash_ret)
    end = selected.index[-1]
    idx = inputs.loc['1999-01-01':end].index
    pidx = inputs.index[inputs.index.get_loc(idx[0]) - 1:inputs.index.get_loc(end) + 1]
    k1, _ = run_k1.combine(run_k1.etf_daily_returns(levels, pidx).reindex(idx), picks, w2, inputs.cash_ret)
    ret = pd.DataFrame({'BH': inputs.eq_ret, 'R0': r0, 'R1': r1, 'V2': r2, 'K1': k1.K1}).loc['1999-02-01':]
    wts = pd.DataFrame({'R0': w0, 'R1': w1, 'V2': w2}).reindex(ret.index)
    return ret, wts, inputs.cash_ret.reindex(ret.index), inputs


def stats(r, cash):
    x = r.dropna()
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    c = cash.reindex(x.index)
    w = (1 + x).cumprod()
    dd = (w / w.cummax().clip(lower=1) - 1).min()
    ex = x - c
    sh = ex.mean() / ex.std() * np.sqrt(252) if len(x) > 60 else np.nan
    return w.iloc[-1] - 1, sh, dd


WINDOWS = [
    ('2009-2013', '2009-03-10', '2013-12-31'), ('2014-2018', '2014-01-01', '2018-12-31'),
    ('2019-2023', '2019-01-01', '2023-12-31'), ('2021-2026', '2021-01-01', None), ('2023-2026', '2023-01-01', None),
    ('COVID fall 19 Feb-23 Mar 2020', '2020-02-20', '2020-03-23'), ('COVID rebound 24 Mar-31 Aug 2020', '2020-03-24', '2020-08-31'),
    ('2022 full year', '2022-01-01', '2022-12-31'), ('2022 fall 3 Jan-12 Oct', '2022-01-04', '2022-10-12'),
    ('2023 rebound', '2023-01-01', '2023-12-31'),
    ('Liberation Day fall 19 Feb-8 Apr 2025', '2025-02-20', '2025-04-08'),
    ('Liberation Day rebound 9 Apr-30 Jun 2025', '2025-04-09', '2025-06-30'), ('2025 full year', '2025-01-01', '2025-12-31'),
]


def main():
    ret, wts, cash, inputs = build()
    rows = []
    for name, a, b in WINDOWS:
        sub = ret.loc[a:b]
        for rule in ret:
            tot, sh, dd = stats(sub[rule], cash)
            rows.append(dict(window=name, rule=rule, start=sub[rule].dropna().index.min(),
                             end=sub[rule].dropna().index.max(), total=tot, sharpe=sh, maxdd=dd,
                             avg_weight=wts[rule].loc[a:b].mean() if rule in wts else (np.nan if rule == 'K1' else 1.)))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / 'out/stress_periods.csv', index=False)
    # weekly equity weight around the two stress windows
    path = wts.assign(DMAS_gauge=np.nan).drop(columns='DMAS_gauge')
    weekly = pd.concat([path.loc['2021-11-01':'2023-03-31'], path.loc['2024-12-01':'2025-08-31']]).resample('W-FRI').last().dropna(how='all')
    weekly.to_csv(HERE / 'out/stress_weights_weekly.csv')
    pd.set_option('display.width', 200)
    piv = t.pivot_table(index='window', columns='rule', values=['total', 'sharpe', 'maxdd'], sort=False)
    print(piv.round(3).to_string())
    print(t.groupby('window', sort=False).end.max().to_string())
    print(weekly.round(2).to_string())


if __name__ == '__main__':
    main()
