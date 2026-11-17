"""Information only: the full combination with the VIX re-entry rule F, by period and in stress windows.

Rules unchanged: R1, V2 (run_k1.v2_weights), F (run_p, ratio 0.5, VIX < 30, 50% floor, R1 cash spells),
K1 sleeve (run_k1.combine: 70% SPY + 3 x 10% picks, drifting, monthly reset) with C5 or macro-only picks.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_k1, run_s2
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals, _simulate
from research.us_gauge.run_m1 import build as build_m1
from research.us_gauge.run_p import base_states, episodes
from research.us_gauge.run_c import floor_weights
from research.us_gauge.trading import net_returns
from research.us_gauge.stress_periods import stats

HERE = Path(__file__).resolve().parent
END = pd.Timestamp('2026-08-31')
WINDOWS = [
    ('FULL 1999-02 to 2026-08', '1999-02-01', None), ('First half 1999-2012', '1999-02-01', '2012-12-31'),
    ('Second half 2013-2026', '2013-01-01', None),
    ('2009-2013', '2009-03-10', '2013-12-31'), ('2014-2018', '2014-01-01', '2018-12-31'),
    ('2019-2023', '2019-01-01', '2023-12-31'), ('2021-2026', '2021-01-01', None), ('2023-2026', '2023-01-01', None),
    ('Dot-com bear 2000-03 to 2002-10', '2000-03-24', '2002-10-09'), ('2008 crisis 2007-10 to 2009-03', '2007-10-09', '2009-03-09'),
    ('2009 rebound Mar-Dec', '2009-03-10', '2009-12-31'),
    ('COVID fall 19 Feb-23 Mar 2020', '2020-02-20', '2020-03-23'), ('COVID rebound 24 Mar-31 Aug 2020', '2020-03-24', '2020-08-31'),
    ('2022 full year', '2022-01-01', '2022-12-31'), ('2023 rebound', '2023-01-01', '2023-12-31'),
    ('Liberation Day fall 19 Feb-8 Apr 2025', '2025-02-20', '2025-04-08'),
    ('Liberation Day rebound 9 Apr-30 Jun 2025', '2025-04-09', '2025-06-30'), ('2025 full year', '2025-01-01', '2025-12-31'),
]


def main():
    panel, c5, _, macro, _ = build_m1()   # first: macro_health's src package must load before market_store's
    with read_only_external():
        inputs = load_inputs()
    etf = run_s2.load_etfs()
    _, held_c5 = run_s2.tilt_returns(c5, etf, panel.r_rf)
    _, held_m1 = run_s2.tilt_returns(macro, etf, panel.r_rf)
    with read_only_external():
        levels = pd.DataFrame({t: load(t + '.US_TR', refresh=False).sort_index() for t in etf.columns})
    idx = inputs.index
    gf = build_signals(inputs)
    w1, r1 = _simulate(inputs, gf, 'R1', 0.)
    w2, _, _ = run_k1.v2_weights(inputs)
    w2 = w2.reindex(idx)
    d1 = gf.loc['1993-04-01':].index
    st1 = base_states(d1, gf.cash_R1, gf.deep)
    ep = episodes(idx, st1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    fl = floor_weights(idx, list(zip(ep.i_entry, ep.i_end)), .5)
    wF = np.maximum(w2, fl).where(w2.notna())
    r2 = net_returns(w2, inputs.eq_ret, inputs.cash_ret)
    rF = net_returns(wF, inputs.eq_ret, inputs.cash_ret)
    k_idx = inputs.loc['1999-01-01':END].index
    pidx = idx[idx.get_loc(k_idx[0]) - 1:idx.get_loc(END) + 1]
    daily = run_k1.etf_daily_returns(levels, pidx).reindex(k_idx)
    picks = lambda held: {d.to_period('M'): list(r.index[r.eq(1)]) for d, r in held.iterrows()}
    k1, _ = run_k1.combine(daily, picks(held_c5), w2, inputs.cash_ret)
    k1F, _ = run_k1.combine(daily, picks(held_c5), wF, inputs.cash_ret)
    k1MF, _ = run_k1.combine(daily, picks(held_m1), wF, inputs.cash_ret)
    ret = pd.DataFrame({'Buy & hold': inputs.eq_ret, 'R1': r1, 'V2': r2, 'V2+F': rF,
                        'K1 (C5)': k1.K1, 'K1+F (C5)': k1F.K1, 'K1+F (macro-only)': k1MF.K1}).loc['1999-02-01':END]
    cash = inputs.cash_ret.reindex(ret.index)
    rows = []
    for name, a, b in WINDOWS:
        sub = ret.loc[a:b]
        for rule in ret:
            tot, sh, dd = stats(sub[rule], cash)
            n = len(sub)
            rows.append(dict(window=name, rule=rule, total=tot, cagr=(1 + tot) ** (252 / n) - 1 if n > 250 else np.nan,
                             sharpe=sh, maxdd=dd))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / 'out/combo_f.csv', index=False)
    order = list(ret.columns)
    L = ['# Full combination with the VIX re-entry rule F (information only)', '',
         f'All rules end {END.date()} (last C5 / macro-only pick month). Daily, 10 bp costs, cash = T-bill then BIL. '
         'V2 = R1 timing x volatility size. F = buy back to at least 50% while R1 is in cash, once VIX <= half its peak of the cash spell and < 30. '
         'K1 = V2 weight x (70% SPY + 3 x 10% sector ETFs). Macro-only = C5 without its intercept (M1). Code: `combo_f.py`.', '',
         'F episodes on R1 (entry → end): ' + ', '.join(f'{r.entry.date()}→{r.end.date()}' for r in ep.itertuples()), '']
    for metric, fmt, title in (('sharpe', '{:.2f}', 'Sharpe (return per unit of risk)'), ('cagr', '{:.1%}', 'Return a year (windows over 1 year)'),
                               ('total', '{:+.1%}', 'Total return in the window'), ('maxdd', '{:.1%}', 'Worst fall inside the window')):
        piv = t.pivot_table(index='window', columns='rule', values=metric, sort=False)[order]
        L += [f'## {title}', '', '| Window | ' + ' | '.join(order) + ' |', '|---|' + '---:|' * len(order)]
        for w, row in piv.iterrows():
            L.append(f'| {w} | ' + ' | '.join('' if pd.isna(v) else fmt.format(v) for v in row) + ' |')
        L.append('')
    (HERE / 'RESULTS_combo_f.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
