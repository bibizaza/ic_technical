"""Run only pre-registered V1/V2, using offline cached inputs."""
from pathlib import Path
import pandas as pd
from research.us_gauge.data import load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals, _simulate
from research.us_gauge.trading import decision_dates, net_returns
from research.us_gauge.volsize import target_weight, daily_weights, turnover_per_year
from research.us_gauge.evaluate import metrics, bootstrap_not_better

HERE = Path(__file__).resolve().parent


def main():
    with read_only_external():
        inputs = load_inputs()
    if inputs.index[0] > pd.Timestamp('1950-01-03') or inputs.loc['1954-01-04':, ['eq_ret', 'cash_ret']].isna().any().any():
        raise ValueError('Required early offline history unavailable')
    v = target_weight(inputs.eq_ret)
    dates = decision_dates(inputs.index, '1955-01-01')
    w1 = daily_weights(inputs.index, dates, v)
    r1 = net_returns(w1, inputs.eq_ret, inputs.cash_ret).dropna()
    runs = {'V1': (w1, r1, pd.Series(1., index=inputs.index), inputs.eq_ret.reindex(r1.index), 'BH')}
    gf = build_signals(inputs)
    dates2 = gf.loc['1993-04-01':].index
    for version, cash_weight in [('all', 0.), ('half', .5)]:
        baseline_w, baseline_r = _simulate(inputs, gf, 'R1', cash_weight)
        # Undo only the exposure shift to retrieve R1's scheduled state at t.
        sleeve = baseline_w.shift(-2).reindex(dates2)
        usable = dates2[inputs.index.get_indexer(dates2) + 2 < len(inputs)]
        w = daily_weights(inputs.index, usable, v, sleeve)
        r = net_returns(w, inputs.eq_ret, inputs.cash_ret).dropna()
        runs['V2-' + version] = (w, r, baseline_w, baseline_r.reindex(r.index), 'R1-' + version)
    rows, report = [], ['# Volatility sizing — V1 and V2', '']
    for rule, (w, r, bw, br, benchmark) in runs.items():
        cash = inputs.cash_ret.reindex(r.index)
        if br.isna().any() or cash.isna().any():
            raise ValueError('Missing common-sample returns')
        split = '1990-01-01' if rule == 'V1' else '2010-01-01'
        windows = {'FULL': r.index, 'H1': r.index[r.index < split], 'H2': r.index[r.index >= split],
                   'P2021': r.loc['2021-01-01':].index}
        if rule == 'V1':
            start = inputs.index[inputs.index.get_loc(dates2[0]) + 2]
            windows.update({'G_FULL': r.loc[start:].index, 'G_H1': r.loc[start:'2009-12-31'].index,
                            'G_H2': r.loc['2010-01-01':].index})
        current = []
        for period, idx in windows.items():
            for label, ret, weight in [(rule, r, w), (benchmark, br, bw)]:
                current.append({'experiment': rule, 'rule': label, 'period': period,
                                'start': idx[0].date(), 'end': idx[-1].date(), 'days': len(idx),
                                **metrics(ret.reindex(idx), cash.reindex(idx)),
                                'turnover': turnover_per_year(weight, idx)})
        rows.extend(current)
        lookup = pd.DataFrame(current).set_index(['rule', 'period'])
        checks = []
        for period in ('FULL', 'H1', 'H2'):
            a, b = [lookup.loc[(label, period), 'sharpe'] for label in (rule, benchmark)]
            checks.append((f'{period} Sharpe > {benchmark}', f'{a:.6f} > {b:.6f}', a > b))
        boot = bootstrap_not_better(r, br, cash, block=63, n=2000, seed=7)
        turnover = turnover_per_year(w, r.index)
        checks += [('Bootstrap not-better share < 0.25', f'{boot:.4f}', boot < .25),
                   ('FULL turnover/year <= 2.35', f'{turnover:.6f}', turnover <= 2.35)]
        report += [f'## {rule} versus {benchmark}', '', f'Verdict: {rule} {"PASS" if all(c[2] for c in checks) else "FAIL"}', '',
                   f'Return sample: {r.index[0].date()} through {r.index[-1].date()}; halves split {split}.', '',
                   '| Gate | Evidence | Result |', '|---|---|---|']
        report += [f'| {name} | {evidence} | {"PASS" if passed else "FAIL"} |' for name, evidence, passed in checks]
        report += ['', '| Period | Rule | Sharpe | CAGR | Max DD | Turnover/year |', '|---|---|---:|---:|---:|---:|']
        report += [f'| {x["period"]} | {x["rule"]} | {x["sharpe"]:.6f} | {x["cagr"]:.4%} | {x["maxdd"]:.4%} | {x["turnover"]:.6f} |' for x in current]
        report += ['']
    report += ['## Method', '',
               'Sigma uses the last 21 total returns through t, ddof=1, annualized by sqrt(252); v=min(1,0.15/sigma). Combined target changes trade at >=0.10 on the first monthly decision, >=0.25 on other weekly decisions. R1 state changes always execute. Tuesday/holiday calendar and R1 states reuse the gauge.',
               'Decision t, execution close t+1, return exposure t+2. Cost is 10 bp times absolute weight change. Cash uses DTB3/100/252 through 2007-05-30 and BIL total returns thereafter. S&P total returns use price change plus SHILLER_SPX_DIV/price/252.',
               'Turnover is sum of absolute daily weight changes divided by sample days/252; boundary changes are retained. All subperiods slice the continuous simulation. G_FULL/G_H1/G_H2 are the April 1993 gauge window and halves split 2010. P2021 is information only; V1 gates use the 1955 FULL/H1/H2 sample.',
               'Sharpe uses excess returns. Bootstrap uses paired 63-day blocks, 2,000 draws, default_rng(7), separately for V1 and both V2 versions. Both V2 versions have separately reported gates. CSV includes dates, days, volatility and Calmar.', '',
               '## Deviations', '',
               'No rule, threshold, window or gate changes; no variants. load_inputs already includes BBG_SPX from 1950 and DTB3 from 1954-01-04, so it is unchanged. Inherited gauge conventions: initial allocation has no turnover/cost; costs are debited on t+2; holiday decisions use the actual trading date’s calendar month. No external writes or network access.', '']
    pd.DataFrame(rows).to_csv(HERE / 'out/vol_metrics.csv', index=False)
    (HERE / 'RESULTS_vol.md').write_text('\n'.join(report))
    print('V1/V2 results written.')


if __name__ == '__main__':
    main()
