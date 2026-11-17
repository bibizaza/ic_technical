"""Pre-registered E1: monthly 126-day relative momentum, offline inputs only."""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge.data import load
from research.us_gauge.trading import decision_dates, weights_from_positions
from research.us_gauge.evaluate import metrics, bootstrap_not_better

HERE = Path(__file__).resolve().parent


def signals(px):
    rel_level = np.log(px.RSP / px.SPY)
    rel = rel_level - rel_level.shift(126)
    dates = decision_dates(px.index, str(px.index[0].date()))
    dates = dates[~dates.to_period('M').duplicated()]
    rel = rel.reindex(dates).dropna()
    return pd.DataFrame({'rel': rel, 'rsp': rel.gt(0).astype(float)})


def simulate(px, sig):
    w = weights_from_positions(px.index, sig.rsp.astype(bool), 0.)
    returns = px.pct_change(fill_method=None)
    cost = .001 * (w.diff().abs() + (1 - w).diff().abs()).fillna(0)
    return w, (w * returns.RSP + (1 - w) * returns.SPY - cost)


def main():
    px = pd.concat({symbol: load(symbol + '.US_TR', refresh=False).sort_index()
                    for symbol in ('SPY', 'RSP')}, axis=1).loc['2003-04-30':].dropna()
    if px.empty or (px <= 0).any().any():
        raise ValueError('Missing or invalid offline ETF prices')
    sig = signals(px)
    w, r = simulate(px, sig)
    r = r.dropna()
    cash = load('FRED_DTB3', refresh=False).sort_index().reindex(r.index, method='ffill') / 100 / 252
    runs = {'SPY': px.SPY.pct_change(fill_method=None).reindex(r.index),
            'RSP': px.RSP.pct_change(fill_method=None).reindex(r.index), 'E1': r}
    if cash.isna().any() or any(x.isna().any() for x in runs.values()):
        raise ValueError('Missing returns in evaluation sample')
    rows = []
    for rule, ret in runs.items():
        for period, sub in [('FULL', ret), ('H1', ret.loc[:'2014-12-31']),
                            ('H2', ret.loc['2015-01-01':]), ('P2021', ret.loc['2021-01-01':])]:
            switches = w.diff().abs().gt(0).reindex(sub.index).sum() if rule == 'E1' else 0
            rows.append({'rule': rule, 'period': period, **metrics(sub, cash.reindex(sub.index)),
                         'switches_per_year': switches / (len(sub) / 252)})
    table = pd.DataFrame(rows)
    lookup = table.set_index(['rule', 'period'])
    checks = []
    for period in ('FULL', 'H1', 'H2'):
        a, b = (lookup.loc[(rule, period), 'sharpe'] for rule in ('E1', 'SPY'))
        checks.append((f'{period} Sharpe E1 > SPY', f'{a:.6f} > {b:.6f}', a > b))
    boot = bootstrap_not_better(r, runs['SPY'], cash)
    switches = lookup.loc[('E1', 'FULL'), 'switches_per_year']
    checks += [('Bootstrap not-better share < 0.25', f'{boot:.4f} < 0.25', boot < .25),
               ('FULL switches/year <= 2.35', f'{switches:.6f} <= 2.35', switches <= 2.35)]
    verdict = 'PASS' if all(c[2] for c in checks) else 'FAIL'
    report = ['# E1 — equal weight vs cap weight', '', f'Verdict: {verdict}', '',
              f'First decision: {sig.index[0].date()}. Common return sample: {r.index[0].date()} to {r.index[-1].date()} ({len(r):,} trading days).',
              'H1 ends 2014-12-31; H2 starts 2015-01-01. P2021 is information only. Sub-periods slice the full run.', '',
              '## Pre-registered gate', '', '| Condition | Evidence | Result |', '|---|---|---|']
    report += [f'| {name} | {evidence} | {"PASS" if passed else "FAIL"} |' for name, evidence, passed in checks]
    report += ['', '## Metrics', '', '| Period | Rule | Sharpe | CAGR | Max DD | Switches/year |',
               '|---|---|---:|---:|---:|---:|']
    for row in table.itertuples():
        report.append(f'| {row.period} | {row.rule} | {row.sharpe:.6f} | {row.cagr:.4%} | {row.maxdd:.4%} | {row.switches_per_year:.6f} |')
    report += ['', '## Method', '',
               'SPY.US_TR and RSP.US_TR; cash = FRED_DTB3/100/252. Inputs loaded with refresh=False.',
               'At each first monthly Tuesday decision, hold RSP if the 126-trading-day change in ln(RSP/SPY) is positive, otherwise SPY. Execution t+1 close; returns from t+2.',
               'Cost: 10 bp times the sum of absolute changes in both ETF weights (20 bp per full switch). Initial allocation has no switch cost; switch costs are charged at t+2, following the existing gauge convention. Holiday Tuesdays use the preceding trading date and its calendar month.',
               'Sharpe uses excess returns and ddof=1. Paired moving-block bootstrap: 63 days, 2,000 draws, default_rng(7). No variants.', '',
               '## Deviations', '', 'No rule, threshold, window or gate changes. Inherited cost and calendar conventions are stated above.', '']
    (HERE / 'out').mkdir(exist_ok=True)
    table.to_csv(HERE / 'out/e1_metrics.csv', index=False)
    (HERE / 'RESULTS_e1.md').write_text('\n'.join(report))
    print(f'E1 verdict: {verdict}')


if __name__ == '__main__':
    main()
