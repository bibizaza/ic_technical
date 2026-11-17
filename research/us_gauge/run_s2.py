"""Pre-registered C5 tilt using cached ETF monthly total-return levels."""
from pathlib import Path
import pandas as pd
from research.us_gauge.run_s1 import MACRO, reproduce, reproduces, read_only_external, monthly_metrics
from research.us_gauge.evaluate import bootstrap_not_better

HERE = Path(__file__).resolve().parent
MAPPING = dict(NoDur='XLP', Durbl='XLY', Manuf='XLI', Enrgy='XLE', Chems='XLB',
               BusEq='XLK', Utils='XLU', Shops='XLY', Hlth='XLV', Money='XLF', Other=None)


def mapped_etf(industry, month):
    if industry == 'Telcm':
        return 'IYZ' if month < pd.Timestamp('2018-06-01') else 'XLC'
    return MAPPING[industry]


def pick_etfs(scores, available, month):
    picks = []
    # Same ascending ranking as S1, traversed from highest score downward.
    for industry in scores.dropna().sort_values().index[::-1]:
        ticker = mapped_etf(industry, month)
        if ticker is not None and ticker not in picks and pd.notna(available.get(ticker)):
            picks.append(ticker)
            if len(picks) == 3:
                break
    return picks


def monthly_returns(levels):
    levels = levels.sort_index().copy()
    levels.index = levels.index.to_period('M').to_timestamp('M')
    levels = levels.groupby(levels.index).last().asfreq('ME')
    return levels.pct_change(fill_method=None)


def load_etfs():
    tickers = sorted(set(MAPPING.values()) - {None} | {'IYZ', 'XLC', 'SPY'})
    with read_only_external():
        return pd.DataFrame({ticker: monthly_returns(pd.read_csv(
            MACRO / f'data/regime_market/raw/bbg_tr_{ticker}.csv', index_col=0,
            parse_dates=True).iloc[:, 0]) for ticker in tickers})


def tilt_returns(pred, etf, rf):
    previous = pd.Series(0., index=etf.columns)
    rows, holdings = [], []
    started = False
    for month, returns in etf.iterrows():
        mapped = {mapped_etf(industry, month) for industry in [*MAPPING, 'Telcm']} - {None}
        eligible = sum(pd.notna(returns.get(ticker)) for ticker in mapped)
        if not started and (pd.isna(returns.SPY) or eligible < 9):
            continue
        if pd.isna(returns.SPY):
            raise ValueError(f'Missing SPY after start: {month}')
        decision = month - pd.offsets.MonthEnd(1)
        if decision not in pred.index or pred.loc[decision].isna().any() or pd.isna(rf.get(month)):
            # End at the last common cached prediction/outcome/RF month.
            if started and month <= pred.dropna().index[-1] and pd.notna(rf.get(month)):
                raise ValueError(f'Missing interior prediction: {decision}')
            continue
        picks = pick_etfs(pred.loc[decision], returns, month)
        if len(picks) != 3:
            raise ValueError(f'Fewer than three available ETFs: {month}')
        weights = pd.Series(0., index=etf.columns)
        weights.loc['SPY'] = .7
        weights.loc[picks] = .1
        turnover = (weights - previous).abs().sum()
        ret = (weights[weights > 0] * returns[weights > 0]).sum() - .001 * turnover
        rows.append((month, ret, returns.SPY - .001 * (not started), rf.loc[month], turnover))
        holdings.append(pd.Series({ticker: int(ticker in picks) for ticker in etf.columns}, name=month))
        previous, started = weights, True
    return (pd.DataFrame(rows, columns=['date', 'S2', 'SPY', 'cash', 'turnover']).set_index('date'),
            pd.DataFrame(holdings))


def main():
    result, panel, pred, _ = reproduce()
    if not reproduces(result):
        (HERE / 'RESULTS_s2.md').write_text('# S2\n\nVerdict: STOPPED — C5 reproduction failed.\n')
        raise RuntimeError('Published C5 result did not reproduce')
    etf = load_etfs()
    returns, held = tilt_returns(pred, etf, panel.r_rf)
    rows = []
    for period, sub in [('FULL', returns), ('H1', returns.loc[:'2012-12-31']),
                        ('H2', returns.loc['2013-01-01':]), ('P2021', returns.loc['2021-01-01':])]:
        for rule in ('SPY', 'S2'):
            rows.append({'period': period, 'rule': rule, 'start': sub.index[0].date(),
                         'end': sub.index[-1].date(), 'months': len(sub),
                         **monthly_metrics(sub[rule], sub.cash)})
    table = pd.DataFrame(rows)
    lookup = table.set_index(['rule', 'period'])
    checks = []
    for period in ('FULL', 'H1', 'H2'):
        a, b = [lookup.loc[(rule, period), 'sharpe'] for rule in ('S2', 'SPY')]
        checks.append((f'{period} Sharpe S2 > SPY', f'{a:.6f} > {b:.6f}', a > b))
    boot = bootstrap_not_better(returns.S2, returns.SPY, returns.cash, block=3, n=2000, seed=7)
    checks.append(('Bootstrap not-better share < 0.25', f'{boot:.4f}', boot < .25))
    report = ['# S2 — C5 on sector ETFs', '', f'Verdict: {"PASS" if all(c[2] for c in checks) else "FAIL"}', '',
              f'C5 reproduction: n={result["n"]}, IC={result["ic"]:+.9f}, long-short={result["ls_ann"]:+.6%}; published precision reproduced.', '',
              f'Return months: {returns.index[0].date()} through {returns.index[-1].date()} ({len(returns)} months). Halves split 2013-01-01; P2021 is information only.', '',
              '## Pre-registered gate', '', '| Gate | Evidence | Result |', '|---|---|---|']
    report += [f'| {name} | {evidence} | {"PASS" if passed else "FAIL"} |' for name, evidence, passed in checks]
    report += ['', '## Metrics', '', '| Period | Rule | Months | Sharpe | CAGR | Max DD |', '|---|---|---:|---:|---:|---:|']
    report += [f'| {r.period} | {r.rule} | {r.months} | {r.sharpe:.6f} | {r.cagr:.4%} | {r.maxdd:.4%} |' for r in table.itertuples()]
    report += ['', '## ETF holding frequency', '', '| ETF | Months held | Share of months |', '|---|---:|---:|',
               f'| SPY | {len(held)} | 100.00% |']
    for ticker in sorted(set(etf.columns) - {'SPY'}):
        report.append(f'| {ticker} | {held[ticker].sum()} | {held[ticker].mean():.2%} |')
    report += ['', '## Method', '',
               'Reused run_s1.reproduce exactly: cached macro_health panel, C5 final=False, expanding industry regressions, lag=1 and the original vintage. Predictions at month-end t are held in t+1. Windows and telecom mapping are indexed by holding/return month.',
               'Cached bbg_tr_<TICKER>.csv month-end levels are normalized to calendar month-end, then converted to returns without filling missing months. Start requires SPY and nine distinct mapped ETFs with returns. Pick the first three distinct available ETFs down the C5 ranking; skip Other and missing returns. Durbl/Shops both map to XLY; Telcm maps to IYZ through May 2018 and XLC from June 2018. XLC is skipped until it has a computable monthly return.',
               'S2 holds 70% SPY and 10% in each selected ETF. Monthly cost is 10 bp times absolute target-weight turnover, matching S1. Both portfolios pay 10 bp initial allocation cost. Cash for Sharpe is outcome-month Ken French RF. Subperiods slice the full run without restart.',
               'Sharpe uses ddof=1 and sqrt(12). Paired bootstrap: 3-month blocks, 2,000 draws, default_rng(7). This is not independent C5 evidence.', '',
               '## Deviations', '',
               f'No rule, threshold, window or gate changes; no variants. The final common return month is {returns.index[-1].date()}, limited by cached C5 predictions and Ken French RF as well as ETF returns (ETF cache ends {etf.dropna(how="all").index[-1].date()}). All inputs were read-only; no downloads or external writes.', '']
    table.to_csv(HERE / 'out/s2_metrics.csv', index=False)
    (HERE / 'RESULTS_s2.md').write_text('\n'.join(report))
    print('S2 results written.')


if __name__ == '__main__':
    main()
