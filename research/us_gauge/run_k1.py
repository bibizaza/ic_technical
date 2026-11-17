"""The single preregistered K1 combination; offline inputs, unchanged V2/S2 rules."""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_vol, run_s2
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.evaluate import metrics, bootstrap_not_better

HERE = Path(__file__).resolve().parent


def etf_daily_returns(levels, index):
    """Carry only the two preregistered ETF gaps; never fill other inputs."""
    allowed = {'XLI': pd.Timestamp('1999-01-26'), 'IYZ': pd.Timestamp('2000-06-28')}
    tickers = set(run_s2.MAPPING.values()) - {None} | {'SPY', 'IYZ', 'XLC'}
    if not set(levels.columns) <= tickers:
        raise ValueError('K1 price rule applies only to SPY and sector ETFs')
    aligned = levels.reindex(index).copy()
    for ticker in aligned:
        series = aligned[ticker]
        first = series.first_valid_index()
        if first is None:
            continue  # Pre-inception inputs remain unavailable to the pick rule.
        gaps = series.index[series.isna() & (series.index >= first)]
        for date in gaps:
            if date != allowed.get(ticker):
                raise ValueError(f'Unregistered ETF gap: {ticker} {date}')
            aligned.loc[date, ticker] = series.loc[:date].ffill().iloc[-1]
    return aligned.pct_change(fill_method=None)


def reproduce_c5():
    # market_store can already have imported its own top-level src package.
    # Extend that package for macro_health, as run_s1 does for tools.
    import src
    path = str(run_s2.MACRO / 'src')
    if path not in src.__path__:
        src.__path__.append(path)
    return run_s2.reproduce()


def v2_weights(inputs):
    """Exactly run_vol.main's V2-all construction, using its existing functions."""
    v = run_vol.target_weight(inputs.eq_ret)
    gf = run_vol.build_signals(inputs)
    dates = gf.loc['1993-04-01':].index
    baseline, _ = run_vol._simulate(inputs, gf, 'R1', 0.)
    sleeve = baseline.shift(-2).reindex(dates)
    usable = dates[inputs.index.get_indexer(dates) + 2 < len(inputs)]
    return run_vol.daily_weights(inputs.index, usable, v, sleeve), v, baseline


def combine(returns, picks, weight, cash):
    """Holdings recorded before each day's return; reset after first-day close.

    Reset costs, like inherited equity costs, debit the next return day.
    The sleeve drifts even while equity exposure is zero, so its composition is
    defined independently of the V2 timing/size layer.
    """
    weight = weight.reindex(returns.index)
    cash = cash.reindex(returns.index)
    delta = weight.diff().abs().fillna(0.)
    held = pd.Series(0., index=returns.columns)
    previous_month, pending = None, 0.
    rows, holdings = [], []
    for date, ret in returns.iterrows():
        month = date.to_period('M')
        active = held > 0
        if ret[active].isna().any() or pd.isna(ret.SPY) or pd.isna(weight.loc[date]) or pd.isna(cash.loc[date]):
            raise ValueError(f'Missing required daily input: {date}; held ETF gaps: {list(ret[active].index[ret[active].isna()])}')
        holdings.append(held.copy().rename(date))
        sleeve_return = (held[active] * ret[active]).sum() if active.any() else np.nan
        reset = pending
        cost = .001 * (delta.loc[date] + weight.loc[date] * reset)
        rows.append((date, sleeve_return, reset, delta.loc[date], weight.loc[date] * reset, cost,
                     weight.loc[date] * sleeve_return + (1 - weight.loc[date]) * cash.loc[date] - cost))
        if active.any():
            held.loc[active] *= (1 + ret[active]) / (1 + sleeve_return)
        pending = 0.
        if month != previous_month:
            if month not in picks or len(set(picks[month])) != 3:
                raise ValueError(f'Missing three distinct picks: {month}')
            target = pd.Series(0., index=returns.columns)
            target.loc['SPY'] = .7
            target.loc[picks[month]] = .1
            pending = (target - held).abs().sum()
            held = target
        previous_month = month
    result = pd.DataFrame(rows, columns=['date', 'sleeve_return', 'reset_turnover', 'equity_turnover',
                                        'sector_turnover', 'cost', 'K1']).set_index('date')
    result.index = returns.index.copy()
    result['V2-SPY'] = run_vol.net_returns(weight, returns.SPY, cash)
    result['BH'] = returns.SPY
    return result, pd.DataFrame(holdings)


def main():
    # Guard all external imports/reads, including upstream C5 reproduction.
    with read_only_external():
        inputs = load_inputs()
        reproduced, panel, pred, _ = reproduce_c5()
        if not run_s2.reproduces(reproduced):
            raise RuntimeError('Published C5 reproduction failed; no K1 run')
        monthly = run_s2.load_etfs()
        _, selected = run_s2.tilt_returns(pred, monthly, panel.r_rf)
        picks = {date.to_period('M'): list(row.index[row.eq(1)]) for date, row in selected.iterrows()}
        levels = pd.DataFrame({ticker: load(ticker + '.US_TR', refresh=False).sort_index()
                               for ticker in monthly.columns})
    start, end = pd.Timestamp('1999-02-01'), pd.Timestamp('2026-08-31')
    if selected.index[-1] != end or pd.Period('1999-01') not in picks:
        raise ValueError('Preregistered end or January warm-up picks unavailable')
    w, v, baseline = v2_weights(inputs)
    idx = inputs.loc['1999-01-01':end].index
    price_index = inputs.index[inputs.index.get_loc(idx[0]) - 1:inputs.index.get_loc(end) + 1]
    daily = etf_daily_returns(levels, price_index).reindex(idx)
    result, _ = combine(daily, picks, w, inputs.cash_ret)
    result = result.loc[start:end]
    if result[['K1', 'V2-SPY', 'BH']].isna().any().any() or result.index[-1] != end:
        raise ValueError('Incomplete preregistered sample')
    rows = []
    for period, sub in [('FULL', result), ('H1', result.loc[:'2012-12-31']),
                        ('H2', result.loc['2013-01-01':]), ('P2021', result.loc['2021-01-01':])]:
        for rule in ['K1', 'V2-SPY', 'BH']:
            rows.append(dict(period=period, rule=rule, start=sub.index[0].date(), end=sub.index[-1].date(), days=len(sub),
                             **metrics(sub[rule], inputs.cash_ret.reindex(sub.index)),
                             equity_turnover_per_year=sub.equity_turnover.sum() * 252 / len(sub) if rule != 'BH' else 0.,
                             sector_reset_turnover_per_year=sub.sector_turnover.sum() * 252 / len(sub) if rule == 'K1' else 0.))
    table = pd.DataFrame(rows)
    lookup = table.set_index(['rule', 'period'])
    checks = []
    for period, benchmark in [('FULL', 'V2-SPY'), ('H1', 'V2-SPY'), ('H2', 'V2-SPY'), ('FULL', 'BH')]:
        a, b = [lookup.loc[(rule, period), 'sharpe'] for rule in ['K1', benchmark]]
        checks.append((f'{period} Sharpe K1 > {benchmark}', f'{a:.6f} > {b:.6f}', a > b))
    boot = bootstrap_not_better(result.K1, result['V2-SPY'], inputs.cash_ret.reindex(result.index), block=63, n=2000, seed=7)
    turnover = lookup.loc[('K1', 'FULL'), 'equity_turnover_per_year']
    checks += [('Bootstrap not-better share < 0.25', f'{boot:.4f}', boot < .25),
               ('Equity turnover/year <= 2.35', f'{turnover:.6f}', turnover <= 2.35)]
    report = ['# K1 — full US combination', '', f'Verdict: {"PASS" if all(c[2] for c in checks) else "FAIL"}', '',
              f'Sample: {result.index[0].date()} through {result.index[-1].date()} ({len(result)} trading days). Halves split 2013-01-01; P2021 is information only.', '',
              f'C5 reproduction: n={reproduced["n"]}, IC={reproduced["ic"]:+.9f}, long-short={reproduced["ls_ann"]:+.6%}.', '',
              '## Gate table', '', '| Gate | Evidence | Result |', '|---|---|---|']
    report += [f'| {name} | {evidence} | {"PASS" if passed else "FAIL"} |' for name, evidence, passed in checks]
    report += ['', '## Metrics', '', '| Period | Rule | Sharpe | CAGR | Max DD | Vol | Equity turnover/yr | Sector-reset turnover/yr |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
    report += [f'| {r.period} | {r.rule} | {r.sharpe:.6f} | {r.cagr:.4%} | {r.maxdd:.4%} | {r.vol:.4%} | {r.equity_turnover_per_year:.6f} | {r.sector_reset_turnover_per_year:.6f} |' for r in table.itertuples()]
    report += ['', '## Latest reading', '',
               f'As of sample end {end.date()}: effective R1 state = {"Cash" if baseline.loc[end] == 0 else "Invested"}; v = {v.loc[end]:.6f}; effective equity weight w = {w.loc[end]:.6f}. v is the current volatility target; w includes the scheduled trading thresholds and lag.',
               f'Last holding month: {end.to_period("M")}; ETFs: {", ".join(picks[end.to_period("M")])}.', '',
               '## Method', '',
               'V2-all uses the unchanged run_vol functions and identical construction; V2-SPY shares the exact same daily w. S2 picks come directly from unchanged run_s2.tilt_returns, including its C5 reproduction, mapping, distinct-ETF selection and availability rule. Daily returns use offline EODHD adjusted total-return levels; sizing retains the preregistered S&P input. Cash retains DTB3 then BIL.',
               'Each monthly reset executes at the first trading-day close and earns from the following day. Holdings drift daily and reset to 70/10/10/10. January 1999 warms up the sleeve; all windows slice the continuous simulation. Costs are 0.001 times absolute equity-weight change plus 0.001 times w times internal reset turnover. Reported sector turnover is w times internal reset turnover, in total-portfolio units. Annual turnover divides by trading days/252.',
               'Sharpe uses daily excess returns, ddof=1 and sqrt(252); volatility uses total returns. Paired bootstrap uses 63-day blocks, 2,000 draws, default_rng(7). No variants; US candidate testing is closed.', '',
               '## Deviations', '',
               'Data rule added in spec commit 3c369ba before any K1 metrics: carry the last observed ETF price on XLI 1999-01-26 and IYZ 2000-06-28; return is zero on the missing day and the full price move is booked on the next observed day. Applied only to K1 SPY/sector price inputs; any unregistered post-inception price gap stops the run, and missing required returns, cash or equity weights still stop. No other rule, threshold, window or gate changes. Integration fix: extend the already loaded market_store src package search path with macro_health/src so unchanged C5 imports resolve; no external files are modified. Inherited cost accounting debits equity changes and monthly resets on the first return day with the new weights (the day after execution close). January warm-up absorbs initial sleeve allocation before the evaluation window. Latest reading is dated to the common sample end, rather than mixing later timing data with stale sector picks. All market-store loads use refresh=False; external inputs are read-only, with network and external writes guarded. No commits or pushes.', '']
    table.to_csv(HERE / 'out/k1_metrics.csv', index=False)
    (HERE / 'RESULTS_k1.md').write_text('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    main()
