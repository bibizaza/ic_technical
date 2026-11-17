"""Reproduce the published C5 result before running the pre-registered S1 tilt."""
from contextlib import contextmanager
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MACRO = Path('/Users/larazanella/Desktop/GitHub/Projects/macro_health')
_GUARD_ACTIVE = False


def _audit(event, args):
    if not _GUARD_ACTIVE:
        return
    if event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request', 'subprocess.Popen'):
        raise RuntimeError(f'Offline S1: network/process attempt blocked ({event})')
    if event == 'open':
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing and isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if not resolved.is_relative_to(HERE.parents[1]):
                raise RuntimeError(f'Offline S1: external write blocked: {resolved}')
    if event == 'os.mkdir' and Path(args[0]).is_dir():
        return  # Existing-directory exist_ok probe cannot create or alter it.
    if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename'):
        for path in args[:2] if event == 'os.rename' else args[:1]:
            if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve().is_relative_to(MACRO):
                raise RuntimeError(f'Offline S1: external write blocked: {path}')


sys.addaudithook(_audit)


@contextmanager
def read_only_external():
    global _GUARD_ACTIVE
    previous, bytecode = _GUARD_ACTIVE, sys.dont_write_bytecode
    _GUARD_ACTIVE, sys.dont_write_bytecode = True, True
    try:
        yield
    finally:
        _GUARD_ACTIVE, sys.dont_write_bytecode = previous, bytecode


def reproduces(result):
    """Match published precision and sample size, without tuning tolerances."""
    return (result['n'] == 247 and f"{result['ic']:+.3f}" == '+0.062'
            and f"{result['ls_ann']:+.1%}" == '+6.3%')


def reproduce():
    with read_only_external():
        sys.path.insert(0, str(MACRO))
        try:
            # This worktree has a regular tools package; macro_health/tools is a
            # namespace directory. Extend its search path without copying code.
            import tools
            external_tools = str(MACRO / 'tools')
            if external_tools not in tools.__path__:
                tools.__path__.append(external_tools)
            from tools.regime_market import data, sectors, candidates
            from tools.regime_market.evaluate import WINDOWS
            # sectors.main used data.load(): preserve that exact cached panel/vintage.
            # Industry returns are read from the original cached raw/ff12_vw.csv.
            for path in (data.PANEL, data.RAW / 'ff12_vw.csv'):
                if not path.is_file():
                    raise RuntimeError(f'Required offline cache missing: {path}')
            panel = data.load()
            targets = sectors.build(panel)
            pred = sectors.trend_pred(candidates.c5_trends(panel, final=False), targets['y1'], 1)
            result = sectors.evaluate(pred, sectors.hist_mean(targets['y1'], 1), targets['mom'],
                                      targets['y1'], *WINDOWS['test'])
            industry = sectors.ff12().reindex(panel.index)
            return result, panel, pred, industry
        finally:
            sys.path.remove(str(MACRO))


def tilt_returns(panel, pred, industry):
    """Decision-date indexed next-month returns; target-weight turnover as sectors.long_short."""
    nxt = industry.shift(-1)
    market, rf = panel.r_eq.shift(-1), panel.r_rf.shift(-1)
    previous = pd.Series(0., index=industry.columns)
    previous_market = 0.
    rows = []
    for date, scores in pred.iterrows():
        if scores.isna().any():
            continue
        weights = pd.Series(0., index=industry.columns)
        weights.loc[scores.sort_values().index[-3:]] = .1
        turnover = (weights - previous).abs().sum() + abs(.7 - previous_market)
        initial = previous_market == 0
        previous, previous_market = weights, .7
        if nxt.loc[date].isna().any() or pd.isna(market.loc[date]) or pd.isna(rf.loc[date]):
            continue
        ret = .7 * market.loc[date] + (weights * nxt.loc[date]).sum() - .001 * turnover
        rows.append((date, ret, market.loc[date] - .001 * initial, rf.loc[date], turnover))
    return pd.DataFrame(rows, columns=['date', 'S1', 'market', 'cash', 'turnover']).set_index('date')


def monthly_metrics(r, cash):
    excess = r - cash
    wealth = (1 + r).cumprod()
    maxdd = (wealth / wealth.cummax().clip(lower=1.) - 1).min()
    cagr = wealth.iloc[-1] ** (12 / len(r)) - 1
    return {'sharpe': excess.mean() / excess.std() * np.sqrt(12),
            'cagr': cagr, 'maxdd': maxdd, 'vol': r.std() * np.sqrt(12)}


def run_tilt(panel, pred, industry):
    from research.us_gauge.evaluate import bootstrap_not_better
    returns = tilt_returns(panel, pred, industry).loc['1973-01-31':'2026-09-30']
    rows = []
    for period, sub in [('FULL', returns), ('1973–2005', returns.loc[:'2005-12-31']),
                        ('2006–2026', returns.loc['2006-01-31':])]:
        for rule in ('market', 'S1'):
            rows.append({'period': period, 'rule': rule, 'months': len(sub),
                         **monthly_metrics(sub[rule], sub.cash)})
    table = pd.DataFrame(rows)
    lookup = table.set_index(['rule', 'period'])
    checks = []
    for period in ('FULL', '1973–2005', '2006–2026'):
        a, b = (lookup.loc[(rule, period), 'sharpe'] for rule in ('S1', 'market'))
        checks.append((f'{period} Sharpe S1 > market', f'{a:.6f} > {b:.6f}', a > b))
    boot = bootstrap_not_better(returns.S1, returns.market, returns.cash, block=3, n=2000, seed=7)
    checks.append(('Bootstrap not-better share < 0.25', f'{boot:.4f} < 0.25', boot < .25))
    verdict = 'PASS' if all(c[2] for c in checks) else 'FAIL'
    report = [f'Verdict: {verdict}', '', '## Pre-registered gate', '',
              '| Condition | Evidence | Result |', '|---|---|---|']
    report += [f'| {name} | {evidence} | {"PASS" if passed else "FAIL"} |' for name, evidence, passed in checks]
    report += ['', '## Metrics', '', '| Period | Rule | Months | Sharpe | CAGR | Max DD |',
               '|---|---|---:|---:|---:|---:|']
    for row in table.itertuples():
        report.append(f'| {row.period} | {row.rule} | {row.months} | {row.sharpe:.6f} | {row.cagr:.4%} | {row.maxdd:.4%} |')
    report += ['', '## Method', '',
               f'Common decision-date sample: {returns.index[0].date()} through {returns.index[-1].date()} ({len(returns)} months); each row earns the following month’s return, as in the published sector test. Sub-periods slice the full run.',
               'S1 holds 70% Ken French market and 10% in each of the top three C5 industries. Benchmark holds 100% market. Cash is Ken French RF for the outcome month. All returns are total returns.',
               'Monthly cost is 10 bp times the sum of absolute target-weight changes, matching the published sector turnover convention. Initial allocation is charged to both portfolios at the first available prediction (1976-02). No restart at window boundaries.',
               'Sharpe uses excess returns, ddof=1 and sqrt(12); CAGR annualizes at 12 months. Paired moving-block bootstrap: 3 months, 2,000 draws, default_rng(7).',
               'This is not independent evidence of C5. Ken French industries do not map one-to-one to SPDR sector ETFs; implementation mapping is deferred.', '',
               '## Deviations', '',
               'No rule, threshold, window or gate changes; no variants. The nominal 1973–2005 window has 359 usable months from 1976-02 because C5 needs feature and regression warm-up; the test window has 247 usable months through 2026-07, matching the published sample. The cached panel is reused exactly as in the published runner, alongside cached raw industry returns. The local tools package search path is extended to import macro_health’s namespace package; existing-directory mkdir probes are allowed but directory creation, external writes and network access are blocked.', '']
    (HERE / 'out').mkdir(exist_ok=True)
    table.to_csv(HERE / 'out/s1_metrics.csv', index=False)
    return report


def main():
    try:
        result, panel, pred, industry = reproduce()
    except RuntimeError as exc:
        (HERE / 'RESULTS_s1.md').write_text(f'# S1 — C5 sector tilt\n\nVerdict: STOPPED\n\n{exc}\n\nNo tilt run.\n')
        print(f'S1 STOPPED: {exc}')
        return
    report = ['# S1 — C5 sector tilt', '', '## Published-result reproduction', '',
              '| Measure | Published | Reproduced |', '|---|---:|---:|',
              f"| Months | 247 | {result['n']} |",
              f"| Beta-adjusted 1-month IC | +0.062 | {result['ic']:+.9f} |",
              f"| Long-short annual return | +6.3% | {result['ls_ann']:+.6%} |", '',
              'Reused macro_health/tools/regime_market: cached panel.parquet (the published sectors.main/data.load path) and raw/ff12_vw.csv; no downloads or external writes.',
              'C5 uses c5_trends(final=False), expanding industry regressions with lag=1 and minimum 60 observations. Beta uses 60 months, minimum 36. Long-short uses top/bottom 3 and 10 bp times turnover. Test decision window: 2006-01-31 through 2026-09-30, exactly evaluate.WINDOWS; outcomes are next-month returns.', '']
    if not reproduces(result):
        report += ['Verdict: STOPPED — published C5 result did not reproduce.', '',
                   'Per pre-registration, S1 was stopped before constructing or evaluating the 70% market / 30% top-3 portfolio. No s1_metrics.csv was generated. No alternative lag, vintage, window or variant was tried.', '']
        (HERE / 'RESULTS_s1.md').write_text('\n'.join(report))
        print(f"S1 STOPPED: IC={result['ic']:.9f}, long-short={result['ls_ann']:.9f}, n={result['n']}")
        return
    report += run_tilt(panel, pred, industry)
    (HERE / 'RESULTS_s1.md').write_text('\n'.join(report))
    print('S1 reproduction passed; tilt results written.')


if __name__ == '__main__':
    main()
