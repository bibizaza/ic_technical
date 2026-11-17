"""Run the approved step-1 research experiment using offline stored data only."""
from research.us_gauge.gauge import dmas_on, gauge_frame
from research.us_gauge.revisions import detect_rolls, revisions, rev_zscore
from research.us_gauge.trading import decision_dates


def build_signals(inputs, rev_window=63, z_threshold=0.0):
    rolls = detect_rolls(inputs.fy1, inputs.fy2)
    rev = revisions(inputs.fy1, inputs.fy2, rolls, window=rev_window)
    z = rev_zscore(rev)
    dates = decision_dates(inputs.index, '1990-01-01')
    gf = gauge_frame(dmas_on(inputs.px, dates), z)
    # Only the explicitly approved sensitivity changes the earnings key threshold.
    gf['R_in'] = gf.rev_z >= z_threshold
    gf['cash_R1'] = ~gf.T_in & ~gf.R_in
    return gf


from pathlib import Path
import pandas as pd
from research.us_gauge.data import load_inputs
from research.us_gauge.trading import sleeve_weights, net_returns
from research.us_gauge.evaluate import (
    metrics, bootstrap_not_better, switch_risk_table, worst_drawdowns,
)

HERE = Path(__file__).resolve().parent


def _simulate(inputs, gf, rule, cash_weight):
    decisions = gf.loc['1993-04-01':]
    w = sleeve_weights(inputs.index, decisions.index, decisions[f'cash_{rule}'], decisions.deep, cash_weight)
    return w, net_returns(w, inputs.eq_ret, inputs.cash_ret)


def _metric_rows(rule, version, r, w, cash):
    rows = []
    for period, subset in [('FULL', r), ('H1', r.loc[:'2009-12-31']), ('H2', r.loc['2010-01-01':])]:
        weights = w.reindex(subset.index)
        # Count actual full-run transitions, including one on a half's first day.
        switches = w.diff().abs().gt(0).reindex(subset.index).sum() / (len(subset) / 252)
        rows.append({'rule': rule, 'version': version, 'period': period,
                     **metrics(subset, cash.reindex(subset.index)),
                     'switches_per_year': switches,
                     'share_in_cash': float(weights.lt(1).mean())})
    return rows


def main():
    inputs = load_inputs()
    gf = build_signals(inputs)
    runs = {}
    for rule in ('R0', 'R1'):
        for version, cash_weight in [('all', 0.), ('half', .5)]:
            w, r = _simulate(inputs, gf, rule, cash_weight)
            runs[(rule, version)] = (w, r.dropna())
    index = runs[('R0', 'all')][1].index
    # Every rule, including BH, uses the same first defined return through the end.
    runs = {key: (w.reindex(index), r.reindex(index)) for key, (w, r) in runs.items()}
    runs = {('BH', 'none'): (pd.Series(1., index=index), inputs.eq_ret.reindex(index)), **runs}
    cash = inputs.cash_ret.reindex(index)
    rows = []
    for (rule, version), (w, r) in runs.items():
        if r.isna().any() or cash.isna().any():
            raise ValueError('Missing returns in evaluation period')
        rows.extend(_metric_rows(rule, version, r, w, cash))
    table = pd.DataFrame(rows)
    lookup = table.set_index(['rule', 'version', 'period'])
    bootstrap = {version: bootstrap_not_better(runs[('R1', version)][1], runs[('R0', version)][1], cash)
                 for version in ('all', 'half')}
    a = lookup.loc[('R1', 'all', 'FULL')]
    b = lookup.loc[('R0', 'all', 'FULL')]
    sharpe_checks = {period: lookup.loc[('R1', 'all', period), 'sharpe'] > lookup.loc[('R0', 'all', period), 'sharpe']
                     for period in ('FULL', 'H1', 'H2')}
    gates = [all(sharpe_checks.values()), a.maxdd >= b.maxdd - .02,
             bootstrap['all'] < .25, a.switches_per_year <= b.switches_per_year]
    verdict = 'PASS' if all(gates) else 'FAIL'
    evaluated_gauge = gf.loc['1993-04-01':]
    risk = switch_risk_table(evaluated_gauge, 'cash_R1')
    drawdowns = []
    for (rule, version), (_, r) in runs.items():
        drawdowns.append(worst_drawdowns(r).assign(rule=rule, version=version))
    sensitivity = []
    for label, window, threshold in [('z_threshold=-0.5', 63, -.5), ('rev_window=126', 126, 0.)]:
        alternative = build_signals(inputs, rev_window=window, z_threshold=threshold)
        for version, cash_weight in [('all', 0.), ('half', .5)]:
            w, r = _simulate(inputs, alternative, 'R1', cash_weight)
            for row in _metric_rows('R1', version, r.reindex(index), w.reindex(index), cash):
                sensitivity.append({'sensitivity': label, 'information_only': True, **row})
    out = HERE / 'out'
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'step1_metrics.csv', index=False)
    gf.to_csv(out / 'step1_signals.csv', index_label='date')
    risk.to_csv(out / 'step1_switch_risk.csv', index=False)
    pd.concat(drawdowns, ignore_index=True).to_csv(out / 'step1_drawdowns.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out / 'step1_sensitivity.csv', index=False)
    latest = gf.iloc[-1]
    grade = risk.set_index('band').loc[latest.band, 'grade']
    result = [
        '# US quant gauge — step 1 results', '', f'Verdict: {verdict}', '',
        f'First decision: {evaluated_gauge.index[0].date()}. Common return sample: {index[0].date()} to {index[-1].date()} ({len(index):,} trading days).',
        'H1 ends 2009-12-31; H2 starts 2010-01-01. Both are slices of the FULL simulation.', '',
        '## Pre-registered gate (all version)', '',
        '| Condition | Evidence | Result |', '|---|---|---|',
    ]
    sharpe_evidence = '; '.join(
        f'{p}: {lookup.loc[("R1", "all", p), "sharpe"]:.6f} > {lookup.loc[("R0", "all", p), "sharpe"]:.6f} ({"PASS" if sharpe_checks[p] else "FAIL"})'
        for p in ('FULL', 'H1', 'H2'))
    conditions = [
        ('Sharpe R1 > R0 in FULL, H1 and H2', sharpe_evidence),
        ('FULL maxdd R1 >= maxdd R0 - 0.02', f'{a.maxdd:.6f} >= {b.maxdd - .02:.6f}'),
        ('Paired bootstrap not-better share < 0.25', f'{bootstrap["all"]:.4f} < 0.25'),
        ('FULL switches/year R1 <= R0', f'{a.switches_per_year:.6f} <= {b.switches_per_year:.6f}'),
    ]
    result.extend(f'| {name} | {evidence} | {"PASS" if passed else "FAIL"} |'
                  for (name, evidence), passed in zip(conditions, gates))
    result += ['', '## All / half comparison (FULL)', '',
               '| Rule | Version | Sharpe | CAGR | Max DD | Calmar | Volatility | Switches/year | Share in Cash |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in table[table.period == 'FULL'].itertuples():
        result.append(f'| {row.rule} | {row.version} | {row.sharpe:.6f} | {row.cagr:.2%} | {row.maxdd:.2%} | {row.calmar:.6f} | {row.vol:.2%} | {row.switches_per_year:.6f} | {row.share_in_cash:.2%} |')
    result += ['', f'Bootstrap R1 <= R0: all {bootstrap["all"]:.4f}; half {bootstrap["half"]:.4f}.',
               'Paired moving blocks: 63 trading days, 2,000 draws, numpy.random.default_rng(7).',
               'All versus half remains the user’s decision; the gate applies only to all.', '',
               '## Latest gauge', '',
               f'Decision date: {gf.index[-1].date()}. State: {"Cash" if latest.cash_R1 else "Invested"}. G: {latest.G:.6f}. Band: {latest.band}. Switch-risk grade: {grade}. DMAS: {int(latest.dmas)}. REV_z: {latest.rev_z:.6f}.', '',
               '## Sensitivity — information only', '',
               'Only the pre-registered REV_z threshold -0.5 and revision window 126 trading days were run, separately, for all and half. FULL/H1/H2 metrics are in out/step1_sensitivity.csv. They do not select the rule or affect the gate.', '',
               '## Method and validation', '',
               '- All six source series were loaded with load(name, refresh=False); no fetch or store writes.',
               '- DMAS calls the production four-factor functions without a benchmark.',
               '- The five-date 1995–2025 signal truncation check passed before results were generated; trading tests verify t+2 return exposure.',
               '- Standard deviations use ddof=1; annualization uses 252 trading days. Cash-state share counts days with equity weight below 1, including the half version.',
               '- Switch-risk grades use evaluation-period decision weeks and exclude terminal observations without a complete forward horizon, separately for 4 and 13 weeks; denominators are in the CSV. Grades are descriptive full-sample estimates.',
               '- Drawdowns are non-overlapping peak-to-recovery episodes; blank recovery means unrecovered. Initial capital is included in the high-water mark.', '',
               '## Deviations', '',
               '- Per user instruction, skipped the already completed cash-data fetch and all commit/push steps.',
               '- The plan conflicts on cost timing: global constraints say execution day (t+1), but Task 4 explicitly charges on the new-weight return day (t+2). This implementation follows Task 4. Exposure also starts at t+2, after execution at the next close.',
               '- For a holiday Tuesday mapped into the preceding month (for example January 1 mapped to December 31), monthly grouping follows Task 4: the actual decision date’s calendar month. This differs from a literal calendar-Tuesday interpretation of spec 6.2; no alternate rule was run.',
               '- The initial position is undefined until the first decision has executed. All rules use the common first defined return (two trading rows after the first April 1993 decision); no initial-allocation switch cost is assumed.',
               '- No thresholds, windows, trading rules, or gates were fitted to results; only the two approved information-only sensitivities were run.', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(result))
    print(f'Verdict: {verdict}; outputs written to {out}')


if __name__ == '__main__':
    main()
