"""Run only the pre-registered step-1b candidates, using offline stored data."""
from pathlib import Path
import pandas as pd
from research.us_gauge.data import load_inputs
from research.us_gauge.run_step1 import build_signals, _simulate, _metric_rows
from research.us_gauge.trading import r2_positions, weights_from_positions, sleeve_weights, net_returns
from research.us_gauge.evaluate import bootstrap_not_better, worst_drawdowns

HERE = Path(__file__).resolve().parent


def main():
    inputs = load_inputs()
    gf = build_signals(inputs).loc['1993-04-01':]
    positions = r2_positions(gf.index, gf.dmas, gf.rev_z)
    runs = {}
    for rule in ('R0', 'R1', 'R2', 'R3'):
        for version, cash_weight in [('all', 0.), ('half', .5)]:
            if rule in ('R0', 'R1'):
                w, r = _simulate(inputs, gf, rule, cash_weight)
            else:
                if rule == 'R2':
                    w = weights_from_positions(inputs.index, positions.pos.eq('Invested'), cash_weight)
                else:
                    w = sleeve_weights(inputs.index, gf.index, gf.rev_z.lt(0).fillna(False),
                                       pd.Series(False, index=gf.index), cash_weight)
                r = net_returns(w, inputs.eq_ret, inputs.cash_ret)
            runs[(rule, version)] = (w, r.dropna())
    index = runs[('R0', 'all')][1].index
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
    bootstrap = {(rule, version): bootstrap_not_better(runs[(rule, version)][1], runs[('R0', version)][1], cash)
                 for rule in ('R2', 'R3') for version in ('all', 'half')}
    result = ['# US quant gauge — step 1b results', '',
              f'First decision: {gf.index[0].date()}. Common return sample: {index[0].date()} to {index[-1].date()} ({len(index):,} trading days).',
              'H1 ends 2009-12-31; H2 starts 2010-01-01. Halves slice the FULL simulation without restarting state.', '',
              'Three candidates (R1, R2, R3) have now been tried on the US. A US PASS qualifies for the unchanged Europe check; it is not adoption.', '']
    for rule in ('R2', 'R3'):
        a, b = lookup.loc[(rule, 'all', 'FULL')], lookup.loc[('R0', 'all', 'FULL')]
        checks = []
        for period in ('FULL', 'H1', 'H2'):
            candidate = lookup.loc[(rule, 'all', period), 'sharpe']
            baseline = lookup.loc[('R0', 'all', period), 'sharpe']
            checks.append((f'{period} Sharpe > R0', f'{candidate:.6f} > {baseline:.6f}', candidate > baseline))
        checks += [('FULL maxDD >= R0 maxDD - 0.02', f'{a.maxdd:.6f} >= {b.maxdd - .02:.6f}', a.maxdd >= b.maxdd - .02),
                   ('Bootstrap not-better share < 0.25', f'{bootstrap[(rule, "all")]:.4f} < 0.25', bootstrap[(rule, 'all')] < .25),
                   ('FULL switches/year <= R0', f'{a.switches_per_year:.6f} <= {b.switches_per_year:.6f}', a.switches_per_year <= b.switches_per_year)]
        verdict = 'PASS' if all(check[2] for check in checks) else 'FAIL'
        result += [f'## {rule} vs R0 — {verdict} (all)', '', '| Condition | Evidence | Result |', '|---|---|---|']
        result.extend(f'| {name} | {evidence} | {"PASS" if passed else "FAIL"} |' for name, evidence, passed in checks)
        result += ['', f'Bootstrap {rule} <= R0: all {bootstrap[(rule, "all")]:.4f}; half {bootstrap[(rule, "half")]:.4f}.', '']
    result += ['Paired moving blocks: 63 trading days, 2,000 draws, numpy.random.default_rng(7), separately for each candidate/version.',
               'All versus half remains the user’s decision; only all enters the gate.', '',
               '## R3 vs BH', '', '| Period | R3 version | R3 Sharpe | BH Sharpe | R3 maxDD | BH maxDD |', '|---|---|---:|---:|---:|---:|']
    for period in ('FULL', 'H1', 'H2'):
        b = lookup.loc[('BH', 'none', period)]
        for version in ('all', 'half'):
            a = lookup.loc[('R3', version, period)]
            result.append(f'| {period} | {version} | {a.sharpe:.6f} | {b.sharpe:.6f} | {a.maxdd:.6f} | {b.maxdd:.6f} |')
    early = positions.index[positions.pos.shift().eq('Cash') & positions.pos.eq('Invested') & positions['mode'].eq('earnings')]
    latest = positions.iloc[-1]
    result += ['', '## R2 earnings re-entries', '',
               f'Count: {len(early)}. These are Cash-to-Invested transitions into earnings mode, including any on a monthly decision date (spec 6b allows any decision date).', '',
               ', '.join(str(d.date()) for d in early) if len(early) else 'None.', '',
               '## Latest R2 reading', '',
               f'Decision date: {gf.index[-1].date()}. pos: {latest.pos}; mode: {latest["mode"]}; armed: {latest.armed}; DMAS: {gf.dmas.iloc[-1]:.6f}; REV_z: {gf.rev_z.iloc[-1]:.6f}.', '',
               '## Method and validation', '',
               '- Six cached source series loaded with load(name, refresh=False); no network, Bloomberg, or Dropbox writes.',
               '- Existing step-1 signals, return construction, metrics, bootstrap, and drawdown functions are reused unchanged. No sensitivities or other variants were run.',
               '- Signal truncation and state-machine prefix tests, plus t+2 exposure tests, passed before this run.',
               '- Full metrics for BH/R0/R1/R2/R3 and both exposure versions are in out/step1b_metrics.csv; five worst drawdown episodes per run are in out/step1b_drawdowns.csv; decision states are in out/step1b_positions_r2.csv.', '',
               '## Deviations', '',
               '- No step-1b threshold, rule, window, or gate changes.',
               '- Inherited step-1 mechanics: costs are debited on the new-weight return day (t+2); the initial allocation has no switch cost; evaluation starts at the first effective weight. This preserves the existing convention despite the previously documented step-1 plan conflict about cost timing.',
               '- Monthly grouping uses the actual decision date’s calendar month, including holiday adjustments, as specified in 6b.',
               '- Normal trend re-entry leaves armed unchanged: 6b explicitly resets it only on earnings re-entry and on every exit. The retained flag has no effect while invested.', '']
    out = HERE / 'out'
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'step1b_metrics.csv', index=False)
    positions.to_csv(out / 'step1b_positions_r2.csv', index_label='date')
    pd.concat([worst_drawdowns(r).assign(rule=rule, version=version)
               for (rule, version), (_, r) in runs.items()], ignore_index=True).to_csv(out / 'step1b_drawdowns.csv', index=False)
    (HERE / 'RESULTS_step1b.md').write_text('Re-run 2026-11-17 with corrected cash leg (BIL.US_TR); see spec correction log.\n\n' + '\n'.join(result))
    print(f'Step 1b outputs written to {out}')


if __name__ == '__main__':
    main()
