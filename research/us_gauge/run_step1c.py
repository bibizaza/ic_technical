"""Run exactly section 6c with offline inputs and unchanged Delphi mechanics."""
from pathlib import Path
import pandas as pd
from research.us_gauge.data import load_inputs
from research.us_gauge.gauge import vix_high, r4_cash
from research.us_gauge.run_step1 import build_signals, _simulate, _metric_rows
from research.us_gauge.trading import r2_positions, weights_from_positions, sleeve_weights, net_returns
from research.us_gauge.evaluate import metrics, bootstrap_not_better, worst_drawdowns

HERE = Path(__file__).resolve().parent


def main():
    inputs = load_inputs()
    gf = build_signals(inputs).loc['1993-04-01':].copy()
    gf['vix_mean'] = inputs.vix.rolling(21, min_periods=21).mean().reindex(gf.index)
    gf['cash_R4'] = r4_cash(gf.T_in, gf.R_in, vix_high(inputs.vix).reindex(gf.index))
    positions = r2_positions(gf.index, gf.dmas, gf.rev_z)
    runs = {}
    for rule in ('R0', 'R1', 'R2', 'R3', 'R4'):
        for version, cash_weight in [('all', 0.), ('half', .5)]:
            if rule in ('R0', 'R1', 'R4'):
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
        subset = r.loc['2021-01-01':]
        rows.append({'rule': rule, 'version': version, 'period': 'P2021',
                     **metrics(subset, cash.reindex(subset.index)),
                     'switches_per_year': w.diff().abs().gt(0).reindex(subset.index).sum() / (len(subset) / 252),
                     'share_in_cash': float(w.reindex(subset.index).lt(1).mean())})
    table = pd.DataFrame(rows)
    lookup = table.set_index(['rule', 'version', 'period'])
    bootstrap = {version: bootstrap_not_better(runs[('R4', version)][1], runs[('R1', version)][1], cash)
                 for version in ('all', 'half')}
    checks = []
    for period in ('FULL', 'H1', 'H2'):
        a = lookup.loc[('R4', 'all', period), 'sharpe']
        b = lookup.loc[('R1', 'all', period), 'sharpe']
        checks.append((f'{period} Sharpe R4 > R1', f'{a:.6f} > {b:.6f}', a > b))
    a = lookup.loc[('R4', 'all', 'FULL')]
    bh = lookup.loc[('BH', 'none', 'FULL')]
    r0 = lookup.loc[('R0', 'all', 'FULL')]
    checks += [('FULL Sharpe R4 > BH', f'{a.sharpe:.6f} > {bh.sharpe:.6f}', a.sharpe > bh.sharpe),
               ('Bootstrap R4 <= R1 share < 0.25', f'{bootstrap["all"]:.4f} < 0.25', bootstrap['all'] < .25),
               ('FULL switches/year R4 <= R0', f'{a.switches_per_year:.6f} <= {r0.switches_per_year:.6f}', a.switches_per_year <= r0.switches_per_year)]
    verdict = 'PASS' if all(check[2] for check in checks) else 'FAIL'
    result = ['# US quant gauge — step 1c results', '', f'Verdict: {verdict}', '',
              f'First decision: {gf.index[0].date()}. Common return sample: {index[0].date()} to {index[-1].date()} ({len(index):,} trading days).',
              'H1 ends 2009-12-31; H2 starts 2010-01-01; P2021 starts 2021-01-01. All sub-periods slice the FULL simulation without restarting state. P2021 is information only.', '',
              '## Pre-registered gate (all version)', '', '| Condition | Evidence | Result |', '|---|---|---|']
    result.extend(f'| {name} | {evidence} | {"PASS" if passed else "FAIL"} |' for name, evidence, passed in checks)
    result += ['', f'Bootstrap R4 <= R1: all {bootstrap["all"]:.4f}; half {bootstrap["half"]:.4f}.',
               'Paired moving blocks: 63 trading days, 2,000 draws, numpy.random.default_rng(7), separately for each version. Only all enters the gate; max drawdown is reported, not gated.',
               'Four candidates have now been tried on the US. Any PASS still needs the unchanged Europe check; earlier verdicts remain unchanged.', '',
               '## Metrics', '', 'CAGR and maxDD are percentages. Sharpe uses excess returns over the unchanged cash series.', '',
               '| Period | Rule | Version | Sharpe | CAGR | maxDD | Switches/year |',
               '|---|---|---|---:|---:|---:|---:|']
    for period in ('FULL', 'H1', 'H2', 'P2021'):
        for rule, version in runs:
            if version == 'half' and rule not in ('R1', 'R4'):
                continue
            row = lookup.loc[(rule, version, period)]
            result.append(f'| {period} | {rule} | {version} | {row.sharpe:.6f} | {row.cagr:.4%} | {row.maxdd:.4%} | {row.switches_per_year:.6f} |')
    # Two empty future slots let the unchanged scheduler expose the latest
    # decision's target even when its t+2 return is not yet observable.
    # These slots never enter returns or evaluation.
    extended = inputs.index.append(pd.bdate_range(inputs.index[-1] + pd.Timedelta(days=1), periods=2))
    target = sleeve_weights(extended, gf.index, gf.cash_R4, gf.deep, 0.).iloc[-1]
    latest = gf.iloc[-1]
    result += ['', '## Latest R4 reading', '',
               f'Decision date: {gf.index[-1].date()}. State: {"Cash" if target == 0 else "Invested"} (scheduled decision state). DMAS: {latest.dmas:.6f}; REV_z: {latest.rev_z:.6f}; VIX 21d average: {latest.vix_mean:.6f}.', '',
               '## Method and validation', '',
               '- Seven cached source series loaded with load(name, refresh=False); no network, Bloomberg, or Dropbox writes.',
               '- VIX is aligned to S&P trading days using the last observation on or before each day; the 21-day trailing mean includes the decision day. Unavailable means are not high.',
               '- Existing signals, sleeve_weights (deep = DMAS < 30), returns, metrics, bootstrap and drawdown functions are reused. No sensitivities or variants were run.',
               '- out/step1c_metrics.csv contains all rules and versions for all four periods; out/step1c_drawdowns.csv contains the five worst FULL-run drawdown episodes per rule/version.', '',
               '## Deviations', '',
               '- No step-1c threshold, rule, window, or gate changes.',
               '- Inherited step-1 convention: costs are debited on the new-weight return day (t+2); initial allocation has no switch cost; evaluation starts at the first effective weight. This retains the previously documented step-1 cost-timing convention.', '']
    out = HERE / 'out'
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'step1c_metrics.csv', index=False)
    pd.concat([worst_drawdowns(r).assign(rule=rule, version=version)
               for (rule, version), (_, r) in runs.items()], ignore_index=True).to_csv(out / 'step1c_drawdowns.csv', index=False)
    (HERE / 'RESULTS_step1c.md').write_text('Re-run 2026-11-17 with corrected cash leg (BIL.US_TR); see spec correction log.\n\n' + '\n'.join(result))
    print(f'Step 1c verdict: {verdict}. Outputs written to {out}')


if __name__ == '__main__':
    main()
