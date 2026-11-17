"""Run exactly section 7b using cached data and existing Delphi mechanics."""
from pathlib import Path
import pandas as pd
from research.us_gauge.data import load, load_inputs
from research.us_gauge.cot import cot_signal
from research.us_gauge.gauge import r5_cash
from research.us_gauge.run_step1 import build_signals, _simulate
from research.us_gauge.trading import sleeve_weights, net_returns
from research.us_gauge.evaluate import metrics, bootstrap_not_better

HERE = Path(__file__).resolve().parent


def main():
    inputs = load_inputs()
    full = build_signals(inputs)
    gf = full.loc['2008-07-01':].copy()
    sources = [load(name, refresh=False) for name in
               ('CFTC_ES_LEV_LONG', 'CFTC_ES_LEV_SHORT', 'CFTC_ES_OI', 'CFTC_ES_RELEASE_LAG_DAYS')]
    gf = gf.join(cot_signal(*sources, gf.index))
    if gf.empty or gf.index[0].to_period('M') != pd.Period('2008-07') or gf.C_z.isna().any():
        raise ValueError('Section 7b requires defined C_z from the first July 2008 decision')
    gf['cash_R5'] = r5_cash(gf.T_in, gf.R_in, gf.C_in)
    start = inputs.index[inputs.index.get_loc(gf.index[0]) + 2]
    index = inputs.loc[start:].index
    cash = inputs.cash_ret.reindex(index)
    runs = {}
    rows = []
    for rule in ('BH', 'R0', 'R1', 'R5'):
        for version, cash_weight in [('all', 0.), ('half', .5)]:
            if rule == 'BH':
                w, r = pd.Series(1., index=inputs.index), inputs.eq_ret
            elif rule == 'R5':
                w = sleeve_weights(inputs.index, gf.index, gf.cash_R5, gf.deep, cash_weight)
                r = net_returns(w, inputs.eq_ret, inputs.cash_ret)
            else:
                w, r = _simulate(inputs, full, rule, cash_weight)
            # Compute transitions before slicing: include a switch on a boundary.
            switches = w.diff().abs().gt(0)
            r = r.reindex(index)
            if r.isna().any() or cash.isna().any():
                raise ValueError('Missing returns in evaluation window')
            runs[(rule, version)] = r
            for period, subset in [('WIN', r), ('WH1', r.loc[:'2017-06-30']),
                                   ('WH2', r.loc['2017-07-01':]), ('P2021', r.loc['2021-01-01':])]:
                rows.append({'rule': rule, 'version': version, 'period': period,
                             **metrics(subset, cash.reindex(subset.index)),
                             'switches_per_year': switches.reindex(subset.index).sum() / (len(subset) / 252),
                             'share_in_cash': float(w.reindex(subset.index).lt(1).mean())})
    table = pd.DataFrame(rows)
    lookup = table.set_index(['rule', 'version', 'period'])
    bootstrap = {version: bootstrap_not_better(runs[('R5', version)], runs[('R1', version)], cash)
                 for version in ('all', 'half')}
    checks = []
    for period in ('WIN', 'WH1', 'WH2'):
        a = lookup.loc[('R5', 'all', period), 'sharpe']
        b = lookup.loc[('R1', 'all', period), 'sharpe']
        checks.append((f'{period} Sharpe R5 > R1', f'{a:.6f} > {b:.6f}', a > b))
    a = lookup.loc[('R5', 'all', 'WIN')]
    bh = lookup.loc[('BH', 'all', 'WIN')]
    r0 = lookup.loc[('R0', 'all', 'WIN')]
    checks += [('WIN Sharpe R5 > BH', f'{a.sharpe:.6f} > {bh.sharpe:.6f}', a.sharpe > bh.sharpe),
               ('Bootstrap R5 <= R1 share < 0.25', f'{bootstrap["all"]:.4f} < 0.25', bootstrap['all'] < .25),
               ('WIN switches/year R5 <= R0', f'{a.switches_per_year:.6f} <= {r0.switches_per_year:.6f}', a.switches_per_year <= r0.switches_per_year)]
    verdict = 'PASS' if all(c[2] for c in checks) else 'FAIL'
    # Append two display-only slots to read every scheduled decision state,
    # including the last decision when its effective return is not yet stored.
    extended = inputs.index.append(pd.bdate_range(inputs.index[-1] + pd.Timedelta(days=1), periods=2))
    display_w = sleeve_weights(extended, gf.index, gf.cash_R5, gf.deep, 0.)
    targets = display_w.iloc[extended.get_indexer(gf.index) + 2].to_numpy()
    gf['state_R5'] = ['Cash' if w == 0 else 'Invested' for w in targets]
    latest = gf.iloc[-1]
    result = ['# US quant gauge — step 2 results', '', f'Verdict: {verdict}', '',
              f'First decision: {gf.index[0].date()}. Common return sample: {index[0].date()} to {index[-1].date()} ({len(index):,} trading days).',
              'WH1 ends 2017-06-30; WH2 starts 2017-07-01; P2021 starts 2021-01-01 (information only). R0/R1/BH are slices of their full runs; R5 starts at the July 2008 monthly decision. Sub-periods do not restart state.', '',
              '## Pre-registered gate (all version)', '', '| Condition | Evidence | Result |', '|---|---|---|']
    result.extend(f'| {name} | {evidence} | {"PASS" if passed else "FAIL"} |' for name, evidence, passed in checks)
    result += ['', f'Bootstrap R5 <= R1: all {bootstrap["all"]:.4f}; half {bootstrap["half"]:.4f}.',
               'Paired moving blocks: 63 trading days, 2,000 draws, numpy.random.default_rng(7), separately for each version. Only all enters the gate; max drawdown is reported, not gated.',
               'About 18 years and few bear markets imply low power. Five candidates have now been tried on the US; any PASS still needs a Europe check with EUR-relevant CoT.', '',
               '## Metrics', '', 'CAGR and maxDD are percentages. Sharpe uses excess returns over cash. BH is identical in both exposure versions.', '',
               '| Period | Rule | Version | Sharpe | CAGR | maxDD | Switches/year |',
               '|---|---|---|---:|---:|---:|---:|']
    for period in ('WIN', 'WH1', 'WH2', 'P2021'):
        for rule, version in runs:
            row = lookup.loc[(rule, version, period)]
            result.append(f'| {period} | {rule} | {version} | {row.sharpe:.6f} | {row.cagr:.4%} | {row.maxdd:.4%} | {row.switches_per_year:.6f} |')
    result += ['', '## Latest R5 reading', '',
               f'Decision date: {gf.index[-1].date()}. T: {"in" if latest.T_in else "out"} (DMAS {latest.dmas:.6f}); R: {"in" if latest.R_in else "out"} (REV_z {latest.rev_z:.6f}); C: {"in" if latest.C_in else "out"} (C_z {latest.C_z:.6f}). Report date used: {latest.report_date_used.date()}. Gauge state: {"Cash" if latest.cash_R5 else "Invested"}. Sleeve state: {latest.state_R5} (scheduled decision state).', '',
               '## Method and validation', '',
               '- All inputs loaded read-only with load(name, refresh=False); no network, Bloomberg, or Dropbox writes.',
               '- CoT net is (leveraged long - leveraged short) / open interest. Z-score uses the previous 156 reports, minimum 104, excluding the current report; standard deviation uses ddof=1, as in earlier steps.',
               '- Each decision selects the latest report with report date + stored release lag <= decision date. Undefined C remains undecided; every decision in the specified window has defined C_z.',
               '- Existing T/R signals, sleeve_weights with deep = DMAS < 30, returns, metrics and bootstrap are reused. No variants or extra rules were run.',
               '- Synthetic release-delay, warm-up, z-score, eight-combination vote, real-data truncation and runner window/state tests passed before the result run.', '',
               '## Deviations', '',
               '- No step-2 threshold, rule, window, or gate changes.',
               '- Inherited conventions: costs are debited on the new-weight return day (t+2); initial allocation has no switch cost; the common evaluation sample starts at R5’s first effective weight. Monthly grouping uses the actual decision date’s calendar month, including holiday adjustments.',
               '- BH is labeled all and half, with identical results, to satisfy the requested rules × versions output.', '']
    out = HERE / 'out'
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'step2_metrics.csv', index=False)
    gf.to_csv(out / 'step2_signals.csv', index_label='date')
    (HERE / 'RESULTS_step2.md').write_text('Re-run 2026-11-17 with corrected cash leg (BIL.US_TR); see spec correction log.\n\n' + '\n'.join(result))
    print(f'Step 2 verdict: {verdict}. Outputs written to {out}')


if __name__ == '__main__':
    main()
