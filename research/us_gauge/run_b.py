"""Breadth thrust re-entry B — draft pre-registration in PREREG_B_draft.md (not yet in the spec).

ONE-RUN TEST. Do not call main() before the pre-registration is committed and the breadth series
reaches the end of the sample. Everything above main() is pure and tested on synthetic data only.

Thrust (Deemer "breakaway momentum", walterdeemer.com/bam.htm): 10-day total NYSE advances greater
than 1.97 times 10-day total NYSE declines. Zweig's thrust is computed for the report only.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from research.us_gauge import run_vol
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals
from research.us_gauge.run_p import decision_days, episodes as f_episodes
from research.us_gauge.run_c import floor_weights, sharpe, window_dd
from research.us_gauge.run_fast import states, to_weights, v2_from_states
from research.us_gauge.run_s import shock_episodes
from research.us_gauge.trading import net_returns
from research.us_gauge.evaluate import metrics, bootstrap_not_better
from research.us_gauge.stress_periods import stats
from research.us_gauge.combo_f import WINDOWS

HERE = Path(__file__).resolve().parent
FREQS = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI']
N_SUM, THRESHOLD = 10, 1.97            # Deemer
L_WINDOW, MAX_DAYS = 63, 126           # stop reference window; time limit (as S)
START, H1_END = '1965-04-01', '1990-12-31'
BEARS = [('2000-03-01', '2003-03-31'), ('2007-10-01', '2009-06-30'), ('2022-01-01', '2022-12-31')]
DROP = [('2020-02-19', '2020-08-31'), ('2025-02-19', '2025-06-30')]
FREE_ADV, FREE_DEC = 'UNICORN_NYSE_ADV', 'UNICORN_NYSE_DEC'      # 1965-03-01 .. 2020-02-10
BBG_ADV, BBG_DEC = 'BBG_NYSE_ADV', 'BBG_NYSE_DEC'                # user download, needed from 2020-02-11
S_SIZE = 'live'                        # adopted form (spec 6sl passed, commit e4b73b6)


def breadth(need_until=None) -> pd.DataFrame:
    """Advances / declines: free series to its last day, Bloomberg after it. Overlap is returned for the report."""
    with read_only_external():
        adv, dec = load(FREE_ADV, refresh=False), load(FREE_DEC, refresh=False)
        try:
            badv, bdec = load(BBG_ADV, refresh=False), load(BBG_DEC, refresh=False)
        except Exception:
            badv = bdec = pd.Series(dtype=float)
    free = pd.DataFrame({'adv': adv, 'dec': dec}).dropna()
    bbg = pd.DataFrame({'adv': badv, 'dec': bdec}).dropna()
    out = pd.concat([free, bbg.loc[bbg.index > free.index[-1]]]) if len(bbg) else free
    out = out[(out.adv > 0) & (out.dec > 0)]
    out.attrs['overlap'] = free.join(bbg, how='inner', lsuffix='_free', rsuffix='_bbg')
    if need_until is not None and out.index[-1] < pd.Timestamp(need_until):
        raise RuntimeError(f'Breadth ends {out.index[-1].date()}; need {need_until}. Download {BBG_ADV}/{BBG_DEC} first.')
    return out


def deemer_ratio(adv: pd.Series, dec: pd.Series, n=N_SUM) -> pd.Series:
    """n-day total advances / n-day total declines, over the n most recent valid breadth days."""
    return (adv.rolling(n, min_periods=n).sum() / dec.rolling(n, min_periods=n).sum()).rename('ratio')


def thrust_days(ratio: pd.Series, threshold=THRESHOLD) -> pd.DatetimeIndex:
    """Days on which the ratio crosses above the threshold (previous valid day at or below it)."""
    up = ratio.gt(threshold) & ratio.shift(1).le(threshold)
    return ratio.index[up.to_numpy()]


def zweig_days(adv: pd.Series, dec: pd.Series) -> pd.DatetimeIndex:
    """Report only. 10-day exponential average of A/(A+D) crosses above 0.615 having been below 0.40 in the prior 10 days."""
    e = (adv / (adv + dec)).ewm(span=10, adjust=False).mean()
    was_low = e.lt(.40).shift(1).rolling(10, min_periods=1).max().eq(1)
    return e.index[(e.gt(.615) & e.shift(1).le(.615) & was_low).to_numpy()]


def thrust_episodes(px: pd.Series, base_w: pd.Series, thrusts) -> pd.DataFrame:
    """Daily state machine. A thrust at close t starts an episode when the base weight in force is below 1.

    L = lowest close of the L_WINDOW closes ending t. End (first that holds, checked every close after t):
    base weight in force is 1, close below L, or MAX_DAYS trading days after t. Thrusts during an episode are
    ignored; the next episode needs a new thrust after the end.
    """
    p, n, idx = px.to_numpy(), len(px), px.index
    bw = base_w.reindex(idx).to_numpy()
    pos = idx.searchsorted(pd.DatetimeIndex(thrusts))
    rows, free_from = [], 0
    for k, t in zip(pos, pd.DatetimeIndex(thrusts)):
        if k >= n or idx[k] != t or k < free_from or k < L_WINDOW - 1 or np.isnan(bw[k]) or bw[k] >= 1:
            continue
        L = p[k - L_WINDOW + 1:k + 1].min()
        end, reason = n - 1, 'open'
        for j in range(k + 1, n):
            if bw[j] >= 1:
                end, reason = j, 'base invested'
                break
            if p[j] < L:
                end, reason = j, 'stop'
                break
            if j == k + MAX_DAYS:
                end, reason = j, 'time'
                break
        rows.append((idx[k], idx[end], reason, L, k, end))
        free_from = end + 1
    return pd.DataFrame(rows, columns=['entry', 'end', 'reason', 'L', 'i_entry', 'i_end'])


def placebo_spans(rng, n_ep, px, base_w):
    """Each episode moved to a random day with base weight 0; same L window, stop, base-invested end and time limit."""
    p = px.to_numpy()
    bw = base_w.reindex(px.index).to_numpy()
    elig = np.flatnonzero((bw == 0) & (np.arange(len(p)) >= L_WINDOW - 1))
    spans = []
    for e in rng.choice(elig, size=n_ep, replace=True):
        L = p[e - L_WINDOW + 1:e + 1].min()
        end = len(p) - 1
        for j in range(e + 1, len(p)):
            if bw[j] >= 1 or p[j] < L or j == e + MAX_DAYS:
                end = j
                break
        spans.append((e, end))
    return spans


def r0_pair(inp, dmas, thrusts, freq, start, cost=10.):
    d = decision_days(inp.index, start, freq)
    m = dmas.reindex(d)
    wb = to_weights(inp.index, states(d, m < 45, m < 30, m, None))
    ep = thrust_episodes(inp.px, wb, thrusts)
    fl = floor_weights(inp.index, list(zip(ep.i_entry, ep.i_end)), 1.)
    wx = np.maximum(wb, fl).where(wb.notna())
    rb = net_returns(wb, inp.eq_ret, inp.cash_ret, cost).where(wb.notna()).dropna()
    rx = net_returns(wx, inp.eq_ret, inp.cash_ret, cost).reindex(rb.index)
    return rb, rx, wb, ep


def gates(inp, dmas, thrusts, start=START, h_split=H1_END, bears=BEARS, drop=DROP, n_placebo=1000, n_boot=2000):
    """G1-G6 on R0+B vs R0 (Tuesday base). Same construction as run_s.gates."""
    rb, rx, wb, ep = r0_pair(inp, dmas, thrusts, 'W-TUE', start)
    cash = inp.cash_ret.reindex(rb.index)
    checks = []
    for nm, a, b in [('FULL', None, None), ('H1', None, h_split), ('H2', pd.Timestamp(h_split) + pd.Timedelta(days=1), None)]:
        s0, s1 = sharpe(rb.loc[a:b], cash.loc[a:b]), sharpe(rx.loc[a:b], cash.loc[a:b])
        checks.append((f'G1 {nm} Sharpe R0+B > R0', f'{s1:.4f} > {s0:.4f}', s1 > s0))
    wd = []
    for f in FREQS:
        b_, x_, _, _ = r0_pair(inp, dmas, thrusts, f, start)
        c_ = inp.cash_ret.reindex(b_.index)
        wd.append((f, sharpe(b_, c_), sharpe(x_, c_)))
    mb, mx = np.mean([w[1] for w in wd]), np.mean([w[2] for w in wd])
    checks.append(('G1 weekday-mean Sharpe', f'{mx:.4f} > {mb:.4f} ({", ".join(f"{f[-3:]} {x:.3f}/{b:.3f}" for f, b, x in wd)})', mx > mb))
    d0, d1 = metrics(rb, cash)['maxdd'], metrics(rx, cash)['maxdd']
    checks.append(('G2 FULL maxDD >= R0 - 0.02', f'{d1:.4f} vs {d0:.4f}', d1 >= d0 - .02))
    for a, b in bears:
        if len(rb.loc[a:b]) > 20:
            x0, x1 = window_dd(rb, a, b), window_dd(rx, a, b)
            checks.append((f'G2 maxDD {a[:7]}..{b[:7]}', f'{x1:.4f} vs {x0:.4f}', x1 >= x0 - .02))
    mask = pd.Series(True, index=rb.index)
    for a, b in drop:
        mask.loc[a:b] = False
    s0, s1 = sharpe(rb[mask], cash[mask]), sharpe(rx[mask], cash[mask])
    checks.append(('G3 Sharpe without 2020 and 2025', f'{s1:.4f} > {s0:.4f}', s1 > s0))
    boot = bootstrap_not_better(rx, rb, cash, block=63, n=n_boot, seed=7)
    checks.append(('G4 bootstrap not-better < 0.10', f'{boot:.4f}', boot < .10))
    rng = np.random.default_rng(7)
    base_sh, gains = sharpe(rb, cash), []
    eq, cr = inp.eq_ret.to_numpy(), inp.cash_ret.to_numpy()
    keep, wbn = wb.notna().to_numpy(), wb.to_numpy()
    for _ in range(n_placebo if len(ep) else 0):
        f = floor_weights(inp.index, placebo_spans(rng, len(ep), inp.px, wb), 1.).to_numpy()
        w = np.where(keep, np.maximum(wbn, f), np.nan)
        dw = np.abs(np.diff(w, prepend=np.nan))
        dw[np.isnan(dw)] = 0.
        x = (w * eq + (1 - w) * cr - dw * 10 / 1e4)[keep] - cr[keep]
        gains.append(x.mean() / x.std(ddof=1) * np.sqrt(252) - base_sh)
    gains = np.array(gains) if gains else np.array([np.inf])
    real = sharpe(rx, cash) - base_sh
    checks.append(('G5 placebo: real gain > 90th pct', f'{real:+.4f} vs {np.quantile(gains, .9):+.4f} (placebo share >= real {(gains >= real).mean():.3f})',
                   bool(real > np.quantile(gains, .9))))
    pnl = []
    for r in ep.itertuples():
        days = inp.index[r.i_entry + 2:r.i_end + 2].intersection(rb.index)
        pnl.append((1 + rx.loc[days]).prod() / (1 + rb.loc[days]).prod() - 1 if len(days) else np.nan)
    ep['pnl_vs_R0'] = pnl
    checks.append(('G6 more than half of episodes add value', f'{(ep.pnl_vs_R0 > 0).sum()} of {len(ep)}', bool(len(ep) and (ep.pnl_vs_R0 > 0).mean() > .5)))
    return dict(checks=checks, rb=rb, rx=rx, ep=ep)


def product_weights(inputs, thrusts, s_size=S_SIZE):
    """V2+F+S (the product) and the same plus B at the live V2 size. Returns (w_without_B, w_with_B, B episodes on R1)."""
    idx = inputs.index
    gf = build_signals(inputs)
    s1 = states(gf.loc['1993-04-01':].index, gf.cash_R1, gf.deep, gf.dmas, None)
    w1, w2 = to_weights(idx, s1), v2_from_states(inputs, s1)
    epf = f_episodes(idx, s1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    w2F = np.maximum(w2, floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)).where(w2.notna())
    v = run_vol.target_weight(inputs.eq_ret)
    live = v.shift(2)                                   # decided t, in force t+2
    sw = pd.Series(0., index=idx)
    for r in shock_episodes(inputs.px, w1).itertuples():
        a, b = r.i_entry + 2, min(r.i_end + 2, len(idx))
        sw.iloc[a:b] = live.iloc[a:b].to_numpy() if s_size == 'live' else v.iloc[r.i_entry]
    base = np.maximum(w2F, sw).where(w2F.notna())
    epb = thrust_episodes(inputs.px, w1, thrusts)
    bw = pd.Series(0., index=idx)
    for r in epb.itertuples():
        a, b = r.i_entry + 2, min(r.i_end + 2, len(idx))
        bw.iloc[a:b] = live.iloc[a:b].to_numpy()
    return base, np.maximum(base, bw).where(base.notna()), epb


def main(argv):
    if '--one-run' not in argv:
        raise SystemExit('B is a one-run test. Commit the pre-registration first, then run with --one-run.')
    with read_only_external():
        inputs = load_inputs()
    br = breadth(need_until=inputs.index[-1] - pd.Timedelta(days=7))
    inputs = inputs.loc[:br.index[-1]]
    dmas = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0]
    ratio = deemer_ratio(br.adv, br.dec)
    thrusts = thrust_days(ratio)
    out = gates(inputs, dmas, thrusts)
    base, withb, epb = product_weights(inputs, thrusts)
    R = lambda w: net_returns(w, inputs.eq_ret, inputs.cash_ret).loc['1993-06-01':]
    r0p, r1p = R(base), R(withb)
    cashp = inputs.cash_ret.reindex(r0p.index)
    mp0, mp1 = metrics(r0p, cashp), metrics(r1p, cashp)
    out['checks'].append(('G7 product: V2+F+S+B Sharpe >= V2+F+S and maxDD >= -0.02',
                          f'{mp1["sharpe"]:.4f} vs {mp0["sharpe"]:.4f}; {mp1["maxdd"]:.4f} vs {mp0["maxdd"]:.4f}',
                          bool(mp1['sharpe'] >= mp0['sharpe'] and mp1['maxdd'] >= mp0['maxdd'] - .02)))
    verdict = all(c[2] for c in out['checks'])
    ov = br.attrs['overlap']
    L = ['# Breadth thrust re-entry B', '', f'Verdict: **{"PASS" if verdict else "FAIL"}**', '',
         f'Primary: R0+B vs R0, S&P 500, Tuesday base, {out["rb"].index[0].date()} → {out["rb"].index[-1].date()}. '
         f'Thrust days: {len(thrusts)}. Episodes: {len(out["ep"])}. S size in the product: {S_SIZE}.',
         f'Breadth: {FREE_ADV}/{FREE_DEC} to {pd.Timestamp("2020-02-10").date()}, then {BBG_ADV}/{BBG_DEC}. Overlap days: {len(ov)}'
         + (f'; median abs difference of the advance share {((ov.adv_free / (ov.adv_free + ov.dec_free)) - (ov.adv_bbg / (ov.adv_bbg + ov.dec_bbg))).abs().median():.4f}.' if len(ov) else '.'),
         '', '| Gate | Evidence | Result |', '|---|---|---|']
    L += [f'| {n} | {e} | {"PASS" if p else "FAIL"} |' for n, e, p in out['checks']]
    L += ['', '| Period | Rule | Sharpe | CAGR | Max DD |', '|---|---|---:|---:|---:|']
    for nm, a, b in [('FULL', None, None), ('H1 to 1990', None, H1_END), ('H2 from 1991', '1991-01-01', None), ('P2021', '2021-01-01', None)]:
        for rule, r in (('R0', out['rb']), ('R0+B', out['rx'])):
            m = metrics(r.loc[a:b], inputs.cash_ret.reindex(r.loc[a:b].index))
            L.append(f'| {nm} | {rule} | {m["sharpe"]:.3f} | {m["cagr"]:.2%} | {m["maxdd"]:.1%} |')
    L += ['', '| Thrust (entry) | End | Reason | Stop level L | P&L vs R0 |', '|---|---|---|---:|---:|']
    L += [f'| {r.entry.date()} | {r.end.date()} | {r.reason} | {r.L:.2f} | {r.pnl_vs_R0:+.2%} |' for r in out['ep'].itertuples()]
    ret = pd.DataFrame({'Buy & hold': inputs.eq_ret, 'V2+F+S': net_returns(base, inputs.eq_ret, inputs.cash_ret),
                        'V2+F+S+B': net_returns(withb, inputs.eq_ret, inputs.cash_ret)}).loc['1999-02-01':]
    cashk = inputs.cash_ret.reindex(ret.index)
    L += ['', '## Product by window (total return / Sharpe / worst fall)', '', '| Window | ' + ' | '.join(ret.columns) + ' |', '|---|' + '---:|' * 3]
    for name, a, b in WINDOWS:
        cells = []
        for c in ret:
            tot, sh, dd = stats(ret[c].loc[a:b], cashk)
            cells.append(f'{tot:+.1%} / {sh:.2f} / {dd:.1%}')
        L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    L += ['', 'B episodes on the R1 base: ' + ', '.join(f'{r.entry.date()}→{r.end.date()} ({r.reason})' for r in epb.itertuples()), '',
          '## Report only (written after the verdict)', '',
          'Zweig thrust days: ' + ', '.join(str(d.date()) for d in zweig_days(br.adv, br.dec)), '']
    (HERE / 'RESULTS_b.md').write_text('\n'.join(L))
    out['ep'].to_csv(HERE / 'out/b_episodes.csv', index=False)
    print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1:])
