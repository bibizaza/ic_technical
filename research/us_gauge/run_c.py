"""Capitulation floor C — pre-registered in the gauge spec section 6x (commit 69c7df4)."""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_k1
from research.us_gauge.data import load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals, _simulate
from research.us_gauge.gauge import dmas_on
from research.us_gauge.trading import sleeve_weights, net_returns
from research.us_gauge.evaluate import metrics, bootstrap_not_better

HERE = Path(__file__).resolve().parent
FLOOR = .50
START = '1955-01-01'
H1_END = '1990-12-31'
BEARS = [('2000-03-01', '2003-03-31'), ('2007-10-01', '2009-06-30')]
DROP = [('2020-02-19', '2020-08-31'), ('2025-02-19', '2025-06-30')]


def zscore(px: pd.Series) -> pd.Series:
    r = np.log(px).diff()
    s = r.rolling(252, min_periods=252).std(ddof=1).shift(21)
    return (np.log(px / px.shift(21)) / (s * np.sqrt(21))).rename('Z')


def episodes(px: pd.Series, z_arm=-3.0) -> pd.DataFrame:
    """Idle/Armed/On state machine on daily closes; returns one row per On episode."""
    p, z, idx = px.to_numpy(), zscore(px).to_numpy(), px.index
    state, blocked, rows = 'idle', -1, []
    for k in range(len(p)):
        if state == 'idle':
            if k > blocked and not np.isnan(z[k]) and z[k] <= z_arm:
                state, L, last_z, arm = 'armed', p[k], k, k
        elif state == 'armed':
            if not np.isnan(z[k]) and z[k] <= z_arm:
                last_z = k
            L = min(L, p[k])
            if k >= 5 and p[k] > p[k - 5] and p[k] > L:
                state, e = 'on', k
            elif k - last_z >= 21:
                state = 'idle'
        else:
            reason = 'stop' if p[k] < L else 'time' if k == e + 63 else None
            if reason:
                rows.append((idx[arm], idx[e], idx[k], reason, L, arm, e, k))
                state, blocked = 'idle', k
    if state == 'on':
        rows.append((idx[arm], idx[e], idx[-1], 'open', L, arm, e, len(p) - 1))
    return pd.DataFrame(rows, columns=['arm', 'entry', 'end', 'reason', 'L', 'i_arm', 'i_entry', 'i_end'])


def floor_weights(index, spans, floor):
    """Floor earns from entry+2 through end+1 (decide t, trade t+1, earn t+2)."""
    f = np.zeros(len(index))
    for e, end in spans:
        f[e + 2:min(end + 2, len(index))] = floor
    return pd.Series(f, index=index)


def sharpe(r, cash):
    x = (r - cash).dropna()
    return x.mean() / x.std(ddof=1) * np.sqrt(252)


def base_r0(inputs, dmas, freq):
    tues = pd.date_range(START, inputs.index[-1], freq=freq)
    pos = inputs.index.searchsorted(tues, side='right') - 1
    dates = inputs.index[pos[pos >= 0]].drop_duplicates()
    d = dmas.reindex(dates)
    return sleeve_weights(inputs.index, dates, d < 45, d < 30, 0.)


def run_pair(w_base, floor, inputs, cost=10.):
    w_c = np.maximum(w_base, floor).where(w_base.notna())
    rb = net_returns(w_base, inputs.eq_ret, inputs.cash_ret, cost).where(w_base.notna()).dropna()
    rc = net_returns(w_c, inputs.eq_ret, inputs.cash_ret, cost).where(w_base.notna()).dropna()
    return rb, rc, w_c


def episode_pnl(ep, rb, rc, index):
    out = []
    for r in ep.itertuples():
        days = index[r.i_entry + 2:r.i_end + 2].intersection(rb.index)
        out.append((1 + rc.loc[days]).prod() / (1 + rb.loc[days]).prod() - 1 if len(days) else np.nan)
    return pd.Series(out, index=ep.index)


def placebo(ep, w_base, px, inputs, rb, cash, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    idx, p = inputs.index, px.to_numpy()
    sample = w_base.dropna()
    eligible = np.flatnonzero((w_base < .5).to_numpy() & w_base.notna().to_numpy())
    eligible = eligible[eligible >= 20]
    base_sh = sharpe(rb, cash)
    eq, cr = inputs.eq_ret.to_numpy(), inputs.cash_ret.to_numpy()
    wb = w_base.to_numpy()
    keep = w_base.notna().to_numpy()
    gains = []
    for _ in range(n):
        spans = []
        for e in rng.choice(eligible, size=len(ep), replace=True):
            L = p[e - 20:e + 1].min()
            end = len(p) - 1
            for k in range(e + 1, len(p)):
                if p[k] < L or k == e + 63:
                    end = k
                    break
            spans.append((e, end))
        f = floor_weights(idx, spans, FLOOR).to_numpy()
        w = np.where(keep, np.maximum(wb, f), np.nan)
        dw = np.abs(np.diff(w, prepend=np.nan))
        dw[np.isnan(dw)] = 0.
        r = (w * eq + (1 - w) * cr - dw * 10 / 1e4)[keep]
        x = r - cr[keep]
        gains.append(x.mean() / x.std(ddof=1) * np.sqrt(252) - base_sh)
    return np.array(gains)


def window_dd(r, a, b):
    sub = r.loc[a:b]
    w = (1 + sub).cumprod()
    return (w / w.cummax().clip(lower=1) - 1).min()


def main():
    with read_only_external():
        inputs = load_inputs()
    px = inputs.px
    days = inputs.loc[START:].index
    cache = HERE / 'out/dmas_daily_1955.csv'
    if cache.exists():
        dmas = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        dmas = dmas_on(px, days)
        dmas.to_csv(cache)
    ep_all = episodes(px)
    ep = ep_all[ep_all.entry >= pd.Timestamp(START)].reset_index(drop=True)
    floor = floor_weights(inputs.index, list(zip(ep.i_entry, ep.i_end)), FLOOR)

    w0 = base_r0(inputs, dmas, 'W-TUE')
    rb, rc, wc = run_pair(w0, floor, inputs)
    cash = inputs.cash_ret.reindex(rb.index)
    checks = []
    for name, a, b in [('FULL', None, None), ('H1', None, H1_END), ('H2', '1991-01-01', None)]:
        sb, sc = sharpe(rb.loc[a:b], cash.loc[a:b]), sharpe(rc.loc[a:b], cash.loc[a:b])
        checks.append((f'G1 {name} Sharpe C > R0', f'{sc:.4f} > {sb:.4f}', sc > sb))
    wd = []
    for freq in ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI']:
        wb_ = base_r0(inputs, dmas, freq)
        b_, c_, _ = run_pair(wb_, floor, inputs)
        cs = inputs.cash_ret.reindex(b_.index)
        wd.append((freq, sharpe(b_, cs), sharpe(c_, cs)))
    mb, mc = np.mean([x[1] for x in wd]), np.mean([x[2] for x in wd])
    checks.append(('G1 weekday-mean Sharpe C > R0', f'{mc:.4f} > {mb:.4f} ({", ".join(f"{f[-3:]} {c:.3f}/{b:.3f}" for f, b, c in wd)})', mc > mb))
    db, dc = metrics(rb, cash)['maxdd'], metrics(rc, cash)['maxdd']
    checks.append(('G2 FULL maxDD C >= R0 - 0.02', f'{dc:.4f} vs {db:.4f}', dc >= db - .02))
    for a, b in BEARS:
        xb, xc = window_dd(rb, a, b), window_dd(rc, a, b)
        checks.append((f'G2 maxDD {a[:7]}..{b[:7]}', f'{xc:.4f} vs {xb:.4f}', xc >= xb - .02))
    mask = pd.Series(True, index=rb.index)
    for a, b in DROP:
        mask.loc[a:b] = False
    sb, sc = sharpe(rb[mask], cash[mask]), sharpe(rc[mask], cash[mask])
    checks.append(('G3 Sharpe without 2020 and 2025 rebounds', f'{sc:.4f} > {sb:.4f}', sc > sb))
    boot = bootstrap_not_better(rc, rb, cash, block=63, n=2000, seed=7)
    checks.append(('G4 bootstrap not-better < 0.10', f'{boot:.4f}', boot < .10))
    gains = placebo(ep, w0, px, inputs, rb, cash)
    real_gain = sharpe(rc, cash) - sharpe(rb, cash)
    p90 = np.quantile(gains, .9)
    checks.append(('G5 real gain > placebo 90th pct', f'{real_gain:+.4f} vs {p90:+.4f} (placebo share >= real: {(gains >= real_gain).mean():.3f})', real_gain > p90))
    ep['pnl_vs_R0'] = episode_pnl(ep, rb, rc, inputs.index)
    ep['R0_weight_at_entry'] = [w0.iloc[min(i + 2, len(w0) - 1)] for i in ep.i_entry]
    share = (ep.pnl_vs_R0 > 0).mean()
    checks.append(('G6 more than half of episodes add value', f'{(ep.pnl_vs_R0 > 0).sum()} of {len(ep)} ({share:.0%})', share > .5))
    verdict = all(c[2] for c in checks)

    # metrics table, 10 bp and 25 bp
    mrows = []
    for cost in (10., 25.):
        b_, c_, _ = run_pair(w0, floor, inputs, cost)
        for name, a, b in [('FULL', None, None), ('H1', None, H1_END), ('H2', '1991-01-01', None), ('P2021', '2021-01-01', None)]:
            for rule, r in (('R0', b_), ('R0+C', c_)):
                m = metrics(r.loc[a:b], inputs.cash_ret.reindex(r.loc[a:b].index))
                mrows.append(f'| {cost:.0f} bp | {name} | {rule} | {m["sharpe"]:.3f} | {m["cagr"]:.2%} | {m["maxdd"]:.1%} |')

    # secondary: R1 and V2, 1993+ (report only)
    gf = build_signals(inputs)
    w1, _ = _simulate(inputs, gf, 'R1', 0.)
    w2, _, _ = run_k1.v2_weights(inputs)
    srows = []
    windows = [('FULL 1993+', None, None), ('P2021', '2021-01-01', None), ('2008 crisis', '2007-10-01', '2009-06-30'),
               ('COVID 19 Feb-31 Aug 2020', '2020-02-20', '2020-08-31'), ('2022', '2022-01-01', '2022-12-31'),
               ('Liberation Day 19 Feb-30 Jun 2025', '2025-02-20', '2025-06-30')]
    for base_name, wb_ in (('R1', w1), ('V2', w2)):
        b_, c_, _ = run_pair(wb_.loc['1993-04-01':].reindex(inputs.index), floor, inputs)
        for name, a, b in windows:
            for rule, r in ((base_name, b_), (base_name + '+C', c_)):
                sub = r.loc[a:b]
                m = metrics(sub, inputs.cash_ret.reindex(sub.index))
                tot = (1 + sub).prod() - 1
                srows.append(f'| {name} | {rule} | {m["sharpe"]:.3f} | {tot:+.1%} | {m["maxdd"]:.1%} |')
    # sensitivity, after the verdict (report only)
    sens = []
    for z_arm in (-2.5, -3.0, -3.5):
        e2 = episodes(px, z_arm)
        e2 = e2[e2.entry >= pd.Timestamp(START)]
        for fl in (.25, .50, 1.):
            f2 = floor_weights(inputs.index, list(zip(e2.i_entry, e2.i_end)), fl)
            b_, c_, _ = run_pair(w0, f2, inputs)
            cs = inputs.cash_ret.reindex(b_.index)
            sens.append(f'| {z_arm} | {fl:.2f} | {len(e2)} | {sharpe(c_, cs):.3f} | {sharpe(b_, cs):.3f} | {metrics(c_, cs)["maxdd"]:.1%} |')

    rep = ['# Capitulation floor C — R0+C vs R0, S&P 500 1955-2026', '',
           f'Verdict: **{"PASS (qualifies for Europe only)" if verdict else "FAIL"}**', '',
           f'Sample: {rb.index[0].date()} → {rb.index[-1].date()} ({len(rb)} days). Episodes: {len(ep)}.', '',
           '## Gates', '', '| Gate | Evidence | Result |', '|---|---|---|']
    rep += [f'| {n} | {e} | {"PASS" if p else "FAIL"} |' for n, e, p in checks]
    rep += ['', '## Metrics (primary)', '', '| Cost | Period | Rule | Sharpe | CAGR | Max DD |', '|---|---|---|---:|---:|---:|'] + mrows
    rep += ['', '## Episodes', '', '| Armed | Entry | End | Reason | R0 weight at entry | P&L vs R0 |', '|---|---|---|---|---:|---:|']
    rep += [f'| {r.arm.date()} | {r.entry.date()} | {r.end.date()} | {r.reason} | {r.R0_weight_at_entry:.2f} | {r.pnl_vs_R0:+.2%} |' for r in ep.itertuples()]
    rep += ['', '## Secondary (report only): R1 and V2 with the floor, 1993-2026', '',
            '| Window | Rule | Sharpe | Total | Max DD |', '|---|---|---:|---:|---:|'] + srows
    rep += ['', '## Sensitivity (report only, written after the verdict)', '',
            '| Z arm | Floor | Episodes | Sharpe R0+C | Sharpe R0 | Max DD R0+C |', '|---|---|---:|---:|---:|---:|'] + sens
    rep += ['', '## Method', '', 'As pre-registered in spec section 6x plus its implementation notes. DMAS computed on every trading day from 1955 (cached in out/dmas_daily_1955.csv) and read on decision days. Placebo: 1,000 draws, default_rng(7), episode count equal to the real one, entries drawn with replacement from days where the R0 weight in force is < 0.50. Inputs read-only.', '']
    (HERE / 'RESULTS_c.md').write_text('\n'.join(rep))
    ep.to_csv(HERE / 'out/c_episodes.csv', index=False)
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
