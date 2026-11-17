"""Faster re-entry X — pre-registered in the gauge spec section 6z (commit c3f8de0)."""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_k1, run_s2, run_vol
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals
from research.us_gauge.run_m1 import build as build_m1
from research.us_gauge.run_p import decision_days, episodes
from research.us_gauge.run_c import floor_weights, sharpe, window_dd
from research.us_gauge.trading import net_returns
from research.us_gauge.volsize import daily_weights
from research.us_gauge.evaluate import metrics, bootstrap_not_better
from research.us_gauge.stress_periods import stats
from research.us_gauge.combo_f import WINDOWS

HERE = Path(__file__).resolve().parent
START, H1_END = '1955-01-01', '1990-12-31'
BEARS = [('2000-03-01', '2003-03-31'), ('2007-10-01', '2009-06-30'), ('2022-01-01', '2022-12-31')]
DROP = [('2020-02-19', '2020-08-31'), ('2025-02-19', '2025-06-30')]
FREQS = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI']


def states(dates, cash, deep, dmas, reentry=None, forced=None):
    """Decision-date state (1 invested, 0 cash). reentry=None is the base rule; forced = placebo dates."""
    out, cur, last = [], np.nan, None
    for d in dates:
        m = d.to_period('M')
        if m != last:
            cur = 0. if cash.loc[d] else 1.
        elif cur == 1. and cash.loc[d] and deep.loc[d]:
            cur = 0.
        elif cur == 0. and ((forced is not None and d in forced) or
                            (forced is None and reentry is not None and dmas.loc[d] >= reentry)):
            cur = 1.
        last = m
        out.append(cur)
    return pd.Series(out, index=dates)


def to_weights(index, st):
    ch = pd.Series(np.nan, index=index)
    pos = index.get_indexer(st.index) + 2
    ok = pos < len(index)
    ch.iloc[pos[ok]] = st.to_numpy()[ok]
    return ch.ffill()


def early_reentries(st, dates):
    first = pd.Series(dates.to_period('M'), index=dates)
    first = first.ne(first.shift())
    up = st.eq(1) & st.shift().eq(0) & ~first
    return list(st.index[up])


def placebo_dates(rng, st_x, st_b, cash, events):
    """For each real early re-entry, a random non-first-of-month date in the same spell before base re-entry."""
    dates = st_x.index
    first = pd.Series(dates.to_period('M'), index=dates)
    first = first.ne(first.shift())
    out = set()
    for e in events:
        k = dates.get_loc(e)
        s = k
        while s > 0 and st_x.iloc[s - 1] == 0:
            s -= 1
        j = k
        while j < len(dates) and not (first.iloc[j] and not cash.iloc[j]):
            j += 1
        cand = [dates[i] for i in range(s, j) if not first.iloc[i]]
        out.add(cand[rng.integers(len(cand))])
    return out


def pair(index, st_b, st_x, inputs, cost=10.):
    wb, wx = to_weights(index, st_b), to_weights(index, st_x)
    rb = net_returns(wb, inputs.eq_ret, inputs.cash_ret, cost).where(wb.notna()).dropna()
    rx = net_returns(wx, inputs.eq_ret, inputs.cash_ret, cost).where(wx.notna()).dropna()
    return rb, rx.reindex(rb.index), wb, wx


def v2_from_states(inputs, st):
    v = run_vol.target_weight(inputs.eq_ret)
    usable = st.index[inputs.index.get_indexer(st.index) + 2 < len(inputs)]
    return daily_weights(inputs.index, usable, v, st)


def main():
    panel, c5, _, macro, _ = build_m1()
    with read_only_external():
        inputs = load_inputs()
    etf = run_s2.load_etfs()
    _, held_c5 = run_s2.tilt_returns(c5, etf, panel.r_rf)
    _, held_m1 = run_s2.tilt_returns(macro, etf, panel.r_rf)
    with read_only_external():
        levels = pd.DataFrame({t: load(t + '.US_TR', refresh=False).sort_index() for t in etf.columns})
    idx = inputs.index
    dmas = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0]

    def r0_states(freq, reentry):
        d = decision_days(idx, START, freq)
        m = dmas.reindex(d)
        return states(d, m < 45, m < 30, m, reentry)

    sb, sx = r0_states('W-TUE', None), r0_states('W-TUE', 60)
    rb, rx, wb, wx = pair(idx, sb, sx, inputs)
    cash = inputs.cash_ret.reindex(rb.index)
    checks = []
    for nm, a, b in [('FULL', None, None), ('H1', None, H1_END), ('H2', '1991-01-01', None)]:
        s0, s1 = sharpe(rb.loc[a:b], cash.loc[a:b]), sharpe(rx.loc[a:b], cash.loc[a:b])
        checks.append((f'G1 {nm} Sharpe R0+X > R0', f'{s1:.4f} > {s0:.4f}', s1 > s0))
    wd = []
    for f in FREQS:
        b_, x_, _, _ = pair(idx, r0_states(f, None), r0_states(f, 60), inputs)
        c_ = inputs.cash_ret.reindex(b_.index)
        wd.append((f, sharpe(b_, c_), sharpe(x_, c_)))
    mb, mx = np.mean([w[1] for w in wd]), np.mean([w[2] for w in wd])
    checks.append(('G1 weekday-mean Sharpe', f'{mx:.4f} > {mb:.4f} ({", ".join(f"{f[-3:]} {x:.3f}/{b:.3f}" for f, b, x in wd)})', mx > mb))
    d0, d1 = metrics(rb, cash)['maxdd'], metrics(rx, cash)['maxdd']
    checks.append(('G2 FULL maxDD >= R0 - 0.02', f'{d1:.4f} vs {d0:.4f}', d1 >= d0 - .02))
    for a, b in BEARS:
        x0, x1 = window_dd(rb, a, b), window_dd(rx, a, b)
        checks.append((f'G2 maxDD {a[:7]}..{b[:7]}', f'{x1:.4f} vs {x0:.4f}', x1 >= x0 - .02))
    mask = pd.Series(True, index=rb.index)
    for a, b in DROP:
        mask.loc[a:b] = False
    s0, s1 = sharpe(rb[mask], cash[mask]), sharpe(rx[mask], cash[mask])
    checks.append(('G3 Sharpe without 2020 and 2025', f'{s1:.4f} > {s0:.4f}', s1 > s0))
    boot = bootstrap_not_better(rx, rb, cash, block=63, n=2000, seed=7)
    checks.append(('G4 bootstrap not-better < 0.10', f'{boot:.4f}', boot < .10))
    ev = early_reentries(sx, sx.index)
    d = sx.index
    m = dmas.reindex(d)
    cash_sig = m < 45
    rng = np.random.default_rng(7)
    base_sh, gains = sharpe(rb, cash), []
    for _ in range(1000):
        forced = placebo_dates(rng, sx, sb, cash_sig, ev)
        sp = states(d, cash_sig, m < 30, m, None, forced)
        _, rp, _, _ = pair(idx, sb, sp, inputs)
        gains.append(sharpe(rp, cash) - base_sh)
    gains = np.array(gains)
    real = sharpe(rx, cash) - base_sh
    checks.append(('G5 real gain > placebo 90th pct', f'{real:+.4f} vs {np.quantile(gains, .9):+.4f} (placebo share >= real {(gains >= real).mean():.3f})', real > np.quantile(gains, .9)))
    rows = []
    for e in ev:
        k = idx.get_loc(e) + 2
        same = np.flatnonzero((wb.to_numpy()[k:] == wx.to_numpy()[k:]))
        end = k + (same[0] if len(same) else len(idx) - k)
        days = idx[k:end].intersection(rb.index)
        pnl = (1 + rx.loc[days]).prod() / (1 + rb.loc[days]).prod() - 1 if len(days) else np.nan
        rows.append((e.date(), int(dmas.loc[e]), idx[min(end, len(idx) - 1)].date(), len(days), pnl))
    evt = pd.DataFrame(rows, columns=['reentry', 'dmas', 'same_again', 'days', 'pnl_vs_R0'])
    checks.append(('G6 more than half of early re-entries add value', f'{(evt.pnl_vs_R0 > 0).sum()} of {len(evt)}', (evt.pnl_vs_R0 > 0).mean() > .5))

    # product: R1 and V2 with X (1993-04+), F on top
    gf = build_signals(inputs)
    d1 = gf.loc['1993-04-01':].index
    s1b = states(d1, gf.cash_R1, gf.deep, gf.dmas, None)
    s1x = states(d1, gf.cash_R1, gf.deep, gf.dmas, 60)
    w2, w2x = v2_from_states(inputs, s1b), v2_from_states(inputs, s1x)
    w2_ref, _, _ = run_k1.v2_weights(inputs)
    if not np.allclose(w2.dropna(), w2_ref.reindex(w2.dropna().index), equal_nan=True):
        raise RuntimeError('Base V2 does not reproduce run_k1.v2_weights')
    ep = episodes(idx, s1b, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    epx = episodes(idx, s1x, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    fl = floor_weights(idx, list(zip(ep.i_entry, ep.i_end)), .5)
    flx = floor_weights(idx, list(zip(epx.i_entry, epx.i_end)), .5)
    w2F = np.maximum(w2, fl).where(w2.notna())
    w2xF = np.maximum(w2x, flx).where(w2x.notna())
    R = lambda w: net_returns(w, inputs.eq_ret, inputs.cash_ret)
    v2r, v2xr = R(w2).dropna(), R(w2x).dropna()
    c2 = inputs.cash_ret.reindex(v2r.index)
    g7a, g7b = sharpe(v2xr.reindex(v2r.index), c2), sharpe(v2r, c2)
    g7c, g7d = metrics(v2xr.reindex(v2r.index), c2)['maxdd'], metrics(v2r, c2)['maxdd']
    checks.append(('G7 V2 with X: Sharpe >= V2 and maxDD >= V2 - 0.02', f'{g7a:.4f} vs {g7b:.4f}; {g7c:.4f} vs {g7d:.4f}', g7a >= g7b and g7c >= g7d - .02))
    verdict = all(c[2] for c in checks)

    # combos by period (report)
    END = pd.Timestamp('2026-08-31')
    k_idx = inputs.loc['1999-01-01':END].index
    pidx = idx[idx.get_loc(k_idx[0]) - 1:idx.get_loc(END) + 1]
    daily = run_k1.etf_daily_returns(levels, pidx).reindex(k_idx)
    pk = lambda held: {dd.to_period('M'): list(r.index[r.eq(1)]) for dd, r in held.iterrows()}
    K = lambda held, w: run_k1.combine(daily, pk(held), w, inputs.cash_ret)[0].K1
    ret = pd.DataFrame({'Buy & hold': inputs.eq_ret, 'R1': R(to_weights(idx, s1b)), 'R1+X': R(to_weights(idx, s1x)),
                        'V2': R(w2), 'V2+X': R(w2x), 'V2+X+F': R(w2xF),
                        'K1 (C5)': K(held_c5, w2), 'K1+F macro-only': K(held_m1, w2F),
                        'K1+X+F macro-only': K(held_m1, w2xF)}).loc['1999-02-01':END]
    cashk = inputs.cash_ret.reindex(ret.index)
    trow = []
    for name, a, b in WINDOWS:
        sub = ret.loc[a:b]
        for rule in ret:
            tot, sh, dd = stats(sub[rule], cashk)
            trow.append(dict(window=name, rule=rule, total=tot, sharpe=sh, maxdd=dd))
    t = pd.DataFrame(trow)
    t.to_csv(HERE / 'out/fast_combo.csv', index=False)
    order = list(ret.columns)

    rep = ['# Faster re-entry X (DMAS >= 60 on any decision date while in Cash)', '',
           f'Verdict: **{"PASS (qualifies for Europe)" if verdict else "FAIL"}**', '',
           f'Primary: R0+X vs R0, Tuesday, {rb.index[0].date()} → {rb.index[-1].date()}. Early re-entries: {len(ev)}.', '',
           '| Gate | Evidence | Result |', '|---|---|---|']
    rep += [f'| {n} | {e} | {"PASS" if p else "FAIL"} |' for n, e, p in checks]
    rep += ['', '## Primary metrics', '', '| Cost | Period | Rule | Sharpe | CAGR | Max DD |', '|---|---|---|---:|---:|---:|']
    for cost in (10., 25.):
        b_, x_, _, _ = pair(idx, sb, sx, inputs, cost)
        for nm, a, b in [('FULL', None, None), ('H1', None, H1_END), ('H2', '1991-01-01', None), ('P2021', '2021-01-01', None)]:
            for rule, r in (('R0', b_), ('R0+X', x_)):
                mm = metrics(r.loc[a:b], inputs.cash_ret.reindex(r.loc[a:b].index))
                rep.append(f'| {cost:.0f} bp | {nm} | {rule} | {mm["sharpe"]:.3f} | {mm["cagr"]:.2%} | {mm["maxdd"]:.1%} |')
    rep += ['', '## Product, 1993-04 → end (report; G7 above)', '', '| Rule | Sharpe | CAGR | Max DD |', '|---|---:|---:|---:|']
    for nm, w in (('R1', to_weights(idx, s1b)), ('R1+X', to_weights(idx, s1x)), ('V2', w2), ('V2+F', w2F), ('V2+X', w2x), ('V2+X+F', w2xF)):
        r = R(w).loc['1993-04-01':].dropna()
        mm = metrics(r, inputs.cash_ret.reindex(r.index))
        rep.append(f'| {nm} | {mm["sharpe"]:.3f} | {mm["cagr"]:.2%} | {mm["maxdd"]:.1%} |')
    for metric, fmt, title in (('sharpe', '{:.2f}', 'Sharpe by period'), ('total', '{:+.1%}', 'Total return by period'), ('maxdd', '{:.1%}', 'Worst fall by period')):
        piv = t.pivot_table(index='window', columns='rule', values=metric, sort=False)[order]
        rep += ['', f'## {title} (1999-02 → 2026-08, all rules same dates)', '', '| Window | ' + ' | '.join(order) + ' |', '|---|' + '---:|' * len(order)]
        for w, row in piv.iterrows():
            rep.append(f'| {w} | ' + ' | '.join('' if pd.isna(v) else fmt.format(v) for v in row) + ' |')
    rep += ['', '## Early re-entries (R0, Tuesday)', '', '| Re-entry | DMAS | Rules agree again | Days | P&L vs R0 |', '|---|---:|---|---:|---:|']
    rep += [f'| {r.reentry} | {r.dmas} | {r.same_again} | {r.days} | {r.pnl_vs_R0:+.2%} |' for r in evt.itertuples()]
    rep += ['', '## Sensitivity (report only, after the verdict)', '', '| Re-entry DMAS | Sharpe R0+X | Sharpe R0 | Max DD |', '|---|---:|---:|---:|']
    for th in (55, 60, 65):
        b_, x_, _, _ = pair(idx, sb, r0_states('W-TUE', th), inputs)
        c_ = inputs.cash_ret.reindex(b_.index)
        rep.append(f'| {th} | {sharpe(x_, c_):.3f} | {sharpe(b_, c_):.3f} | {metrics(x_, c_)["maxdd"]:.1%} |')
    rep += ['', '## Method', '', 'As spec 6z. Base states reproduce trading.sleeve_weights; base V2 reproduces run_k1.v2_weights (checked in code). Placebo: DMAS re-entry switched off, each real early re-entry replaced by a random non-first-of-month decision date in the same spell before the base re-entry. F as in run_p (ratio 0.5, VIX < 30, 50% floor) on the R1 or R1+X cash spells. K1 = run_k1.combine. Inputs read-only.', '']
    (HERE / 'RESULTS_fast.md').write_text('\n'.join(rep))
    evt.to_csv(HERE / 'out/fast_reentries.csv', index=False)
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
