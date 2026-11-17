"""'Panic is over' floor P and VIX fallback F — pre-registered in the gauge spec section 6y (commit a66641a)."""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_k1
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals
from research.us_gauge.run_c import base_r0, run_pair, sharpe, window_dd, floor_weights
from research.us_gauge.evaluate import metrics, bootstrap_not_better

HERE = Path(__file__).resolve().parent
OAS_DAILY = pd.Timestamp('2000-08-15')
H1_END = '2009-12-31'
BEARS = [('2000-03-01', '2003-03-31'), ('2007-10-01', '2009-06-30'), ('2022-01-01', '2022-12-31')]
DROP = [('2020-02-19', '2020-08-31'), ('2025-02-19', '2025-06-30')]
FREQS = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI']


def decision_days(index, start, freq):
    d = pd.date_range(start, index[-1], freq=freq)
    pos = index.searchsorted(d, side='right') - 1
    return index[pos[pos >= 0]].drop_duplicates()


def base_states(dates, cash, deep):
    """Decision-date state of trading.sleeve_weights (1 invested, 0 cash), all-to-cash."""
    out, cur, last = [], np.nan, None
    for d in dates:
        m = d.to_period('M')
        if m != last:
            cur = 0. if cash.loc[d] else 1.
        elif cur == 1. and cash.loc[d] and deep.loc[d]:
            cur = 0.
        last = m
        out.append(cur)
    return pd.Series(out, index=dates)


def spells(state):
    """Spell id per decision date (NaN when invested)."""
    cash = state.eq(0)
    sid = (cash & ~cash.shift(fill_value=False)).cumsum().where(cash)
    return sid


def episodes(index, state, stress, vix, px, ratio=.75, cap=30.):
    """Entry: stress <= ratio*peak and vix < cap; end: base invested, stress > peak at entry, or P < L."""
    pos = {d: index.get_loc(d) for d in state.index}
    sid = spells(state)
    rows, on, used = [], None, set()
    for t in state.index:
        s = sid.loc[t]
        if on is not None:
            reason = 'base invested' if state.loc[t] == 1 else (
                'stress stop' if stress.loc[t] > on['pk'] else 'price stop' if px.loc[t] < on['L'] else None)
            if reason:
                on.update(end=t, i_end=pos[t], reason=reason)
                rows.append(on)
                on = None
            continue
        if pd.isna(s) or s in used:
            continue
        d0 = sid.index[sid.eq(s)][0]
        i0 = pos[d0]
        pk = stress.iloc[max(i0 - 21, 0):pos[t] + 1].max()
        if pd.notna(stress.loc[t]) and stress.loc[t] <= ratio * pk and vix.loc[t] < cap:
            used.add(s)
            on = dict(spell_start=d0, entry=t, i_entry=pos[t], pk=pk, L=px.loc[d0:t].min(),
                      stress=stress.loc[t], vix=vix.loc[t])
    if on is not None:
        on.update(end=index[-1], i_end=len(index) - 1, reason='open')
        rows.append(on)
    return pd.DataFrame(rows, columns=['spell_start', 'entry', 'end', 'reason', 'stress', 'pk', 'vix', 'L',
                                       'i_entry', 'i_end'])


def placebo_spans(rng, n, state, index, px):
    sid = spells(state)
    ids = sid.dropna().unique()
    pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
    spans = []
    for s in pick:
        days = sid.index[sid.eq(s)]
        t = days[rng.integers(len(days))]
        L = px.loc[days[0]:t].min()
        end = index[-1]
        for u in state.index[state.index > t]:
            if state.loc[u] == 1 or px.loc[u] < L:
                end = u
                break
        spans.append((index.get_loc(t), index.get_loc(end)))
    return spans


def evaluate_rule(name, inputs, dmas, stress, start, ratio=.75, cap=30., floor_level=.5, full=True):
    idx, px, vix = inputs.index, inputs.px, inputs.vix
    res = {}
    for freq in FREQS:
        dates = decision_days(idx, '1955-01-01', freq)
        d = dmas.reindex(dates)
        st = base_states(dates, d < 45, d < 30)
        st_s = st.loc[start:]
        ep = episodes(idx, st_s, stress, vix, px, ratio, cap)
        fl = floor_weights(idx, list(zip(ep.i_entry, ep.i_end)), floor_level)
        wb = base_r0(inputs, dmas, freq)
        rb, rc, _ = run_pair(wb, fl, inputs)
        first = idx[idx.get_loc(st_s.index[0]) + 2]
        res[freq] = (rb.loc[first:], rc.loc[first:], ep, st_s, wb)
    if not full:
        rb, rc = res['W-TUE'][:2]
        return sharpe(rc, inputs.cash_ret.reindex(rc.index)), sharpe(rb, inputs.cash_ret.reindex(rb.index)), len(res['W-TUE'][2]), metrics(rc, inputs.cash_ret.reindex(rc.index))['maxdd']
    rb, rc, ep, st_s, wb = res['W-TUE']
    cash = inputs.cash_ret.reindex(rb.index)
    checks = []
    for nm, a, b in [('FULL', None, None), ('H1', None, H1_END), ('H2', '2010-01-01', None)]:
        sb, sc = sharpe(rb.loc[a:b], cash.loc[a:b]), sharpe(rc.loc[a:b], cash.loc[a:b])
        checks.append((f'G1 {nm} Sharpe {name} > R0', f'{sc:.4f} > {sb:.4f}', sc > sb))
    wd = [(f, sharpe(r[0], inputs.cash_ret.reindex(r[0].index)), sharpe(r[1], inputs.cash_ret.reindex(r[1].index))) for f, r in res.items()]
    mb, mc = np.mean([x[1] for x in wd]), np.mean([x[2] for x in wd])
    checks.append(('G1 weekday-mean Sharpe', f'{mc:.4f} > {mb:.4f} ({", ".join(f"{f[-3:]} {c:.3f}/{b:.3f}" for f, b, c in wd)})', mc > mb))
    db, dc = metrics(rb, cash)['maxdd'], metrics(rc, cash)['maxdd']
    checks.append(('G2 FULL maxDD >= R0 - 0.02', f'{dc:.4f} vs {db:.4f}', dc >= db - .02))
    for a, b in BEARS:
        xb, xc = window_dd(rb, a, b), window_dd(rc, a, b)
        checks.append((f'G2 maxDD {a[:7]}..{b[:7]}', f'{xc:.4f} vs {xb:.4f}', xc >= xb - .02))
    mask = pd.Series(True, index=rb.index)
    for a, b in DROP:
        mask.loc[a:b] = False
    sb, sc = sharpe(rb[mask], cash[mask]), sharpe(rc[mask], cash[mask])
    checks.append(('G3 Sharpe without 2020 and 2025', f'{sc:.4f} > {sb:.4f}', sc > sb))
    boot = bootstrap_not_better(rc, rb, cash, block=63, n=2000, seed=7)
    checks.append(('G4 bootstrap not-better < 0.10', f'{boot:.4f}', boot < .10))
    rng = np.random.default_rng(7)
    base_sh, gains = sharpe(rb, cash), []
    for _ in range(1000):
        fl = floor_weights(inputs.index, placebo_spans(rng, len(ep), st_s, inputs.index, inputs.px), floor_level)
        _, rp, _ = run_pair(wb, fl, inputs)
        rp = rp.loc[rb.index[0]:]
        gains.append(sharpe(rp, cash) - base_sh)
    gains = np.array(gains)
    real = sharpe(rc, cash) - base_sh
    checks.append(('G5 real gain > placebo 90th pct', f'{real:+.4f} vs {np.quantile(gains, .9):+.4f} (placebo share >= real {(gains >= real).mean():.3f})', real > np.quantile(gains, .9)))
    pnl = []
    for r in ep.itertuples():
        days = inputs.index[r.i_entry + 2:r.i_end + 2].intersection(rb.index)
        pnl.append((1 + rc.loc[days]).prod() / (1 + rb.loc[days]).prod() - 1 if len(days) else np.nan)
    ep['pnl_vs_R0'] = pnl
    share = (ep.pnl_vs_R0 > 0).mean() if len(ep) else 0.
    checks.append(('G6 more than half of episodes add value', f'{(ep.pnl_vs_R0 > 0).sum()} of {len(ep)}', share > .5))
    # G7 product check: V2 with R1 spells, Tuesday
    gf = build_signals(inputs)
    d1 = gf.loc['1993-04-01':].index
    st1 = base_states(d1, gf.cash_R1, gf.deep).loc[start:]
    ep1 = episodes(inputs.index, st1, stress, inputs.vix, inputs.px, ratio, cap)
    w2, _, _ = run_k1.v2_weights(inputs)
    fl1 = floor_weights(inputs.index, list(zip(ep1.i_entry, ep1.i_end)), floor_level)
    b2, c2, _ = run_pair(w2.reindex(inputs.index), fl1, inputs)
    b2, c2 = b2.loc[rb.index[0]:], c2.loc[rb.index[0]:]
    c2c = inputs.cash_ret.reindex(b2.index)
    s2b, s2c = sharpe(b2, c2c), sharpe(c2, c2c)
    m2b, m2c = metrics(b2, c2c)['maxdd'], metrics(c2, c2c)['maxdd']
    checks.append(('G7 V2+rule Sharpe >= V2 and maxDD >= V2 - 0.02', f'{s2c:.4f} vs {s2b:.4f}; {m2c:.4f} vs {m2b:.4f}', s2c >= s2b and m2c >= m2b - .02))
    return dict(checks=checks, rb=rb, rc=rc, ep=ep, ep1=ep1, b2=b2, c2=c2, verdict=all(c[2] for c in checks))


def captured(rb, rc, bh, a, b):
    g = lambda r: (1 + r.loc[a:b]).prod() - 1
    gap = g(bh) - g(rb)
    return g(rb), g(rc), g(bh), (g(rc) - g(rb)) / gap if gap else np.nan


def section(title, out, inputs, stress_name):
    L = [f'## {title}', '', f'Verdict: **{"PASS" if out["verdict"] else "FAIL"}**', '',
         f'Sample: {out["rb"].index[0].date()} → {out["rb"].index[-1].date()}. Episodes (R0, Tuesday): {len(out["ep"])}.', '',
         '| Gate | Evidence | Result |', '|---|---|---|']
    L += [f'| {n} | {e} | {"PASS" if p else "FAIL"} |' for n, e, p in out['checks']]
    cash = inputs.cash_ret
    L += ['', '| Period | Rule | Sharpe | CAGR | Max DD |', '|---|---|---:|---:|---:|']
    for nm, a, b in [('FULL', None, None), ('H1', None, H1_END), ('H2', '2010-01-01', None), ('P2021', '2021-01-01', None)]:
        for rule, r in (('R0', out['rb']), ('R0+rule', out['rc']), ('V2', out['b2']), ('V2+rule', out['c2'])):
            m = metrics(r.loc[a:b], cash.reindex(r.loc[a:b].index))
            L.append(f'| {nm} | {rule} | {m["sharpe"]:.3f} | {m["cagr"]:.2%} | {m["maxdd"]:.1%} |')
    bh = inputs.eq_ret
    L += ['', 'Rebound capture (share of the gap between the base and buy & hold that the rule closes):', '',
          '| Window | Base | Base+rule | Buy & hold | Captured |', '|---|---:|---:|---:|---:|']
    for nm, a, b in [('COVID rebound 24 Mar-31 Aug 2020', '2020-03-24', '2020-08-31'), ('Liberation Day rebound 9 Apr-30 Jun 2025', '2025-04-09', '2025-06-30')]:
        for lab, (x, y) in (('R0', (out['rb'], out['rc'])), ('V2', (out['b2'], out['c2']))):
            g0, g1, g2, cap = captured(x, y, bh, a, b)
            L.append(f'| {nm} ({lab}) | {g0:+.1%} | {g1:+.1%} | {g2:+.1%} | {cap:.0%} |')
    L += ['', f'Episodes (R0 base, Tuesday; stress = {stress_name}):', '',
          '| Spell start | Entry | End | Reason | Stress | Peak | VIX | P&L vs R0 |', '|---|---|---|---|---:|---:|---:|---:|']
    L += [f'| {r.spell_start.date()} | {r.entry.date()} | {r.end.date()} | {r.reason} | {r.stress:.2f} | {r.pk:.2f} | {r.vix:.1f} | {r.pnl_vs_R0:+.2%} |' for r in out['ep'].itertuples()]
    L += ['', f'V2 product episodes (R1 spells): ' + ', '.join(f'{r.entry.date()}→{r.end.date()} ({r.reason})' for r in out['ep1'].itertuples()), '']
    return L


def main():
    with read_only_external():
        inputs = load_inputs()
        oas = load('BBG_LF98OAS', refresh=False).sort_index()
        vix3m = load('CBOE_VIX3M', refresh=False).sort_index()
    dmas = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0]
    idx = inputs.index
    oas_d = oas.loc[OAS_DAILY:].reindex(idx, method='ffill').where(idx >= OAS_DAILY).shift(1)
    start_p = idx[idx.get_loc(idx[idx > OAS_DAILY][0]) + 22]
    vix_s = inputs.vix
    start_f = idx[idx.get_loc(vix_s.first_valid_index()) + 22]
    P = evaluate_rule('P', inputs, dmas, oas_d, start_p)
    F = evaluate_rule('F', inputs, dmas, vix_s, start_f, ratio=.5)
    rep = ['# "Panic is over" floor P (credit + VIX) and fallback F (VIX only)', '',
           f'Adoption order (pre-registered): P if P passes; F only if P fails and F passes. P {"PASS" if P["verdict"] else "FAIL"}, F {"PASS" if F["verdict"] else "FAIL"}.', '']
    rep += section('P — US high-yield OAS (lagged 1 day) + VIX < 30', P, inputs, 'LF98OAS')
    rep += section('F — VIX half of spell peak + VIX < 30 (fallback, 1990+)', F, inputs, 'VIX')
    v3 = vix3m.reindex(idx, method='ffill')
    rep += ['## Report only (written after the verdicts)', '', 'VIX / VIX3M at P entries from 2009-09: ' +
            ', '.join(f'{r.entry.date()} {inputs.vix.loc[r.entry] / v3.loc[r.entry]:.2f}' for r in P['ep'].itertuples() if r.entry >= pd.Timestamp('2009-09-18')), '',
            '| Rule | Ratio | VIX cap | Floor | Sharpe rule | Sharpe R0 | Episodes | Max DD rule |', '|---|---|---|---|---:|---:|---:|---:|']
    for nm, stress, st, base_ratio in (('P', oas_d, start_p, .75), ('F', vix_s, start_f, .5)):
        for ratio in ((.67, .75, .80) if nm == 'P' else (.4, .5, .6)):
            for cap in (25., 30., 35.):
                for fl in (.5, 1.):
                    sc, sb, n, dd = evaluate_rule(nm, inputs, dmas, stress, st, ratio, cap, fl, full=False)
                    rep.append(f'| {nm} | {ratio} | {cap:.0f} | {fl} | {sc:.3f} | {sb:.3f} | {n} | {dd:.1%} |')
    rep += ['', '## Deviations', '', 'First run (same day) had a coding error: F was evaluated with the P ratio 0.75 instead of the pre-registered 0.5. Its F gate table was wrong and discarded; P was unaffected. The sensitivity table of that run already showed the correct F row (ratio 0.5, cap 30, floor 0.5) before this corrected gate run. No rule, threshold or gate changed.', '', '## Method', '', 'As spec 6y and its implementation notes. Base R0 = run_c.base_r0; spells and P state read on decision dates; floor earns entry+2 → end+1. Placebo: 1,000 draws, default_rng(7), spells without replacement, uniform date inside the spell, price stop and base-invested end only. G7 uses V2 (run_k1.v2_weights) with R1 cash spells, compared over the P sample. Sensitivity rows use the Tuesday R0 base only. F sensitivity ratios 0.4/0.5/0.6 (P: 0.67/0.75/0.80 as specified). Inputs read-only.', '']
    (HERE / 'RESULTS_p.md').write_text('\n'.join(rep))
    P['ep'].to_csv(HERE / 'out/p_episodes.csv', index=False)
    F['ep'].to_csv(HERE / 'out/f_episodes.csv', index=False)
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
