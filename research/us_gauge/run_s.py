"""Shock re-entry S — pre-registered in the gauge spec section 6s (commit 838c4a2).

US run = information only (S was found on US data). Europe run (STOXX 600) = the test.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from research.us_gauge import run_k1, run_s2, run_vol
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals
from research.us_gauge.run_m1 import build as build_m1
from research.us_gauge.run_p import decision_days, episodes as f_episodes
from research.us_gauge.run_c import floor_weights, sharpe, window_dd
from research.us_gauge.run_fast import states, to_weights, v2_from_states
from research.us_gauge.gauge import dmas_on
from research.us_gauge.trading import net_returns
from research.us_gauge.evaluate import metrics, bootstrap_not_better
from research.us_gauge.stress_periods import stats
from research.us_gauge.combo_f import WINDOWS

HERE = Path(__file__).resolve().parent
FREQS = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI']
SCRATCH = Path('/private/tmp/claude-501/-Users-larazanella-Desktop-GitHub-Projects-ic-technical/abe5825f-ac19-4130-a032-71fdd52b333c/scratchpad/eur')


def shock_episodes(px: pd.Series, base_w: pd.Series) -> pd.DataFrame:
    """Daily state machine of spec 6s. base_w = base equity weight in force each day (1 = Invested)."""
    p = px.to_numpy()
    n = len(p)
    hi = np.full(n, -1)
    win = sliding_window_view(p, 252)
    hi[251:] = np.arange(n - 251) + win.argmax(axis=1)
    bw = base_w.reindex(px.index).to_numpy()
    rows, k = [], 252
    last_h, last_L = -1, np.inf
    while k < n:
        h = hi[k]
        seg = p[h:k + 1]
        l = h + int(seg.argmin())
        L = p[l]
        if h != last_h:
            last_h, last_L = h, np.inf
        if (L <= .85 * p[h] and l - h <= 63 and p[k] >= 1.10 * L and L < last_L
                and not np.isnan(bw[k]) and bw[k] < 1):
            e, end, reason = k, n - 1, 'open'
            for j in range(k + 1, n):
                if bw[j] >= 1:
                    end, reason = j, 'base invested'
                    break
                if p[j] < L:
                    end, reason = j, 'stop'
                    break
                if j == k + 126:
                    end, reason = j, 'time'
                    break
            rows.append((px.index[h], px.index[l], px.index[e], px.index[end], reason, p[h], L, e, end))
            last_L = L
            k = end + 1
            continue
        k += 1
    return pd.DataFrame(rows, columns=['high', 'low', 'entry', 'end', 'reason', 'H', 'L', 'i_entry', 'i_end'])


def placebo_spans(rng, n_ep, px, base_w):
    p = px.to_numpy()
    bw = base_w.reindex(px.index).to_numpy()
    elig = np.flatnonzero((bw == 0) & (np.arange(len(p)) >= 21))
    spans = []
    for e in rng.choice(elig, size=n_ep, replace=True):
        L = p[e - 20:e + 1].min()
        end = len(p) - 1
        for j in range(e + 1, len(p)):
            if bw[j] >= 1 or p[j] < L or j == e + 126:
                end = j
                break
        spans.append((e, end))
    return spans


def r0_pair(inp, dmas, freq, start, cost=10.):
    d = decision_days(inp.index, start, freq)
    m = dmas.reindex(d)
    wb = to_weights(inp.index, states(d, m < 45, m < 30, m, None))
    ep = shock_episodes(inp.px, wb)
    fl = floor_weights(inp.index, list(zip(ep.i_entry, ep.i_end)), 1.)
    ws = np.maximum(wb, fl).where(wb.notna())
    rb = net_returns(wb, inp.eq_ret, inp.cash_ret, cost).where(wb.notna()).dropna()
    rs = net_returns(ws, inp.eq_ret, inp.cash_ret, cost).reindex(rb.index)
    return rb, rs, wb, ep


def gates(inp, dmas, start, h_split, bears, drop, europe=False):
    rb, rs, wb, ep = r0_pair(inp, dmas, 'W-TUE', start)
    cash = inp.cash_ret.reindex(rb.index)
    checks = []
    for nm, a, b in [('FULL', None, None), ('H1', None, h_split), ('H2', pd.Timestamp(h_split) + pd.Timedelta(days=1), None)]:
        s0, s1 = sharpe(rb.loc[a:b], cash.loc[a:b]), sharpe(rs.loc[a:b], cash.loc[a:b])
        if not europe or nm == 'FULL':
            checks.append((f'{nm} Sharpe R0+S > R0', f'{s1:.4f} > {s0:.4f}', s1 > s0))
        else:
            checks.append((f'(report) {nm} Sharpe R0+S vs R0', f'{s1:.4f} vs {s0:.4f}', None))
    wd = []
    for f in FREQS:
        b_, s_, _, _ = r0_pair(inp, dmas, f, start)
        c_ = inp.cash_ret.reindex(b_.index)
        wd.append((f, sharpe(b_, c_), sharpe(s_, c_)))
    mb, ms = np.mean([w[1] for w in wd]), np.mean([w[2] for w in wd])
    checks.append(('Weekday-mean Sharpe', f'{ms:.4f} > {mb:.4f} ({", ".join(f"{f[-3:]} {s:.3f}/{b:.3f}" for f, b, s in wd)})', ms > mb))
    d0, d1 = metrics(rb, cash)['maxdd'], metrics(rs, cash)['maxdd']
    checks.append(('FULL maxDD >= R0 - 0.02', f'{d1:.4f} vs {d0:.4f}', d1 >= d0 - .02))
    if not europe:
        for a, b in bears:
            x0, x1 = window_dd(rb, a, b), window_dd(rs, a, b)
            checks.append((f'maxDD {a[:7]}..{b[:7]}', f'{x1:.4f} vs {x0:.4f}', x1 >= x0 - .02))
        mask = pd.Series(True, index=rb.index)
        for a, b in drop:
            mask.loc[a:b] = False
        s0, s1 = sharpe(rb[mask], cash[mask]), sharpe(rs[mask], cash[mask])
        checks.append(('Sharpe without 2020 and 2025', f'{s1:.4f} > {s0:.4f}', s1 > s0))
        boot = bootstrap_not_better(rs, rb, cash, block=63, n=2000, seed=7)
        checks.append(('Bootstrap not-better < 0.10', f'{boot:.4f}', boot < .10))
    rng = np.random.default_rng(7)
    base_sh, gains = sharpe(rb, cash), []
    eq, cr = inp.eq_ret.to_numpy(), inp.cash_ret.to_numpy()
    keep = wb.notna().to_numpy()
    wbn = wb.to_numpy()
    for _ in range(1000):
        f = floor_weights(inp.index, placebo_spans(rng, len(ep), inp.px, wb), 1.).to_numpy()
        w = np.where(keep, np.maximum(wbn, f), np.nan)
        dw = np.abs(np.diff(w, prepend=np.nan))
        dw[np.isnan(dw)] = 0.
        x = (w * eq + (1 - w) * cr - dw * 10 / 1e4)[keep] - cr[keep]
        gains.append(x.mean() / x.std(ddof=1) * np.sqrt(252) - base_sh)
    gains = np.array(gains)
    real = sharpe(rs, cash) - base_sh
    checks.append(('Placebo: real gain > 90th pct', f'{real:+.4f} vs {np.quantile(gains, .9):+.4f} (placebo share >= real {(gains >= real).mean():.3f})', real > np.quantile(gains, .9)))
    pnl = []
    for r in ep.itertuples():
        days = inp.index[r.i_entry + 2:r.i_end + 2].intersection(rb.index)
        pnl.append((1 + rs.loc[days]).prod() / (1 + rb.loc[days]).prod() - 1 if len(days) else np.nan)
    ep['pnl_vs_R0'] = pnl
    checks.append(('More than half of episodes add value', f'{(ep.pnl_vs_R0 > 0).sum()} of {len(ep)}', (ep.pnl_vs_R0 > 0).mean() > .5))
    verdict = all(c[2] for c in checks if c[2] is not None)
    return dict(checks=checks, verdict=verdict, rb=rb, rs=rs, ep=ep)


def table(L, out, inp, periods):
    L += ['| Gate | Evidence | Result |', '|---|---|---|']
    L += [f'| {n} | {e} | {"report" if p is None else "PASS" if p else "FAIL"} |' for n, e, p in out['checks']]
    L += ['', '| Period | Rule | Sharpe | CAGR | Max DD |', '|---|---|---:|---:|---:|']
    for nm, a, b in periods:
        for rule, r in (('R0', out['rb']), ('R0+S', out['rs'])):
            m = metrics(r.loc[a:b], inp.cash_ret.reindex(r.loc[a:b].index))
            L.append(f'| {nm} | {rule} | {m["sharpe"]:.3f} | {m["cagr"]:.2%} | {m["maxdd"]:.1%} |')
    L += ['', '| 1-year high | Low | Entry | End | Reason | Fall | P&L vs R0 |', '|---|---|---|---|---|---:|---:|']
    L += [f'| {r.high.date()} | {r.low.date()} | {r.entry.date()} | {r.end.date()} | {r.reason} | {r.L / r.H - 1:.1%} | {r.pnl_vs_R0:+.2%} |' for r in out['ep'].itertuples()]
    L.append('')


def europe_inputs():
    with read_only_external():
        px = load('BBG_SXXP', refresh=False).sort_index()
    eur = eur_cash()
    idx = px.index
    cash = eur.reindex(idx, method='ffill') / 100 / 252
    return pd.DataFrame({'px': px, 'eq_ret': px.pct_change(fill_method=None) + .03 / 252, 'cash_ret': cash,
                         'eq_ret_price_only': px.pct_change(fill_method=None)})


def eur_cash() -> pd.Series:
    """EUR 3-month rate, % a year, monthly. Spec: FRED Germany to 1998-12, euro area from 1999-01;
    fallback ECB Euribor 3M (from 1994-01) if FRED is not reachable (recorded in the report)."""
    de, ez = SCRATCH / 'IR3TIB01DEM156N.csv', SCRATCH / 'IR3TIB01EZM156N.csv'
    if de.exists() and ez.exists():
        rd = lambda f: pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
        s = pd.concat([rd(de).loc[:'1998-12-31'], rd(ez).loc['1999-01-01':]])
        s.attrs['source'] = 'FRED IR3TIB01DEM156N (to 1998-12) + IR3TIB01EZM156N (from 1999-01)'
        return s
    d = pd.read_csv(SCRATCH / 'euribor3m.csv')
    s = pd.Series(d.OBS_VALUE.astype(float).to_numpy(), index=pd.to_datetime(d.TIME_PERIOD))
    s.attrs['source'] = 'ECB Euribor 3M monthly (from 1994-01); FRED unreachable'
    return s


def main():
    panel, c5, _, macro, _ = build_m1()
    with read_only_external():
        inputs = load_inputs()
    etf = run_s2.load_etfs()
    _, held_m1 = run_s2.tilt_returns(macro, etf, panel.r_rf)
    with read_only_external():
        levels = pd.DataFrame({t: load(t + '.US_TR', refresh=False).sort_index() for t in etf.columns})
    idx = inputs.index
    dmas = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0]
    us = gates(inputs, dmas, '1955-01-01', '1990-12-31',
               [('2000-03-01', '2003-03-31'), ('2007-10-01', '2009-06-30'), ('2022-01-01', '2022-12-31')],
               [('2020-02-19', '2020-08-31'), ('2025-02-19', '2025-06-30')])

    # product: V2+F vs V2+F+S, and K1 macro-only
    gf = build_signals(inputs)
    d1 = gf.loc['1993-04-01':].index
    s1 = states(d1, gf.cash_R1, gf.deep, gf.dmas, None)
    w1 = to_weights(idx, s1)
    w2 = v2_from_states(inputs, s1)
    epf = f_episodes(idx, s1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    w2F = np.maximum(w2, floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)).where(w2.notna())
    ep1 = shock_episodes(inputs.px, w1)
    v = run_vol.target_weight(inputs.eq_ret)
    sw = pd.Series(0., index=idx)
    for r in ep1.itertuples():
        sw.iloc[r.i_entry + 2:min(r.i_end + 2, len(idx))] = v.iloc[r.i_entry]
    w2FS = np.maximum(w2F, sw).where(w2F.notna())
    R = lambda w: net_returns(w, inputs.eq_ret, inputs.cash_ret)
    END = pd.Timestamp('2026-08-31')
    k_idx = inputs.loc['1999-01-01':END].index
    pidx = idx[idx.get_loc(k_idx[0]) - 1:idx.get_loc(END) + 1]
    daily = run_k1.etf_daily_returns(levels, pidx).reindex(k_idx)
    picks = {dd.to_period('M'): list(r.index[r.eq(1)]) for dd, r in held_m1.iterrows()}
    K = lambda w: run_k1.combine(daily, picks, w, inputs.cash_ret)[0].K1
    ret = pd.DataFrame({'Buy & hold': inputs.eq_ret, 'V2+F': R(w2F), 'V2+F+S': R(w2FS),
                        'K1+F macro-only': K(w2F), 'K1+F+S macro-only': K(w2FS)}).loc['1999-02-01':END]
    cashk = inputs.cash_ret.reindex(ret.index)
    trow = []
    for name, a, b in WINDOWS:
        sub = ret.loc[a:b]
        for rule in ret:
            tot, sh, dd = stats(sub[rule], cashk)
            trow.append(dict(window=name, rule=rule, total=tot, sharpe=sh, maxdd=dd))
    t = pd.DataFrame(trow)
    t.to_csv(HERE / 'out/s_combo.csv', index=False)

    # Europe, one run
    eu = europe_inputs()
    cache = HERE / 'out/dmas_sxxp_daily.csv'
    if cache.exists():
        dm_eu = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        dm_eu = dmas_on(eu.px, eu.loc['1988-01-01':].index)
        dm_eu.to_csv(cache)
    src = eur_cash().attrs['source']
    eu_start = '1988-01-01' if src.startswith('FRED') else '1994-02-01'
    eu.loc[eu.index < pd.Timestamp(eu_start), 'cash_ret'] = np.nan
    euo = gates(eu, dm_eu, eu_start, '2006-12-31', [], [], europe=True)
    eu_po = eu.assign(eq_ret=eu.eq_ret_price_only)
    rb_po, rs_po, _, _ = r0_pair(eu_po, dm_eu, 'W-TUE', eu_start)

    rep = ['# Shock re-entry S', '',
           f'**Europe (the test): {"PASS" if euo["verdict"] else "FAIL"}** · US (information only, S was found on US data): {"pass" if us["verdict"] else "fail"}', '',
           '## Europe — STOXX 600, R0 base, one run', '',
           f'Sample {euo["rb"].index[0].date()} → {euo["rb"].index[-1].date()}. Cash: {src}. Equity = SXXP price change + 3%/252. Episodes: {len(euo["ep"])}.', '']
    table(rep, euo, eu, [('FULL', None, None), ('H1 to 2006', None, '2006-12-31'), ('H2 from 2007', '2007-01-01', None), ('P2021', '2021-01-01', None)])
    c_po = eu.cash_ret.reindex(rb_po.index)
    rep += [f'Price-only returns (no dividend proxy): Sharpe R0+S {sharpe(rs_po, c_po):.3f} vs R0 {sharpe(rb_po, c_po):.3f}.', '',
            '## US — S&P 500, R0 base, 1955-2026 (information only)', '']
    table(rep, us, inputs, [('FULL', None, None), ('H1 to 1990', None, '1990-12-31'), ('H2 from 1991', '1991-01-01', None), ('P2021', '2021-01-01', None)])
    order = list(ret.columns)
    for metric, fmt, title in (('sharpe', '{:.2f}', 'Sharpe'), ('total', '{:+.1%}', 'Total return'), ('maxdd', '{:.1%}', 'Worst fall')):
        piv = t.pivot_table(index='window', columns='rule', values=metric, sort=False)[order]
        rep += [f'## US product by period — {title} (1999-02 → 2026-08)', '', '| Window | ' + ' | '.join(order) + ' |', '|---|' + '---:|' * len(order)]
        for w, row in piv.iterrows():
            rep.append(f'| {w} | ' + ' | '.join('' if pd.isna(x) else fmt.format(x) for x in row) + ' |')
        rep.append('')
    rep += ['US product S episodes (R1 base): ' + ', '.join(f'{r.entry.date()}→{r.end.date()} ({r.reason})' for r in ep1.itertuples()), '',
            '## Method', '', 'As spec 6s. A trigger while the base weight in force is already 1 is not an episode (S has nothing to do); '
            'after an episode the scan restarts the day after it ends. R0 base = run_fast.states (reproduces trading.sleeve_weights). '
            'Europe DMAS = gauge.dmas_on on SXXP closes, every trading day from 1988 (cached out/dmas_sxxp_daily.csv). '
            'Placebo: entries on random days with base weight 0, L = lowest of the 21 closes ending there, same stop, base-invested end and 126-day limit.', '']
    (HERE / 'RESULTS_s.md').write_text('\n'.join(rep))
    for name, o in (('us', us), ('eu', euo)):
        o['ep'].to_csv(HERE / f'out/s_episodes_{name}.csv', index=False)
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
