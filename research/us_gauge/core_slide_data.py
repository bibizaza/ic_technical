"""Data for the core slide (spec 11d): core = 100% S&P 500 x max(keys x vol size, F floor 50%, S at max(live, 50%), B at 100%).

Writes mockups/core_slide_data.json: 2026 daily price and weight, Tuesday decisions, current state of every
layer, and back-test stats of the core vs buy and hold by period (1999-02 → last day, daily, 10 bp).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_vol
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals
from research.us_gauge.revisions import detect_rolls, revisions, rev_zscore
from research.us_gauge.run_p import episodes as f_episodes
from research.us_gauge.run_c import floor_weights
from research.us_gauge.run_fast import states, to_weights, v2_from_states
from research.us_gauge.run_s import shock_episodes
from research.us_gauge.run_b import breadth, deemer_ratio, thrust_days, thrust_episodes
from research.us_gauge.trading import net_returns
from research.us_gauge.stress_periods import stats
from research.us_gauge.combo_f import WINDOWS

ROOT = Path(__file__).resolve().parents[2]


def layer(index, ep, live):
    s = pd.Series(0., index=index)
    for r in ep.itertuples():
        a, b = r.i_entry + 2, min(r.i_end + 2, len(index))
        s.iloc[a:b] = live.iloc[a:b].to_numpy()
    return s


def committee(index, dec_dates, desired, band=.10):
    """Wednesday committee: at each Tuesday close t, move to the desired weight if it differs from the
    held weight by >= band, or if we go to / leave cash. Trade Wednesday (t+1), earn from Thursday (t+2)."""
    ch, held = pd.Series(np.nan, index=index), np.nan
    for d in dec_dates:
        want = float(desired.loc[d])
        if np.isnan(held) or (want == 0) != (held == 0) or abs(want - held) >= band - 1e-12:
            held = want
        p = index.get_loc(d) + 2
        if p < len(index):
            ch.iloc[p] = held
    return ch.ffill()


def act_on_change(index, desired, band=.10):
    """Product rule: every day t, move to the desired weight if it differs from the held weight by >= band,
    or if we go to / leave cash. Trade t+1, earn from t+2. The Wednesday committee reviews, it does not wait.
    Returns (weight in force on each day, weight held after the last signal day, list of trades)."""
    out, held, d, trades = pd.Series(np.nan, index=index), np.nan, desired.reindex(index).to_numpy(), []
    for i in range(len(index)):
        if np.isnan(d[i]):
            continue
        if np.isnan(held) or (d[i] == 0) != (held == 0) or abs(d[i] - held) >= band - 1e-12:
            if not np.isnan(held):
                trades.append(dict(signal=str(index[i].date()),
                                   trade=str(index[i + 1].date()) if i + 1 < len(index) else 'next trading day',
                                   frm=float(held), to=float(d[i])))
            held = d[i]
        if i + 2 < len(index):
            out.iloc[i + 2] = held
    return out.ffill(), held, trades


def desired_from_states(inputs, s1, v, live, td, b_size=None):
    """Desired weight on each signal day: max(key state x v, F floor, S, B); S and B sized by `live` (a series)."""
    idx = inputs.index
    w1 = to_weights(idx, s1)
    epf = f_episodes(idx, s1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    wf = floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)
    ws = layer(idx, shock_episodes(inputs.px, w1), live)
    wb = layer(idx, thrust_episodes(inputs.px, w1, td), live if b_size is None else b_size)
    st = s1.reindex(idx).ffill()
    return pd.concat([st * v, wf.shift(-2), ws.shift(-2), wb.shift(-2)], axis=1).max(axis=1)


def m13612(tr: pd.Series) -> pd.Series:
    r = lambda h: tr / tr.shift(h) - 1
    return 12 * r(21) + 4 * r(63) + 2 * r(126) + r(252)


def canary(agg: pd.Series, eem: pd.Series, asof: pd.Timestamp) -> dict:
    """Spec 11e warning light (= 6T T2 cap, information only): 100% / 50% / 0% for 0 / 1 / 2 negative momenta."""
    a, e = m13612(agg.sort_index()).loc[:asof].dropna(), m13612(eem.sort_index()).loc[:asof].dropna()
    bad = int(a.iloc[-1] < 0) + int(e.iloc[-1] < 0)
    return dict(canary=(1., .5, 0.)[bad], canary_agg=float(a.iloc[-1]), canary_eem=float(e.iloc[-1]),
                canary_asof=str(min(a.index[-1], e.index[-1]).date()))


def main():
    with read_only_external():
        inputs = load_inputs()
        agg, eem = load('AGG.US_TR', refresh=False), load('EEM.US_TR', refresh=False)
    br = breadth()
    idx = inputs.index
    gf = build_signals(inputs)                             # Tuesday keys: information rows only
    v = run_vol.target_weight(inputs.eq_ret)
    live = v.shift(2)
    full = pd.Series(1., index=idx)                       # spec 11d: B at 100% during an episode
    half = np.maximum(live.fillna(0.), .5)                # spec 11d: S at max(live vol size, 50%)
    ratio = deemer_ratio(br.adv, br.dec)
    td = thrust_days(ratio)
    # ---- product keys: read every trading day; the month lock stays (first day of the month sets the state,
    #      mid-month exit only if DMAS < 30). Without the lock the Sharpe falls to 0.53 (info row). ----
    days = idx[idx >= '1993-04-01']
    dm = pd.read_csv(ROOT / 'research/us_gauge/out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0].reindex(days)
    rz = rev_zscore(revisions(inputs.fy1, inputs.fy2, detect_rolls(inputs.fy1, inputs.fy2), window=63)).reindex(days)
    cash_d = (dm < 45) & (rz < 0)
    s1 = states(days, cash_d, dm < 30, dm, None)
    w1, w2 = to_weights(idx, s1), v2_from_states(inputs, s1).reindex(idx)
    epf = f_episodes(idx, s1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    wf = floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)
    eps_s, eps_b = shock_episodes(inputs.px, w1), thrust_episodes(inputs.px, w1, td)
    ws, wb = layer(idx, eps_s, half), layer(idx, eps_b, full)
    ws_full = layer(idx, eps_s, full)
    w_live = act_on_change(idx, (desired_from_states(inputs, s1, v, live, td) * 20).round() / 20)[0]
    core = pd.concat([w2, wf, ws, wb], axis=1).max(axis=1).where(w2.notna())
    desired = pd.concat([s1.reindex(idx).ffill() * v, wf.shift(-2), ws.shift(-2), wb.shift(-2)], axis=1).max(axis=1)
    desired5 = (desired * 20).round() / 20                 # sizes in steps of 5 points
    wc, held_now, trades = act_on_change(idx, desired5)
    w_raw = act_on_change(idx, desired)[0]
    # ---- information rows: keys on Tuesdays only (the earlier rule), and daily keys without the month lock ----
    s_tue = states(gf.loc['1993-04-01':].index, gf.cash_R1, gf.deep, gf.dmas, None)
    d_tue = desired_from_states(inputs, s_tue, v, half, td, full)
    w_tue = act_on_change(idx, d_tue)[0]
    w_wed = committee(idx, s_tue.index[idx.get_indexer(s_tue.index) + 2 < len(idx)], d_tue)
    w_free = act_on_change(idx, desired_from_states(inputs, (~cash_d).astype(float), v, half, td, full))[0]
    r = net_returns(core, inputs.eq_ret, inputs.cash_ret).loc['1999-02-01':]
    bh = inputs.eq_ret.loc[r.index]
    cash = inputs.cash_ret.reindex(r.index)
    # ---- single source of truth: the product built step by step, same period, same execution ----
    CORE = '6. CORE: daily, 5-pt sizes, trade next day if gap >= 10 pts'
    steps = {'S&P 500 buy & hold': None, '1. Keys only (100% or 0%)': w1, '2. + volatility sizing': w2,
             '3. + fear-over re-entry (F)': pd.concat([w2, wf], axis=1).max(axis=1).where(w2.notna()),
             '4. + shock re-entry (S, max(live, 50%))': pd.concat([w2, wf, ws], axis=1).max(axis=1).where(w2.notna()),
             '5. + breadth re-entry (B, 100%)': core,
             CORE: wc,
             'info: same, sizes not rounded': w_raw,
             'info: S and B at live vol size (spec 11b)': w_live,
             'info: S at 100% (spec 11c)': act_on_change(idx, (pd.concat([s1.reindex(idx).ffill() * v, wf.shift(-2), ws_full.shift(-2), wb.shift(-2)], axis=1).max(axis=1) * 20).round() / 20)[0],
             'info: keys on Tuesdays, act next day': w_tue,
             'info: keys on Tuesdays, wait for Wednesday': w_wed,
             'info: keys daily, no month lock': w_free}
    rets = {k: (bh if wk is None else net_returns(wk, inputs.eq_ret, inputs.cash_ret).loc[r.index]) for k, wk in steps.items()}
    full_lab = f'Full {r.index[0]:%Y-%m} → {r.index[-1]:%Y-%m-%d}'
    wins = [(full_lab, None, None)] + [w_ for w_ in WINDOWS if not w_[0].startswith('FULL')]
    rows = []
    for name, a, b in wins:
        for k, x in rets.items():
            t_, s_, d_ = stats(x.loc[a:b], cash)
            nd = len(x.loc[a:b])
            rows.append(dict(window=name, step=k, total=t_, cagr=(1 + t_) ** (252 / nd) - 1 if nd > 250 else np.nan, sharpe=s_, maxdd=d_))
    bu = pd.DataFrame(rows)
    bu.to_csv(ROOT / 'research/us_gauge/out/core_buildup.csv', index=False)
    L = ['# The core, built step by step (single source of truth)', '',
         f'Every number on the slide comes from row 6 (the CORE). Rows marked info are other timing rules, for comparison only. Period {r.index[0].date()} → {r.index[-1].date()}. '
         'Signals at the close of day t, trade at the close of day t+1, earn from day t+2. '
         '10 bp per unit of weight traded. Equity = S&P 500 with dividends; cash = 3-month T-bill, then BIL from 2007-06. '
         'ETF fee (0.09%/yr) and taxes not deducted. Code: `core_slide_data.py`.', '']
    for metric, fmt, title in (('sharpe', '{:.2f}', 'Sharpe'), ('cagr', '{:.1%}', 'Return a year'),
                               ('total', '{:+.0%}', 'Total return'), ('maxdd', '{:.1%}', 'Worst fall')):
        piv = bu.pivot_table(index='window', columns='step', values=metric, sort=False)[list(steps)]
        L += [f'## {title}', '', '| Period | ' + ' | '.join(steps) + ' |', '|---|' + '---:|' * len(steps)]
        L += [f'| {w_} | ' + ' | '.join('' if pd.isna(v_) else fmt.format(v_) for v_ in row) + ' |' for w_, row in piv.iterrows()]
        L.append('')
    (ROOT / 'research/us_gauge/RESULTS_core.md').write_text('\n'.join(L))
    per = []
    for name, a, b in wins:
        tc, sc, dc = stats(rets[CORE].loc[a:b], cash)
        tb, sb, db = stats(bh.loc[a:b], cash)
        per.append(dict(window=name, core_total=tc, core_sharpe=sc, core_dd=dc, bh_total=tb, bh_sharpe=sb, bh_dd=db))
    y = idx[idx >= '2025-01-01']
    trades = [t_ for t_ in trades if t_['signal'] >= '2025-01-01']
    last = idx[-1]
    sig = inputs.eq_ret.rolling(21).std().iloc[-1] * np.sqrt(252)
    out = dict(
        asof=str(last.date()), breadth_asof=str(br.index[-1].date()),
        dates=[str(d.date()) for d in y], px=[round(float(p), 2) for p in inputs.px.loc[y]],
        w=[None if pd.isna(x) else round(float(x), 3) for x in wc.loc[y]],
        tgt=[None if pd.isna(x) else round(float(x), 3) for x in (v * (w1 > 0)).loc[y]],
        trades=trades,
        ma200=[round(float(x), 2) for x in inputs.px.rolling(200).mean().loc[y]],
        now=dict(weight=float(held_now), in_force=float(wc.iloc[-1]), desired=float(desired5.iloc[-1]), desired_raw=float(desired.iloc[-1]), v2=float(w2.iloc[-1]), target=float(v.iloc[-1]), sigma21=float(sig),
                 dmas=float(dm.iloc[-1]), rev_z=float(rz.iloc[-1]), r1_cash=bool(cash_d.iloc[-1]), state=float(s1.iloc[-1]),
                 vix=float(inputs.vix.iloc[-1]), breadth_ratio=float(ratio.iloc[-1]),
                 fall_1y=float(1 - inputs.px.iloc[-1] / inputs.px.iloc[-252:].max()),
                 last_decision=str(idx[-1].date()), shock_trigger=float(inputs.px.iloc[-252:].max() * .85),
                 f_on=bool(wf.iloc[-1] > 0), s_on=bool(ws.iloc[-1] > 0), b_on=bool(wb.iloc[-1] > 0),
                 **canary(agg, eem, last)),
        periods=per, full_label=full_lab)
    (ROOT / 'mockups/core_slide_data.json').write_text(json.dumps(out))
    print(json.dumps(out['now'], indent=1), out['asof'], out['breadth_asof'])
    for p in per:
        print(f"{p['window']:42s} core {p['core_total']:+7.1%} {p['core_sharpe']:5.2f} {p['core_dd']:6.1%} | BH {p['bh_total']:+7.1%} {p['bh_sharpe']:5.2f} {p['bh_dd']:6.1%}")


if __name__ == '__main__':
    main()
