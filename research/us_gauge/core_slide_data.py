"""Data for the core slide (spec section 11): core = 100% S&P 500 x max(V2, F floor, S-live, B).

Writes mockups/core_slide_data.json: 2026 daily price and weight, Tuesday decisions, current state of every
layer, and back-test stats of the core vs buy and hold by period (1999-02 → last day, daily, 10 bp).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_vol
from research.us_gauge.data import load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals
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


def main():
    with read_only_external():
        inputs = load_inputs()
    br = breadth()
    idx = inputs.index
    gf = build_signals(inputs)
    s1 = states(gf.loc['1993-04-01':].index, gf.cash_R1, gf.deep, gf.dmas, None)
    w1, w2 = to_weights(idx, s1), v2_from_states(inputs, s1).reindex(idx)
    v = run_vol.target_weight(inputs.eq_ret)
    live = v.shift(2)
    epf = f_episodes(idx, s1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    wf = floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)
    eps = shock_episodes(inputs.px, w1)
    ws = layer(idx, eps, live)
    ratio = deemer_ratio(br.adv, br.dec)
    epb = thrust_episodes(inputs.px, w1, thrust_days(ratio))
    wb = layer(idx, epb, live)
    core = pd.concat([w2, wf, ws, wb], axis=1).max(axis=1).where(w2.notna())
    r = net_returns(core, inputs.eq_ret, inputs.cash_ret).loc['1999-02-01':]
    bh = inputs.eq_ret.loc[r.index]
    cash = inputs.cash_ret.reindex(r.index)
    per = []
    for name, a, b in WINDOWS:
        tc, sc, dc = stats(r.loc[a:b], cash)
        tb, sb, db = stats(bh.loc[a:b], cash)
        per.append(dict(window=name, core_total=tc, core_sharpe=sc, core_dd=dc, bh_total=tb, bh_sharpe=sb, bh_dd=db))
    y = idx[idx >= '2026-01-01']
    last = idx[-1]
    dec = gf.loc['2026-01-01':]
    sig = inputs.eq_ret.rolling(21).std().iloc[-1] * np.sqrt(252)
    out = dict(
        asof=str(last.date()), breadth_asof=str(br.index[-1].date()),
        dates=[str(d.date()) for d in y], px=[round(float(p), 2) for p in inputs.px.loc[y]],
        w=[None if pd.isna(x) else round(float(x), 3) for x in core.loc[y]],
        dec=[dict(d=str(d.date()), dmas=int(round(rw.dmas)), rev=round(float(rw.rev_z), 2), cash=bool(rw.cash_R1)) for d, rw in dec.iterrows()],
        now=dict(weight=float(core.iloc[-1]), v2=float(w2.iloc[-1]), target=float(v.iloc[-1]), sigma21=float(sig),
                 dmas=float(gf.dmas.iloc[-1]), rev_z=float(gf.rev_z.iloc[-1]), r1_cash=bool(gf.cash_R1.iloc[-1]),
                 vix=float(inputs.vix.iloc[-1]), breadth_ratio=float(ratio.iloc[-1]),
                 fall_1y=float(1 - inputs.px.iloc[-1] / inputs.px.iloc[-252:].max()),
                 f_on=bool(wf.iloc[-1] > 0), s_on=bool(ws.iloc[-1] > 0), b_on=bool(wb.iloc[-1] > 0)),
        periods=per)
    (ROOT / 'mockups/core_slide_data.json').write_text(json.dumps(out))
    print(json.dumps(out['now'], indent=1), out['asof'], out['breadth_asof'])
    for p in per:
        print(f"{p['window']:42s} core {p['core_total']:+7.1%} {p['core_sharpe']:5.2f} {p['core_dd']:6.1%} | BH {p['bh_total']:+7.1%} {p['bh_sharpe']:5.2f} {p['bh_dd']:6.1%}")


if __name__ == '__main__':
    main()
