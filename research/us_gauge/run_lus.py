"""L-US: shock re-entry at 100% on S&P 500 1928-1954 — pre-registered in spec 6LU (commit ed36b11). One run."""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_vol
from research.us_gauge.data import load
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.gauge import dmas_on
from research.us_gauge.run_p import decision_days
from research.us_gauge.run_c import sharpe
from research.us_gauge.run_fast import states, to_weights, v2_from_states
from research.us_gauge.run_s import shock_episodes, placebo_spans
from research.us_gauge.trading import net_returns
from research.us_gauge.evaluate import metrics
from research.us_gauge.core_slide_data import layer

HERE = Path(__file__).resolve().parent
END = '1954-12-31'
DAYS = ('W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI')


def inputs():
    with read_only_external():
        px = load('GSPC.INDX', refresh=False).sort_index().astype(float).loc[:END]
        div = load('SHILLER_SPX_DIV', refresh=False).sort_index()
    px = px[px.notna() & (px > 0)]
    px = px[~px.index.duplicated()]
    d = div.reindex(px.index, method='ffill')
    return pd.DataFrame({'px': px, 'eq_ret': px.pct_change(fill_method=None) + d / px / 252, 'cash_ret': 0.})


def full_layer(idx, spans):
    s = pd.Series(0., index=idx)
    for a, b in spans:
        s.iloc[a + 2:min(b + 2, len(idx))] = 1.
    return s


def run(df, dm, freq):
    idx = df.index
    d = decision_days(idx, str(idx[252].date()), freq)
    m = dm.reindex(d)
    st = states(d, m < 45, m < 30, m, None)
    wb = to_weights(idx, st)
    ep = shock_episodes(df.px, wb)
    ws = np.maximum(wb, full_layer(idx, list(zip(ep.i_entry, ep.i_end)))).where(wb.notna())
    R = lambda w: net_returns(w, df.eq_ret, df.cash_ret).where(w.notna())
    ret = pd.DataFrame({'R0': R(wb), 'R0+S': R(ws)}).dropna()
    return dict(ret=ret, ep=ep, wb=wb, st=st)


def main():
    df = inputs()
    idx = df.index
    dm = dmas_on(df.px, idx[252:])
    runs = {f: run(df, dm, f) for f in DAYS}
    T = runs['W-TUE']
    ret, ep = T['ret'], T['ep']
    z = lambda r: pd.Series(0., index=r.index)
    sh = {c: sharpe(ret[c], z(ret)) for c in ret}
    dd = {c: metrics(ret[c], z(ret))['maxdd'] for c in ret}
    wk = [(f, sharpe(runs[f]['ret']['R0'], z(runs[f]['ret'])), sharpe(runs[f]['ret']['R0+S'], z(runs[f]['ret']))) for f in DAYS]
    m0, m1 = np.mean([w[1] for w in wk]), np.mean([w[2] for w in wk])
    rows = []
    for r in ep.itertuples():
        days = idx[r.i_entry + 2:r.i_end + 2]
        rows.append(dict(high=r.high.date(), low=r.low.date(), entry=r.entry.date(), end=r.end.date(), reason=r.reason,
                         fall=r.L / r.H - 1, ndays=len(days), idx_ret=float((1 + df.eq_ret.reindex(days)).prod() - 1)))
    E = pd.DataFrame(rows)
    rng = np.random.default_rng(7)
    pl = []
    for _ in range(1000):
        sp = placebo_spans(rng, len(ep), df.px, T['wb'])
        w = np.maximum(T['wb'], full_layer(idx, sp)).where(T['wb'].notna())
        r_ = net_returns(w, df.eq_ret, df.cash_ret).reindex(ret.index)
        pl.append(sharpe(r_, z(r_)))
    pl = np.array(pl)
    p_val = float((pl >= sh['R0+S']).mean())
    n = len(E)
    gates = [('G1 Sharpe R0+S > R0, Tuesday and weekday mean', f'Tue {sh["R0+S"]:.3f} vs {sh["R0"]:.3f}; mean {m1:.3f} vs {m0:.3f} ('
              + ', '.join(f'{f[-3:]} {b:.3f}/{a:.3f}' for f, a, b in wk) + ')', sh['R0+S'] > sh['R0'] and m1 > m0),
             ('G2 worst fall no more than 3 pts deeper', f'{dd["R0+S"]:.1%} vs {dd["R0"]:.1%}', dd['R0+S'] >= dd['R0'] - .03),
             ('G3 more than half of episodes positive', f'{int((E.idx_ret > 0).sum()) if n else 0} of {n}', n > 0 and (E.idx_ret > 0).mean() > .5),
             ('G4 worst single episode >= -15%', f'{E.idx_ret.min():+.1%}' if n else 'no episode', n > 0 and E.idx_ret.min() >= -.15),
             ('G5 placebo p <= 0.10', f'p = {p_val:.3f} (placebo median {np.median(pl):.3f})', p_val <= .10)]
    ok = {g[0][:2]: g[2] for g in gates}
    if n < 5:
        verdict, cons = 'TOO FEW TO CONCLUDE', '11c stays.'
    elif all(ok.values()):
        verdict, cons = 'PASS', '11c stays (S at 100%).'
    elif ok['G1'] and ok['G3'] and ok['G5'] and not (ok['G2'] and ok['G4']):
        verdict, cons = 'FAIL on G2/G4 only', 'S at max(live, 50%); B stays at 100%.'
    elif not ok['G1'] and not ok['G5']:
        verdict, cons = 'FAIL on G1 and G5', 'S back to the live vol size; B stays at 100%.'
    else:
        verdict, cons = 'MIXED', '11c stays; result recorded.'
    # information: vol-sized base, S-live vs S-full
    st = T['st']
    wv = v2_from_states(df, st).reindex(idx)
    v = run_vol.target_weight(df.eq_ret)
    live = layer(idx, ep, v.shift(2).fillna(0.))
    R = lambda w: net_returns(w, df.eq_ret, df.cash_ret).where(w.notna())
    info = pd.DataFrame({'V': R(wv), 'V+S live': R(np.maximum(wv, live).where(wv.notna())),
                         'V+S full': R(np.maximum(wv, full_layer(idx, list(zip(ep.i_entry, ep.i_end)))).where(wv.notna()))}).dropna()
    L = ['# L-US: shock re-entry at 100%, S&P 500 1928–1954 (pre-registered, spec 6LU, one run)', '',
         f'**Verdict: {verdict}. Consequence (declared before the run): {cons}**', '',
         f'Sample {ret.index[0].date()} → {ret.index[-1].date()} (Tuesday). Equity = GSPC price + Shiller dividends; cash 0%; 10 bp. Code: `run_lus.py`.', '',
         '| Gate | Evidence | Result |', '|---|---|---|']
    L += [f'| {g} | {e} | {"PASS" if p else "FAIL"} |' for g, e, p in gates]
    L += ['', '| Tuesday | R0 | R0+S |', '|---|---:|---:|', f'| Sharpe | {sh["R0"]:.3f} | {sh["R0+S"]:.3f} |',
          f'| Return a year | {metrics(ret["R0"], z(ret))["cagr"]:.2%} | {metrics(ret["R0+S"], z(ret))["cagr"]:.2%} |',
          f'| Worst fall | {dd["R0"]:.1%} | {dd["R0+S"]:.1%} |', '',
          '## Episodes (Tuesday)', '', '| 1-year high | Low | Entry | End | Reason | Fall | Days | Index return |', '|---|---|---|---|---|---:|---:|---:|']
    L += [f'| {r.high} | {r.low} | {r.entry} | {r.end} | {r.reason} | {r.fall:.1%} | {r.ndays} | {r.idx_ret:+.1%} |' for r in E.itertuples()]
    ci = pd.Series(0., index=info.index)
    L += ['', '## Information only: vol-sized base (R0 x vol size)', '', '| | Sharpe | Worst fall |', '|---|---:|---:|']
    L += [f'| {c} | {sharpe(info[c], ci):.3f} | {metrics(info[c], ci)["maxdd"]:.1%} |' for c in info]
    (HERE / 'RESULTS_lus.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
