"""S-live: shock re-entry sized by the live volatility target — pre-registered in spec section 6sl (commit 5a8a510).

Europe (STOXX 600, R0 timing x vol size) = the test, one run. US product = information only.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_k1, run_s2, run_vol
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.run_step1 import build_signals
from research.us_gauge.run_m1 import build as build_m1
from research.us_gauge.run_p import decision_days, episodes as f_episodes
from research.us_gauge.run_c import floor_weights, sharpe
from research.us_gauge.run_fast import states, to_weights, v2_from_states
from research.us_gauge.run_s import shock_episodes, europe_inputs, eur_cash, FREQS
from research.us_gauge.trading import net_returns
from research.us_gauge.evaluate import metrics
from research.us_gauge.stress_periods import stats
from research.us_gauge.combo_f import WINDOWS

HERE = Path(__file__).resolve().parent
EU_START = '1994-02-01'
MODES = ('frozen', 'live', 'full')


def s_weight(index: pd.DatetimeIndex, ep: pd.DataFrame, v: pd.Series, mode: str) -> pd.Series:
    """S weight on earn days [i_entry+2, i_end+2): frozen = v(trigger close); live = v(t) on earn day t+2; full = 1."""
    s = pd.Series(0., index=index)
    vl = v.reindex(index).shift(2)
    for r in ep.itertuples():
        a, b = r.i_entry + 2, min(r.i_end + 2, len(index))
        if mode == 'frozen':
            s.iloc[a:b] = v.reindex(index).iloc[r.i_entry]
        elif mode == 'live':
            s.iloc[a:b] = vl.iloc[a:b].fillna(0.).to_numpy()
        else:
            s.iloc[a:b] = 1.
    return s


def eu_run(eu: pd.DataFrame, dmas: pd.Series, freq: str) -> dict:
    """Returns of base (R0 x vol size) and base+S in the three S sizings, plus the episodes."""
    idx = eu.index
    d = decision_days(idx, EU_START, freq)
    m = dmas.reindex(d)
    st = states(d, m < 45, m < 30, m, None)
    wb, wv = to_weights(idx, st), v2_from_states(eu, st).reindex(idx)
    ep = shock_episodes(eu.px, wb)
    v = run_vol.target_weight(eu.eq_ret)
    R = lambda w: net_returns(w, eu.eq_ret, eu.cash_ret).where(w.notna())
    out = {'base': R(wv)}
    for mo in MODES:
        out[mo] = R(np.maximum(wv, s_weight(idx, ep, v, mo)).where(wv.notna()))
    ret = pd.DataFrame(out).dropna()
    return dict(ret=ret, ep=ep)


def episode_pnl(ret: pd.DataFrame, ep: pd.DataFrame, index) -> pd.DataFrame:
    rows = []
    for r in ep.itertuples():
        days = index[r.i_entry + 2:r.i_end + 2].intersection(ret.index)
        g = {c: (1 + ret.loc[days, c]).prod() - 1 for c in ret}
        rows.append(dict(high=r.high.date(), low=r.low.date(), entry=r.entry.date(), end=r.end.date(), reason=r.reason,
                         fall=r.L / r.H - 1, **g))
    return pd.DataFrame(rows)


def main():
    panel, c5, _, macro, _ = build_m1()   # first: macro_health's src package must load before market_store's
    with read_only_external():
        inputs = load_inputs()

    # ---------- Europe: the test ----------
    eu = europe_inputs()
    src = eur_cash().attrs['source']
    eu.loc[eu.index < pd.Timestamp(EU_START), 'cash_ret'] = np.nan
    dm = pd.read_csv(HERE / 'out/dmas_sxxp_daily.csv', index_col=0, parse_dates=True).iloc[:, 0]
    main_run = eu_run(eu, dm, 'W-TUE')
    ret = main_run['ret']
    cash = eu.cash_ret.reindex(ret.index)
    sh = {c: sharpe(ret[c], cash) for c in ret}
    dd = {c: metrics(ret[c], cash)['maxdd'] for c in ret}
    wd = []
    for f in FREQS:
        r_ = eu_run(eu, dm, f)['ret']
        c_ = eu.cash_ret.reindex(r_.index)
        wd.append((f, sharpe(r_['frozen'], c_), sharpe(r_['live'], c_)))
    wf, wl = np.mean([w[1] for w in wd]), np.mean([w[2] for w in wd])
    checks = [('1. FULL Sharpe S-live > S-frozen', f'{sh["live"]:.4f} > {sh["frozen"]:.4f}', sh['live'] > sh['frozen']),
              ('2. Weekday-mean Sharpe S-live > S-frozen', f'{wl:.4f} > {wf:.4f} (' + ', '.join(f'{f[-3:]} {l:.3f}/{fz:.3f}' for f, fz, l in wd) + ')', wl > wf),
              ('3. FULL maxDD S-live >= S-frozen - 0.02', f'{dd["live"]:.4f} vs {dd["frozen"]:.4f}', dd['live'] >= dd['frozen'] - .02)]
    verdict = all(c[2] for c in checks)
    eu_po = eu.assign(eq_ret=eu.eq_ret_price_only)
    po = eu_run(eu_po, dm, 'W-TUE')['ret']
    cpo = eu.cash_ret.reindex(po.index)
    epn = episode_pnl(ret, main_run['ep'], eu.index)

    # ---------- US product: information only ----------
    etf = run_s2.load_etfs()
    _, held_m1 = run_s2.tilt_returns(macro, etf, panel.r_rf)
    with read_only_external():
        levels = pd.DataFrame({t: load(t + '.US_TR', refresh=False).sort_index() for t in etf.columns})
    idx = inputs.index
    gf = build_signals(inputs)
    s1 = states(gf.loc['1993-04-01':].index, gf.cash_R1, gf.deep, gf.dmas, None)
    w1, w2 = to_weights(idx, s1), v2_from_states(inputs, s1)
    epf = f_episodes(idx, s1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    w2F = np.maximum(w2, floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)).where(w2.notna())
    ep1, v = shock_episodes(inputs.px, w1), run_vol.target_weight(inputs.eq_ret)
    wS = {mo: np.maximum(w2F, s_weight(idx, ep1, v, mo)).where(w2F.notna()) for mo in MODES}
    R = lambda w: net_returns(w, inputs.eq_ret, inputs.cash_ret)
    END = pd.Timestamp('2026-08-31')
    k_idx = inputs.loc['1999-01-01':END].index
    pidx = idx[idx.get_loc(k_idx[0]) - 1:idx.get_loc(END) + 1]
    daily = run_k1.etf_daily_returns(levels, pidx).reindex(k_idx)
    picks = {d_.to_period('M'): list(r.index[r.eq(1)]) for d_, r in held_m1.iterrows()}
    K = lambda w: run_k1.combine(daily, picks, w, inputs.cash_ret)[0].K1
    us = pd.DataFrame({'Buy & hold': inputs.eq_ret, 'V2+F': R(w2F), 'V2+F+S frozen': R(wS['frozen']),
                       'V2+F+S live': R(wS['live']), 'V2+F+S full': R(wS['full']),
                       'K1+F+S frozen macro-only': K(wS['frozen']), 'K1+F+S live macro-only': K(wS['live'])}).loc['1999-02-01':END]
    cu = inputs.cash_ret.reindex(us.index)
    t = pd.DataFrame([dict(window=n, rule=c, **dict(zip(('total', 'sharpe', 'maxdd'), stats(us.loc[a:b, c], cu))))
                      for n, a, b in WINDOWS for c in us])
    t.to_csv(HERE / 'out/s_live_us.csv', index=False)
    epn.to_csv(HERE / 'out/s_live_eu_episodes.csv', index=False)

    # ---------- report ----------
    L = ['# S-live: shock re-entry sized by the live volatility target', '',
         f'**Europe (the test, one run): {"PASS" if verdict else "FAIL"}**', '',
         f'Pre-registered in spec 6sl (commit 5a8a510). STOXX 600, {ret.index[0].date()} → {ret.index[-1].date()}. Cash: {src}. '
         'Equity = SXXP price change + 3%/252. Base = R0 timing (Tuesday) x vol size (V2 mechanics). Episodes: '
         f'{len(main_run["ep"])}. Code: `run_s_live.py`.', '',
         '| Gate | Evidence | Result |', '|---|---|---|']
    L += [f'| {n} | {e} | {"PASS" if p else "FAIL"} |' for n, e, p in checks]
    L += ['', '| Period | Base (no S) | +S frozen | +S live | +S full (report) |', '|---|---:|---:|---:|---:|']
    for nm, a, b in (('FULL Sharpe', None, None), ('H1 to 2006 Sharpe', None, '2006-12-31'), ('H2 from 2007 Sharpe', '2007-01-01', None)):
        c_ = cash.loc[a:b]
        L.append(f'| {nm} | ' + ' | '.join(f'{sharpe(ret[c].loc[a:b], c_):.3f}' for c in ('base', *MODES)) + ' |')
    L.append('| FULL return a year | ' + ' | '.join(f'{metrics(ret[c], cash)["cagr"]:.2%}' for c in ('base', *MODES)) + ' |')
    L.append('| FULL worst fall | ' + ' | '.join(f'{dd[c]:.1%}' for c in ('base', *MODES)) + ' |')
    L.append('| Price-only FULL Sharpe | ' + ' | '.join(f'{sharpe(po[c], cpo):.3f}' for c in ('base', *MODES)) + ' |')
    L += ['', '### Europe episodes (return inside the episode)', '',
          '| 1-year high | Low | Entry | End | Reason | Fall | Base | +S frozen | +S live | +S full |', '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    L += [f'| {r.high} | {r.low} | {r.entry} | {r.end} | {r.reason} | {r.fall:.1%} | {r.base:+.1%} | {r.frozen:+.1%} | {r.live:+.1%} | {r.full:+.1%} |'
          for r in epn.itertuples()]
    order = list(us.columns)
    for metric, fmt, title in (('sharpe', '{:.2f}', 'Sharpe'), ('total', '{:+.1%}', 'Total return'), ('maxdd', '{:.1%}', 'Worst fall')):
        piv = t.pivot_table(index='window', columns='rule', values=metric, sort=False)[order]
        L += ['', f'## US product by period — {title} (information only; 1999-02 → 2026-08)', '',
              '| Window | ' + ' | '.join(order) + ' |', '|---|' + '---:|' * len(order)]
        L += [f'| {w} | ' + ' | '.join('' if pd.isna(x) else fmt.format(x) for x in row) + ' |' for w, row in piv.iterrows()]
    L += ['', 'US S episodes (R1 base): ' + ', '.join(f'{r.entry.date()}→{r.end.date()} ({r.reason}, v at entry {v.iloc[r.i_entry]:.2f})' for r in ep1.itertuples()), '']
    (HERE / 'RESULTS_s_live.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
