"""L: re-entry layers at full weight — pre-registered in spec section 6L (commit 78ee17e). One run.

Six fresh indices (price only, cash 0%, 10 bp): S-full vs S-live on the plain trend rule x vol size.
US product effect = information only (in-sample).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_vol
from research.us_gauge.data import load, load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.gauge import dmas_on
from research.us_gauge.run_p import decision_days, episodes as f_episodes
from research.us_gauge.run_c import floor_weights, sharpe
from research.us_gauge.run_fast import states, to_weights, v2_from_states
from research.us_gauge.run_s import shock_episodes, placebo_spans
from research.us_gauge.run_b import breadth, deemer_ratio, thrust_days, thrust_episodes
from research.us_gauge.revisions import detect_rolls, revisions, rev_zscore
from research.us_gauge.trading import net_returns
from research.us_gauge.evaluate import metrics
from research.us_gauge.core_slide_data import layer, act_on_change

HERE = Path(__file__).resolve().parent
MARKETS = ['N225', 'TWII', 'GSPTSE', 'HSI', 'KS11', 'AXJO']
N_PLACEBO = 1000


def market(name):
    with read_only_external():
        px = load(f'{name}.INDX', refresh=False).sort_index().astype(float)
    px = px[px.notna() & (px > 0)]
    px = px[~px.index.duplicated()]
    r = px.pct_change(fill_method=None)
    big = r[r.abs() > .25]
    df = pd.DataFrame({'px': px, 'eq_ret': r, 'cash_ret': 0.})
    return df, big


def fixed(index, ep, w):
    s = pd.Series(0., index=index)
    for a, b in ep:
        s.iloc[a + 2:min(b + 2, len(index))] = w
    return s


def run_market(name):
    df, big = market(name)
    idx = df.index
    d = decision_days(idx, str(idx[252].date()), 'W-TUE')
    m = dmas_on(df.px, d)
    st = states(d, m < 45, m < 30, m, None)
    wb, wv = to_weights(idx, st), v2_from_states(df, st).reindex(idx)
    ep = shock_episodes(df.px, wb)
    v = run_vol.target_weight(df.eq_ret)
    spans = list(zip(ep.i_entry, ep.i_end))
    live = layer(idx, ep, v.shift(2).fillna(0.))
    R = lambda w: net_returns(w, df.eq_ret, df.cash_ret).where(w.notna())
    w_live = np.maximum(wv, live).where(wv.notna())
    w_full = np.maximum(wv, fixed(idx, spans, 1.)).where(wv.notna())
    w_half = np.maximum(wv, np.maximum(live, fixed(idx, spans, .5))).where(wv.notna())
    ret = pd.DataFrame({'base': R(wv), 'live': R(w_live), 'full': R(w_full), 'half': R(w_half)}).dropna()
    rows = []
    for r in ep.itertuples():
        days = idx[r.i_entry + 2:r.i_end + 2]
        rows.append(dict(market=name, entry=r.entry, end=r.end, reason=r.reason, fall=r.L / r.H - 1 if hasattr(r, 'H') else np.nan,
                         idx_ret=float((1 + df.eq_ret.reindex(days)).prod() - 1), ndays=len(days)))
    return dict(df=df, wv=wv, wb=wb, ret=ret, ep=ep, rows=rows, big=big, n=len(ep))


def cluster(E):
    E = E.sort_values('entry').reset_index(drop=True)
    cid, start, out = -1, None, []
    for t in E.entry:
        if start is None or (t - start).days > 14:
            cid, start = cid + 1, t
        out.append(cid)
    E['cluster'] = out
    return E


def us_info():
    """US product (spec 11b) with S and B at live / 50% floor / 100%. In-sample, information only."""
    with read_only_external():
        I = load_inputs()
    idx = I.index
    br = breadth()
    td = thrust_days(deemer_ratio(br.adv, br.dec))
    v = run_vol.target_weight(I.eq_ret)
    days = idx[idx >= '1993-04-01']
    dm = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0].reindex(days)
    rz = rev_zscore(revisions(I.fy1, I.fy2, detect_rolls(I.fy1, I.fy2), window=63)).reindex(days)
    s1 = states(days, (dm < 45) & (rz < 0), dm < 30, dm, None)
    w1 = to_weights(idx, s1)
    epf = f_episodes(idx, s1, I.vix, I.vix, I.px, .5, 30.)
    wf = floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)
    sp = [(r.i_entry, r.i_end) for e in (shock_episodes(I.px, w1), thrust_episodes(I.px, w1, td)) for r in e.itertuples()]
    live = v.shift(2)
    out = {}
    for name, lay in (('S, B at live size (core)', None), ('S, B at max(live, 50%)', .5), ('S, B at 100%', 1.)):
        l = pd.Series(0., index=idx)
        for a, b in sp:
            seg = live.iloc[a + 2:min(b + 2, len(idx))]
            l.iloc[a + 2:min(b + 2, len(idx))] = np.maximum(seg.to_numpy(), lay if lay else 0.)
        des = pd.concat([s1.reindex(idx).ffill() * v, wf.shift(-2), l.shift(-2)], axis=1).max(axis=1)
        w = act_on_change(idx, (des * 20).round() / 20)[0]
        out[name] = net_returns(w, I.eq_ret, I.cash_ret).loc['1999-02-01':]
    cash = I.cash_ret.reindex(out['S, B at 100%'].index)
    return out, cash


def main():
    M = {n: run_market(n) for n in MARKETS}
    E = cluster(pd.DataFrame([r for n in MARKETS for r in M[n]['rows']]))
    sh = {n: {k: sharpe(M[n]['ret'][k], pd.Series(0., index=M[n]['ret'].index)) for k in ('base', 'live', 'full', 'half')} for n in MARKETS}
    dd = {n: {k: metrics(M[n]['ret'][k], pd.Series(0., index=M[n]['ret'].index))['maxdd'] for k in ('base', 'live', 'full', 'half')} for n in MARKETS}
    dS = {n: sh[n]['full'] - sh[n]['live'] for n in MARKETS}
    cl = E.groupby('cluster').agg(first=('entry', 'min'), markets=('market', lambda s: ', '.join(s)), mean_ret=('idx_ret', 'mean'))
    yr = cl['first'].dt.year
    cl_ex = cl[~((yr == 2020) & (cl['first'].dt.month <= 6)) & ~((yr == 2025) & (cl['first'].dt.month <= 6))]
    # placebo
    rng = np.random.default_rng(7)
    real_gain = np.mean([sh[n]['full'] - sh[n]['base'] for n in MARKETS])
    pl = np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        g = []
        for n in MARKETS:
            m_ = M[n]
            idx = m_['df'].index
            spans = placebo_spans(rng, m_['n'], m_['df'].px, m_['wb'])
            w = np.maximum(m_['wv'], fixed(idx, spans, 1.)).where(m_['wv'].notna())
            r = net_returns(w, m_['df'].eq_ret, m_['df'].cash_ret).reindex(m_['ret'].index)
            g.append(sharpe(r, pd.Series(0., index=r.index)) - sh[n]['base'])
        pl[i] = np.mean(g)
    p_val = float((pl >= real_gain).mean())
    gates = [
        ('G1 mean Sharpe gain > 0 and positive in >= 4 of 6', f'mean {np.mean(list(dS.values())):+.3f}; positive in {sum(x > 0 for x in dS.values())} of 6',
         np.mean(list(dS.values())) > 0 and sum(x > 0 for x in dS.values()) >= 4),
        ('G2 worst fall no more than 3 pts deeper, every market', '; '.join(f'{n} {dd[n]["full"]:.1%} vs {dd[n]["live"]:.1%}' for n in MARKETS),
         all(dd[n]['full'] >= dd[n]['live'] - .03 for n in MARKETS)),
        ('G3 > half of clusters positive, also without 2020 and 2025', f'{int((cl.mean_ret > 0).sum())} of {len(cl)}; without 2020/2025: {int((cl_ex.mean_ret > 0).sum())} of {len(cl_ex)}',
         (cl.mean_ret > 0).mean() > .5 and (cl_ex.mean_ret > 0).mean() > .5),
        ('G4 worst single episode >= -15%', f'{E.idx_ret.min():+.1%} ({E.loc[E.idx_ret.idxmin(), "market"]} {E.loc[E.idx_ret.idxmin(), "entry"].date()})', E.idx_ret.min() >= -.15),
        ('G5 placebo p <= 0.10', f'real pooled gain {real_gain:+.3f}; p = {p_val:.3f} (placebo median {np.median(pl):+.3f})', p_val <= .10)]
    ok = {g[0][:2]: g[2] for g in gates}
    if len(cl) < 10:
        verdict = 'TOO FEW TO CONCLUDE (fewer than 10 clusters): no change'
    elif all(ok.values()):
        verdict = 'PASS: S and B at 100% during episodes'
    elif ok['G1'] and ok['G3'] and ok['G5']:
        verdict = 'FALLBACK: S and B at max(live size, 50%)'
    else:
        verdict = 'FAIL: keep S and B at the live size'
    us, cash = us_info()
    L = ['# L: re-entry layers at full weight (pre-registered, spec 6L, one run)', '', f'**Verdict: {verdict}**', '',
         'Six fresh indices, price only, cash 0%, 10 bp. Base = plain trend rule (DMAS 45/30, Tuesday, month lock) x vol size. '
         'S-live vs S-full = shock episodes at live vol size vs at 100%. B has no fresh sample; it follows S by symmetry. Code: `run_l.py`.', '',
         '| Gate | Evidence | Result |', '|---|---|---|']
    L += [f'| {n} | {e} | {"PASS" if p else "FAIL"} |' for n, e, p in gates]
    L += ['', '## By market', '', '| Market | From | Episodes | Sharpe base | S-live | S-full | S-50% floor | Worst fall S-live | S-full |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    L += [f'| {n} | {M[n]["ret"].index[0].date()} | {M[n]["n"]} | {sh[n]["base"]:.3f} | {sh[n]["live"]:.3f} | {sh[n]["full"]:.3f} | {sh[n]["half"]:.3f} | {dd[n]["live"]:.1%} | {dd[n]["full"]:.1%} |' for n in MARKETS]
    L += ['', '## Clusters', '', '| # | First entry | Markets | Mean index return in episode |', '|---:|---|---|---:|']
    L += [f'| {i} | {r["first"].date()} | {r.markets} | {r.mean_ret:+.1%} |' for i, r in cl.iterrows()]
    L += ['', '## Episodes', '', '| Market | Entry | End | Reason | Days | Index return |', '|---|---|---|---|---:|---:|']
    L += [f'| {r.market} | {r.entry.date()} | {r.end.date()} | {r.reason} | {r.ndays} | {r.idx_ret:+.1%} |' for r in E.itertuples()]
    L += ['', '## Data notes (daily moves above 25%, not removed)', '']
    L += [f'- {n}: ' + (', '.join(f'{d.date()} {x:+.0%}' for d, x in M[n]['big'].items()) or 'none') for n in MARKETS]
    L += ['', '## US product, information only (in-sample, 1999-02 → last day)', '', '| Variant | Sharpe | Return a year | Worst fall | COVID rebound | 2009 rebound |', '|---|---:|---:|---:|---:|---:|']
    for k, r in us.items():
        mt = metrics(r, cash)
        cv = (1 + r.loc['2020-03-24':'2020-08-31']).prod() - 1
        r9 = (1 + r.loc['2009-03-10':'2009-12-31']).prod() - 1
        L.append(f'| {k} | {sharpe(r, cash):.2f} | {mt["cagr"]:.1%} | {mt["maxdd"]:.1%} | {cv:+.0%} | {r9:+.0%} |')
    (HERE / 'RESULTS_l.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
