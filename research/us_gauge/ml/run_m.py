"""Step 7: the one run of the ML sizing study (spec 2026-11-18-ml-reentry-sizing-design.md, frozen in section 12).

    PYTHONPATH=. python -m research.us_gauge.ml.run_m --dry-run    # checks only, stops before any label
    PYTHONPATH=. python -m research.us_gauge.ml.run_m --one-run    # THE run (once; refuses if RESULTS_ml.md exists)

Guards: needs --one-run; refuses if research/us_gauge/RESULTS_ml.md exists, if the SHA-256 of features.csv (and of
the other frozen files) differs from the spec's Freeze section, or if the event list / features recomputed from the
store differ from the frozen files. Labels are computed only after every guard has passed, and only with --one-run.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.us_gauge.ml import labels as lb
from research.us_gauge.ml import models as mo
from research.us_gauge.ml import validate as va
from research.us_gauge.ml.events import scan_events, product_entries, plain_trend_weight, dmas_cached
from research.us_gauge.ml.features import FEATURES, Context, features_for, event_features
from research.us_gauge.ml.panel import (OUT, MARKETS, CLEAN, DATA_END, REGION, load_panel, load_macro,
                                        big_moves_lines, _store_load)

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'research/us_gauge'
SPEC = ROOT / 'docs/superpowers/specs/2026-11-18-ml-reentry-sizing-design.md'
RESULTS = HERE / 'RESULTS_ml.md'
FROZEN = {'features.csv': r'SHA-256 of features.csv: `([0-9a-f]{64})`',
          'events.csv': r'events\.csv `([0-9a-f]{64})`',
          'product_episodes.csv': r'product_episodes\.csv `([0-9a-f]{64})`',
          'product_features.csv': r'product_features\.csv `([0-9a-f]{64})`'}
G4_START = '2000-01-01'
G7_START = '1999-02-01'


# ------------------------------------------------------------------ guards

def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hashes(spec_text: str) -> dict:
    out = {}
    for f, pat in FROZEN.items():
        m = re.search(pat, spec_text)
        if not m:
            raise SystemExit(f'REFUSED: the spec has no frozen hash for {f}')
        out[f] = m.group(1)
    return out


def guard(one_run: bool, dry_run: bool, spec=SPEC, results=RESULTS, out=OUT):
    if not (one_run or dry_run):
        raise SystemExit('REFUSED: this is the one run of a pre-registered study; pass --one-run (or --dry-run for checks only)')
    if one_run and dry_run:
        raise SystemExit('REFUSED: choose --one-run or --dry-run')
    if Path(results).exists():
        raise SystemExit(f'REFUSED: {results} exists; the study has been run (one run only)')
    want = frozen_hashes(Path(spec).read_text())
    for f, h in want.items():
        got = sha256(Path(out) / f)
        if got != h:
            raise SystemExit(f'REFUSED: SHA-256 of {f} is {got}, the spec freezes {h}')
    return want


def load_frozen(out=OUT):
    E = pd.read_csv(out / 'events.csv', parse_dates=['h', 'l', 't', 't_embargo'])
    F = pd.read_csv(out / 'features.csv')
    PE = pd.read_csv(out / 'product_episodes.csv', parse_dates=['h', 'l', 't'])
    G = pd.read_csv(out / 'product_features.csv')
    assert (F.event_id == E.event_id).all() and (G.episode_id == PE.episode_id).all()
    return E, F, PE, G


def verify_frozen(P, macro, E, F, PE, G, dmas=None):
    """The frozen files must be what the frozen code gives on the store today (pre-trigger data only)."""
    bad = []
    for m in MARKETS:
        ev = scan_events(P[m])
        fr = E[E.market == m].sort_values('t')
        if list(ev.t) != list(fr.t) or list(ev.i_t) != list(fr.i_t) or list(ev.i_h) != list(fr.i_h):
            bad.append(f'events {m}')
        wb = plain_trend_weight(P[m], dmas_cached(m, P[m]) if dmas is None else dmas[m])
        pe = product_entries(P[m], wb)
        if list(pe.t) != list(PE[PE.market == m].sort_values('t').t):
            bad.append(f'product episodes {m}')
    ctx = Context(P, macro)
    entries = {m: list(E.loc[E.market == m, 't']) for m in MARKETS}
    for frozen, rows in ((F, E), (G, PE)):
        new = features_for(rows, P, ctx, entries)
        a, b = new[FEATURES].to_numpy(float), frozen[FEATURES].to_numpy(float)
        if not np.allclose(a, b, rtol=1e-8, atol=1e-12, equal_nan=True):
            bad.append('features' if frozen is F else 'product features')
    if bad:
        raise SystemExit('REFUSED: recomputed inputs differ from the frozen files: ' + ', '.join(bad))
    return ctx, entries


# ------------------------------------------------------------------ evaluation of one model's pooled predictions

def fit_blocks(S, X, y, factory):
    """Per block: the fitted model, pi, and the test predictions (same split and asserts as validate.walk_forward)."""
    lab = y.notna().to_numpy()
    out = {}
    for b in va.BLOCKS:
        tr, te = va.split(S, b, lab)
        va.assert_split(S, tr, te, b)
        w = va.cluster_weights(S.cluster[tr])
        m = factory().fit(X[tr], y[tr].astype(int), w)
        out[b.name] = dict(model=m, pi=mo.train_pi(y[tr], w), test=S.index[te],
                           p=pd.Series(m.predict(X[te]), index=S.index[te]))
    return out


def pooled(S, y, R, v, fits) -> pd.DataFrame:
    rows = []
    for name, f in fits.items():
        rows.append(pd.DataFrame({'p': f['p'], 'pi': f['pi'], 'block': name}))
    d = pd.concat(rows)
    d = d.join(S[['event_id', 'market', 'region', 'seen', 'cluster', 't']]).assign(y=y, R=R, v=v)
    d = d[d.y.notna()].copy()
    return va.size_events(d)


def subset_K(d, mask):
    s = d[mask]
    return va.K_stat(s) if len(s) else np.nan


def model_stats(d: pd.DataFrame, n_perm=va.N_PERM) -> dict:
    w = va.cluster_weights(d.cluster)
    auc_e, auc_c, p_auc = va.perm_auc(d, n=n_perm)
    K, p_K, _ = va.perm_K(d, n=n_perm)
    yr = d.t.dt.year
    cl2020 = set(d.cluster[yr == 2020])
    cl0809 = set(d.cluster[yr.isin([2008, 2009])])
    clean = d[d.market.isin(CLEAN)]
    st = dict(n=len(d), clusters=d.cluster.nunique(), auc=auc_e, auc_cluster=auc_c, p_auc=p_auc,
              slope=va.calibration_slope(d.y, d.p, w), brier=va.brier(d.y, d.p, w), brier_base=va.brier(d.y, d.pi, w),
              logloss=va.logloss(d.y, d.p, w), K=K, p_K=p_K,
              K_0009=subset_K(d, d.block.isin(['2000–2004', '2005–2009'])),
              K_1026=subset_K(d, d.block.isin(['2010–2015', '2016–2026'])),
              K_no2020=subset_K(d, ~d.cluster.isin(cl2020)), K_no0809=subset_K(d, ~d.cluster.isin(cl0809)),
              K_region={r: subset_K(d, d.region == r) for r in sorted(d.region.unique())},
              buckets=d.bucket.value_counts().to_dict())
    if len(clean):
        a_e, _, p_c = va.perm_auc(clean, n=n_perm)
        st.update(clean_auc=a_e, clean_p_auc=p_c, clean_K=va.K_stat(clean), clean_n=len(clean), clean_clusters=clean.cluster.nunique())
    else:
        st.update(clean_auc=np.nan, clean_p_auc=np.nan, clean_K=np.nan, clean_n=0, clean_clusters=0)
    return st


def gates_from(st: dict, single_K: dict, g4: dict, g7: dict) -> dict:
    reg = st['K_region']
    g = {
        'G1': (st['auc'] >= .60 and st['p_auc'] <= .01,
               f"AUC {st['auc']:.3f} (cluster-level {st['auc_cluster']:.3f}), permutation p {st['p_auc']:.4f}"),
        'G2': (.5 <= st['slope'] <= 1.5 and st['brier'] < st['brier_base'],
               f"slope {st['slope']:.2f}; Brier {st['brier']:.4f} vs base rate {st['brier_base']:.4f}"),
        'G3': (st['K'] > 0 and st['p_K'] <= .01, f"K {st['K']:+.5f}, permutation p {st['p_K']:.4f}"),
        'G4': (g4['sharpe_ok'] and g4['dd_ok'], g4['text']),
        'G5': (st['K_0009'] > 0 and st['K_1026'] > 0 and st['K_no2020'] > 0 and st['K_no0809'] > 0
               and sum(k > 0 for k in reg.values()) >= 3,
               f"K 2000–09 {st['K_0009']:+.5f}; 2010–26 {st['K_1026']:+.5f}; without 2020 {st['K_no2020']:+.5f}; "
               f"without 2008–09 {st['K_no0809']:+.5f}; regions " + ', '.join(f'{r} {k:+.5f}' for r, k in reg.items())),
        'G6': (st['clean_auc'] > .5 and st['clean_p_auc'] <= .05 and st['clean_K'] > 0,
               f"{st['clean_n']} events / {st['clean_clusters']} clusters: AUC {st['clean_auc']:.3f}, p {st['clean_p_auc']:.4f}, K {st['clean_K']:+.5f}"),
        'G7': (g7['ok'], g7['text']),
        'G8': (all(st['K'] > k for k in single_K.values()),
               f"K {st['K']:+.5f} vs " + ', '.join(f'{n} {k:+.5f}' for n, k in single_K.items())),
    }
    return g


def consequence(g: dict, st: dict, too: bool, g4_dd_only: bool) -> str:
    """Spec section 7, in the order of the Freeze note 15."""
    ok = {k: v[0] for k, v in g.items()}
    if too:
        return 'TOO FEW TO CONCLUDE: no change (sizes stay 11d)'
    if st['p_auc'] > .10:
        return 'STOP: G1 fails at p > 0.10. The file "ML sizing of re-entries" is closed with these data; record in the rejected list'
    if all(ok.values()):
        return 'ADOPT: three-bucket S size in the core (section 8), live kill rules of section 9'
    core4 = ok['G1'] and ok['G2'] and ok['G3'] and ok['G6']
    fails = {k for k, v in ok.items() if not v}
    if core4 and ok['G7'] and fails <= {'G4', 'G5', 'G8'} and (ok['G4'] or g4_dd_only):
        if not ok['G8']:
            return 'ADOPT NOTHING (G8: a one-feature rule does as well); report the one-feature finding for a later pre-registration'
        return 'DEFENSIVE HALF ONLY: the high-risk bucket goes to the live size; no step up to 100%'
    if (st['auc'] >= .60 and .01 < st['p_auc'] <= .10) or (not ok['G3'] and ok['G1'] and ok['G2']):
        return 'REPORT ONLY: retest risk shown on the dashboard as context; sizes stay 11d; live log starts; no adoption before 20 live pooled events'
    return 'NO CHANGE: gates failed outside the declared fallbacks; result recorded'


# ------------------------------------------------------------------ G4: 21-market product simulation

def market_sim(px: pd.Series, dmas: pd.Series, buckets: dict, start=G4_START):
    """Plain trend rule (run_l.run_market) x live vol size + S, 5-pt sizes, 10-pt rule, 10 bp, cash 0%.
    buckets: {i_entry: 'low'|'normal'|'high'}; episodes not in it get the 11d size. Returns (r_11d, r_model, ep)."""
    from research.us_gauge import run_vol
    from research.us_gauge.run_p import decision_days
    from research.us_gauge.run_fast import states, to_weights
    from research.us_gauge.run_s import shock_episodes
    from research.us_gauge.core_slide_data import layer, act_on_change
    from research.us_gauge.trading import net_returns
    idx = px.index
    eq = px.pct_change(fill_method=None)
    d = decision_days(idx, str(idx[252].date()), 'W-TUE')
    m = dmas.reindex(d)
    st = states(d, m < 45, m < 30, m, None)
    wb = to_weights(idx, st)
    ep = shock_episodes(px, wb)
    v = run_vol.target_weight(eq)
    live = v.shift(2)
    half = np.maximum(live.fillna(0.), .5)
    s11 = layer(idx, ep, half)
    sm = pd.Series(0., index=idx)
    for r in ep.itertuples():
        a, b = r.i_entry + 2, min(r.i_end + 2, len(idx))
        bk = buckets.get(r.i_entry, 'normal')
        seg = {'low': np.ones(b - a), 'normal': half.iloc[a:b].to_numpy(), 'high': live.iloc[a:b].fillna(0.).to_numpy()}[bk]
        sm.iloc[a:b] = seg
    keys = st.reindex(idx).ffill() * v
    out = []
    for ws in (s11, sm):
        des = pd.concat([keys, ws.shift(-2)], axis=1).max(axis=1)
        w = act_on_change(idx, (des * 20).round() / 20)[0]
        out.append(net_returns(w, eq, pd.Series(0., index=idx)).loc[start:].dropna())
    return out[0], out[1], ep


def g4_product(P, PE, G, fits, dmas=None):
    from research.us_gauge.run_c import sharpe
    from research.us_gauge.evaluate import metrics
    rows, pes = [], []
    for m in MARKETS:
        px = P[m]
        pe = PE[PE.market == m]
        buckets = {}
        for i, r in pe.iterrows():
            blk = va.block_of([r.t]).iloc[0]
            if pd.isna(blk):
                continue
            f = fits[blk]
            p = float(f['model'].predict(G.loc[[i]])[0])
            buckets[int(r.i_t)] = str(mo.bucket([p], f['pi'])[0])
            pes.append(dict(market=m, entry=r.t, block=blk, p=p, pi=f['pi'], bucket=buckets[int(r.i_t)]))
        r11, rm, ep = market_sim(px, dmas_cached(m, px) if dmas is None else dmas[m], buckets)
        assert list(ep.i_entry) == list(pe.i_t), f'{m}: run_s.shock_episodes entries differ from the frozen list'
        z = pd.Series(0., index=r11.index)
        rows.append(dict(market=m, sh_11d=sharpe(r11, z), sh_model=sharpe(rm, z),
                         dd_11d=metrics(r11, z)['maxdd'], dd_model=metrics(rm, z)['maxdd'],
                         n_scored=len(buckets)))
    T = pd.DataFrame(rows)
    T['d_sh'] = T.sh_model - T.sh_11d
    T['d_dd'] = T.dd_model - T.dd_11d
    sharpe_ok = T.d_sh.mean() > 0 and (T.d_sh > 0).sum() >= 13
    dd_ok = (T.d_dd >= -.02).sum() >= 18 and (T.d_dd >= -.05).all()
    text = (f"mean Sharpe diff {T.d_sh.mean():+.4f}, positive in {(T.d_sh > 0).sum()} of {len(T)}; worst fall within 2 pts in "
            f"{(T.d_dd >= -.02).sum()} of {len(T)}, deepest extra {T.d_dd.min():+.1%}")
    return dict(table=T, episodes=pd.DataFrame(pes), sharpe_ok=sharpe_ok, dd_ok=dd_ok, text=text)


# ------------------------------------------------------------------ G7: US core non-inferiority (not evidence)

def g7_core(ctx, entries, fits, start=G7_START, end=DATA_END):
    from research.us_gauge import run_vol
    from research.us_gauge.data import load_inputs
    from research.us_gauge.run_s1 import read_only_external
    from research.us_gauge.revisions import detect_rolls, revisions, rev_zscore
    from research.us_gauge.run_p import episodes as f_episodes
    from research.us_gauge.run_c import floor_weights, sharpe
    from research.us_gauge.run_fast import states, to_weights
    from research.us_gauge.run_s import shock_episodes
    from research.us_gauge.run_b import breadth, deemer_ratio, thrust_days, thrust_episodes
    from research.us_gauge.core_slide_data import layer, act_on_change
    from research.us_gauge.trading import net_returns
    from research.us_gauge.evaluate import metrics
    with read_only_external():
        inputs = load_inputs().loc[:end]
    br = breadth()
    idx = inputs.index
    v = run_vol.target_weight(inputs.eq_ret)
    live = v.shift(2)
    full = pd.Series(1., index=idx)
    half = np.maximum(live.fillna(0.), .5)
    td = thrust_days(deemer_ratio(br.adv, br.dec))
    days = idx[idx >= '1993-04-01']
    dm = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0].reindex(days)
    rz = rev_zscore(revisions(inputs.fy1, inputs.fy2, detect_rolls(inputs.fy1, inputs.fy2), window=63)).reindex(days)
    s1 = states(days, (dm < 45) & (rz < 0), dm < 30, dm, None)
    w1 = to_weights(idx, s1)
    epf = f_episodes(idx, s1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    wf = floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)
    eps_s, eps_b = shock_episodes(inputs.px, w1), thrust_episodes(inputs.px, w1, td)
    wbl = layer(idx, eps_b, full)
    ws11 = layer(idx, eps_s, half)
    wsm = pd.Series(0., index=idx)
    shown = []
    for r in eps_s.itertuples():
        a, b = r.i_entry + 2, min(r.i_end + 2, len(idx))
        blk = va.block_of([r.entry]).iloc[0]
        bk = 'normal'
        if not pd.isna(blk):
            f = fits[blk]
            prior = [d for d in entries['GSPC'] if d < r.entry]
            x = pd.DataFrame([event_features('GSPC', inputs.px, idx.get_loc(r.high), idx.get_loc(r.low), r.i_entry, ctx, prior)])
            p = float(f['model'].predict(x)[0])
            bk = str(mo.bucket([p], f['pi'])[0])
            shown.append(f'{r.entry.date()} p {p:.2f} (pi {f["pi"]:.2f}) {bk}')
        seg = {'low': np.ones(b - a), 'normal': half.iloc[a:b].to_numpy(), 'high': live.iloc[a:b].fillna(0.).to_numpy()}[bk]
        wsm.iloc[a:b] = seg
    keys = s1.reindex(idx).ffill() * v
    res = {}
    for name, ws in (('core', ws11), ('model', wsm)):
        des = pd.concat([keys, wf.shift(-2), ws.shift(-2), wbl.shift(-2)], axis=1).max(axis=1)
        w = act_on_change(idx, (des * 20).round() / 20)[0]
        r_ = net_returns(w, inputs.eq_ret, inputs.cash_ret).loc[start:]
        cash = inputs.cash_ret.reindex(r_.index)
        res[name] = (sharpe(r_, cash), metrics(r_, cash)['maxdd'])
    ok = res['model'][0] >= res['core'][0] - .01 and res['model'][1] >= res['core'][1] - .02
    text = (f"Sharpe {res['model'][0]:.3f} vs core {res['core'][0]:.3f}; worst fall {res['model'][1]:.1%} vs {res['core'][1]:.1%}. "
            f"Core S episodes scored: {'; '.join(shown) or 'none'}")
    return dict(ok=ok, text=text, res=res)


# ------------------------------------------------------------------ report-only: extended model (revised data)

EXT_10Y = {'N225': 'JP', 'GSPTSE': 'CA', 'KS11': 'KR', 'AXJO': 'AU', 'SSMI': 'CH', 'GDAXI': 'DE', 'FCHI': 'FR',
           'AEX': 'NL', 'IBEX': 'ES', 'OMXS30': 'SE', 'ATX': 'AT', 'NZ50': 'NZ', 'MXX': 'MX'}
EXT_CLI = {'GSPC': 'USA', 'N225': 'JPN', 'GSPTSE': 'CAN', 'KS11': 'KOR', 'AXJO': 'AUS', 'SSMI': 'CHE', 'GDAXI': 'DEU',
           'FCHI': 'FRA', 'AEX': 'NLD', 'IBEX': 'ESP', 'OMXS30': 'SWE', 'ATX': 'AUT', 'NZ50': 'NZL', 'MXX': 'MEX',
           'BSESN': 'IND', 'BVSP': 'BRA'}


def _monthly(name):
    from research.us_gauge.run_s1 import read_only_external
    try:
        with read_only_external():
            s = _store_load(name).sort_index().astype(float).dropna()
    except Exception:
        return None
    s.index = pd.DatetimeIndex(s.index).to_period('M')
    return s[~s.index.duplicated()]


def _mval(s, d, lag):
    if s is None:
        return np.nan
    m = pd.Timestamp(d).to_period('M') - lag
    return float(s.loc[m]) if m in s.index else np.nan


def extended_features(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cache = {}
    get = lambda n: cache.setdefault(n, _monthly(n))
    for r in S.itertuples():
        c, i3 = EXT_10Y.get(r.market), EXT_CLI.get(r.market)
        l10 = get(f'FRED_IRLTLT01{c}M156N') if c else None
        s3 = get(f'FRED_IR3TIB01{c}M156N') if c else None
        cli = get(f'FRED_{i3}LOLITOAASTSAM') if i3 else None
        rows.append(dict(loc10_chg=_mval(l10, r.t, 1) - _mval(l10, r.h, 1),
                         loc_curve=_mval(l10, r.t, 1) - _mval(s3, r.t, 1),
                         cli_6m=_mval(cli, r.t, 3) - _mval(cli, r.t, 9)))
    return pd.DataFrame(rows, index=S.index)


# ------------------------------------------------------------------ the run

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--one-run', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    hashes = guard(a.one_run, a.dry_run)
    E, F, PE, G = load_frozen()
    P, B = load_panel()
    macro = load_macro()
    ctx, entries = verify_frozen(P, macro, E, F, PE, G)
    st_mask = E.study.to_numpy()
    S = E[st_mask].copy()
    X = F[st_mask].set_index(S.index)
    blk = va.block_of(S.t)
    span = blk[blk.notna()].groupby(S.cluster[blk.notna()].to_numpy()).nunique()
    if (span > 1).any():
        raise SystemExit(f'REFUSED: clusters span two blocks: {list(span[span > 1].index)}')
    if a.dry_run:
        print(f'DRY RUN OK: hashes match the Freeze section; events, product episodes and features reproduce from the store. '
              f'Study events {len(S)}, clusters {S.cluster.nunique()}, not open {int((~S.open).sum())}. '
              f'No label, return or outcome was computed.')
        return 0

    # ---------------- labels (real data, one run) ----------------
    from research.us_gauge import run_vol
    L = lb.label_events(P, S)
    assert not (S.open & ~L.open).any()            # frozen open events stay open
    y, R = L.y, L.R
    v = pd.Series([float(run_vol.target_weight(P[r.market].pct_change(fill_method=None)).iloc[r.i_t]) for r in S.itertuples()],
                  index=S.index)

    fits = {name: fit_blocks(S, X, y, fac) for name, fac in mo.FACTORIES.items()}
    wf_check = va.walk_forward(S, X, y, mo.MLogit)
    assert np.allclose(wf_check.p.sort_index(), pd.concat([f['p'] for f in fits['M-L'].values()]).sort_index())
    D = {name: pooled(S, y, R, v, f) for name, f in fits.items()}
    too, n_cl, n_failed = va.too_few(D['M-L'])
    single = [n for n in fits if n.startswith('only ')]
    stats = {n: model_stats(D[n]) for n in ('M-L', 'M-G')}
    single_K = {n: va.K_stat(D[n]) for n in single}
    bench = {n: dict(auc=va.wauc(D[n].y, D[n].p, va.cluster_weights(D[n].cluster)),
                     brier=va.brier(D[n].y, D[n].p, va.cluster_weights(D[n].cluster)),
                     logloss=va.logloss(D[n].y, D[n].p, va.cluster_weights(D[n].cluster)),
                     K=va.K_stat(D[n])) for n in D}
    g4 = {n: g4_product(P, PE, G, fits[n]) for n in ('M-L', 'M-G')}
    g7 = {n: g7_core(ctx, entries, fits[n]) for n in ('M-L', 'M-G')}
    gates = {n: gates_from(stats[n], single_K, g4[n], g7[n]) for n in ('M-L', 'M-G')}
    g4_dd_only = {n: g4[n]['sharpe_ok'] and not g4[n]['dd_ok'] for n in ('M-L', 'M-G')}
    verdict = consequence(gates['M-L'], stats['M-L'], too, g4_dd_only['M-L'])
    both = D['M-L'][['cluster', 'y', 'p']].join(D['M-G'][['p']].rename(columns={'p': 'p_g'}))
    p_boot = va.bootstrap_logloss(both)
    mg_replaces = (all(v_[0] for v_ in gates['M-L'].values()) and all(v_[0] for v_ in gates['M-G'].values()) and p_boot < .10)

    # ---------------- report-only lines ----------------
    lab = y.notna()
    loyo = va.loyo(S, X, y, mo.MLogit)
    w_lab = va.cluster_weights(S.cluster[lab])
    loyo_auc = va.wauc(y[lab], loyo[lab], w_lab)
    signs = va.coef_sign_stability(S, X, y, mo.MLogit, n=1000)
    final = mo.MLogit().fit(X[lab], y[lab].astype(int), w_lab)
    pi_all = mo.train_pi(y[lab], w_lab)
    early = E[E.us_early & (E.t <= '1954-12-31')]
    Le = lb.label_events(P, early)
    pe_ = final.predict(F.loc[early.index])
    ext = extended_features(S)
    Xe = X.join(ext)
    ext_fits = fit_blocks(S, Xe, y, lambda: mo.MBoost(mo.MG_FEATURES + list(ext.columns)))
    De = pooled(S, y, R, v, ext_fits)
    ext_line = (f"AUC {va.wauc(De.y, De.p, va.cluster_weights(De.cluster)):.3f}, K {va.K_stat(De):+.5f} "
                f"(coverage: local 10y {ext.loc10_chg.notna().mean():.0%}, curve {ext.loc_curve.notna().mean():.0%}, CLI {ext.cli_6m.notna().mean():.0%})")

    # ---------------- report ----------------
    Lr = ['# ML sizing of the shock re-entry — the one run (spec 2026-11-18, frozen in section 12)', '',
          f'**Verdict (M-L, the gate model): {verdict}**', '',
          f'M-G replaces M-L: {"YES" if mg_replaces else "no"} (cluster bootstrap P(M-G log-loss >= M-L) = {p_boot:.3f}).', '',
          f'Frozen features.csv SHA-256 `{hashes["features.csv"]}`; data cut {DATA_END.date()}. '
          f'Out-of-sample with a label: {len(D["M-L"])} events, {n_cl} clusters, failed clusters (cluster-weighted) {n_failed:.1f}'
          f'{" — TOO FEW" if too else ""}.', '',
          '| Gate | M-L | Evidence (M-L) | M-G (report unless M-L passes all) |', '|---|---|---|---|']
    for k in gates['M-L']:
        Lr.append(f"| {k} | {'PASS' if gates['M-L'][k][0] else 'FAIL'} | {gates['M-L'][k][1]} | "
                  f"{'pass' if gates['M-G'][k][0] else 'fail'}: {gates['M-G'][k][1]} |")
    Lr += ['', '## Benchmarks (pooled out-of-sample 2000–2026, cluster weights)', '',
           '| Model | AUC | Brier | Log-loss | K |', '|---|---:|---:|---:|---:|']
    Lr += [f"| {n} | {b['auc']:.3f} | {b['brier']:.4f} | {b['logloss']:.4f} | {b['K']:+.5f} |" for n, b in bench.items()]
    best1 = max(single, key=lambda n: bench[n]['K'])
    if bench['M-L']['K'] <= bench[best1]['K']:
        Lr.append(f'\nM-L does not beat its best single feature ({best1}) on K: not worth its complexity.')
    Lr += ['', '## By block (M-L)', '', '| Block | Events | Clusters | pi (train) | AUC | K | Buckets |', '|---|---:|---:|---:|---:|---:|---|']
    for b in va.BLOCKS:
        d = D['M-L'][D['M-L'].block == b.name]
        Lr.append(f"| {b.name} | {len(d)} | {d.cluster.nunique()} | {d.pi.iloc[0]:.3f} | "
                  f"{va.wauc(d.y, d.p, va.cluster_weights(d.cluster)) if d.y.nunique() > 1 else float('nan'):.3f} | "
                  f"{va.K_stat(d):+.5f} | {d.bucket.value_counts().to_dict()} |")
    Lr += ['', '## G4 by market (M-L, 2000 → data end)', '', '| Market | Episodes scored | Sharpe 11d | Sharpe model | Worst fall 11d | model |', '|---|---:|---:|---:|---:|---:|']
    Lr += [f'| {r.market} | {r.n_scored} | {r.sh_11d:.3f} | {r.sh_model:.3f} | {r.dd_11d:.1%} | {r.dd_model:.1%} |' for r in g4['M-L']['table'].itertuples()]
    us = D['M-L'][D['M-L'].market == 'GSPC']
    Lr += ['', '## US (never evidence)', '', 'US walk-forward events (in-sample for design): ' +
           '; '.join(f'{r.t.date()} p {r.p:.2f} {r.bucket} (stopped {int(r.y)})' for r in us.itertuples()),
           '', f"G7 detail: {g7['M-L']['text']}", '',
           '## Report-only lines', '',
           f'- Leave-one-calendar-year-out (M-L, trains on later data too): AUC {loyo_auc:.3f}. US 1987-10-21: '
           + ', '.join(f'p {loyo[i]:.2f}' for i in S.index[(S.market == "GSPC") & (S.t == "1987-10-21")]) + '.',
           '- M-L on all labelled events (expected sign; share of 1,000 cluster bootstraps with the same sign): '
           + '; '.join(f'{f} {signs.coef[f]:+.3f} (expected {mo.EXPECTED_SIGNS[f]}; {signs.same_sign_share[f]:.0%})' for f in signs.index) + '.',
           f'- US 1929–1954 events, M-L fitted on all labelled events (pi {pi_all:.2f}): '
           + '; '.join(f'{r.t.date()} p {p:.2f} {mo.bucket([p], pi_all)[0]} (stopped {"open" if np.isnan(Le.y[i]) else int(Le.y[i])})'
                       for (i, r), p in zip(early.iterrows(), pe_)) + '.',
           f'- Extended model (report only, REVISED DATA, never a gate): {ext_line}.', '',
           '## Data notes (daily moves above 25%, not removed)', ''] + big_moves_lines(B)
    Lr += ['', 'Code: `research/us_gauge/ml/` (run_m.py). One run; no second run with other features, labels, thresholds, blocks, buckets or markets.', '']
    D['M-L'].to_csv(OUT / 'oos_predictions_ml.csv', date_format='%Y-%m-%d')
    D['M-G'].to_csv(OUT / 'oos_predictions_mg.csv', date_format='%Y-%m-%d')
    g4['M-L']['table'].to_csv(OUT / 'g4_markets_ml.csv', index=False)
    g4['M-L']['episodes'].to_csv(OUT / 'g4_episodes_ml.csv', index=False, date_format='%Y-%m-%d')
    RESULTS.write_text('\n'.join(Lr))
    print('\n'.join(Lr))
    return 0


if __name__ == '__main__':
    sys.exit(main())
