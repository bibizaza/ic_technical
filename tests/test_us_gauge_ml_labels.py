"""Labels and the guarded run script, on SYNTHETIC data only (no real post-trigger price is read here)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research.us_gauge.ml import labels as lb
from research.us_gauge.ml import run_m
from research.us_gauge.ml import validate as va
from research.us_gauge.ml import models as mo
from research.us_gauge.ml.events import cluster_ids
from research.us_gauge.ml.features import FEATURES

ROOT = Path(__file__).resolve().parents[1]


def test_label_v_shape_not_stopped():
    p = np.r_[np.linspace(120, 96, 31), np.linspace(96, 106, 6), np.linspace(106, 130, 200)]
    i_t = 35
    d = lb.label_one(p, i_t, 96.)
    assert d['y'] == 0 and d['i_x'] == i_t + 126 and not d['open']
    assert d['R'] == pytest.approx(p[i_t + 127] / p[i_t + 1] - 1)


def test_label_retest_stopped_on_first_close_below_L():
    p = np.r_[np.full(10, 100.), [90., 99., 100., 95., 89.9, 85., 80.], np.full(200, 95.)]
    i_t = 12                                            # entry close, L = 90
    d = lb.label_one(p, i_t, 90.)
    assert d['y'] == 1 and d['i_x'] == 14 and d['R'] == pytest.approx(p[15] / p[13] - 1)


def test_close_equal_to_L_is_not_a_stop():
    p = np.r_[np.full(5, 100.), [90.], np.full(5, 99.), [90.], np.full(150, 99.)]
    assert lb.label_one(p, 7, 90.)['y'] == 0


def test_open_when_window_incomplete_or_exit_day_missing():
    p = np.linspace(100, 110, 100)
    assert lb.label_one(p, 10, 90.)['open']
    p2 = np.full(10 + 127, 100.)                         # window complete, but no close at x + 1
    assert lb.label_one(p2, 10, 90.)['open']
    p3 = np.full(10 + 128, 100.)
    assert not lb.label_one(p3, 10, 90.)['open']


def test_label_events_frame():
    idx = pd.bdate_range('2000-01-03', periods=400)
    px = pd.Series(np.r_[np.full(200, 100.), np.full(200, 80.)], index=idx)
    E = pd.DataFrame({'market': ['A', 'A', 'A'], 'i_t': [10, 250, 300], 'L': [90., 70., 70.]})
    L = lb.label_events({'A': px}, E)
    assert L.y.iloc[0] == 0 and L.y.iloc[1] == 0 and L.x.iloc[0] == idx[136]
    assert L.open.iloc[2] and np.isnan(L.y.iloc[2])
    E2 = pd.DataFrame({'market': ['A'], 'i_t': [150], 'L': [90.]})
    assert lb.label_events({'A': px}, E2).y.iloc[0] == 1


# ---------------- guard ----------------

def _frozen_dir(tmp_path, text_hash=None):
    out = tmp_path / 'out'
    out.mkdir()
    for f in run_m.FROZEN:
        (out / f).write_text(f'synthetic {f}\n')
    h = {f: run_m.sha256(out / f) for f in run_m.FROZEN}
    spec = tmp_path / 'spec.md'
    spec.write_text(f"- **SHA-256 of features.csv: `{text_hash or h['features.csv']}`**. Also frozen: events.csv "
                    f"`{h['events.csv']}`, product_episodes.csv `{h['product_episodes.csv']}`, product_features.csv `{h['product_features.csv']}`.")
    return out, spec


def test_guard_needs_one_run_flag(tmp_path):
    out, spec = _frozen_dir(tmp_path)
    with pytest.raises(SystemExit, match='--one-run'):
        run_m.guard(False, False, spec, tmp_path / 'RESULTS_ml.md', out)
    assert run_m.guard(True, False, spec, tmp_path / 'RESULTS_ml.md', out)['features.csv']


def test_guard_refuses_if_results_exist(tmp_path):
    out, spec = _frozen_dir(tmp_path)
    (tmp_path / 'RESULTS_ml.md').write_text('done')
    with pytest.raises(SystemExit, match='one run only'):
        run_m.guard(True, False, spec, tmp_path / 'RESULTS_ml.md', out)


def test_guard_refuses_on_hash_mismatch(tmp_path):
    out, spec = _frozen_dir(tmp_path, text_hash='0' * 64)
    with pytest.raises(SystemExit, match='SHA-256 of features.csv'):
        run_m.guard(True, False, spec, tmp_path / 'RESULTS_ml.md', out)
    out2, spec2 = _frozen_dir(tmp_path / 'b' if (tmp_path / 'b').mkdir() is None else None)
    (out2 / 'features.csv').write_text('changed\n')
    with pytest.raises(SystemExit, match='REFUSED'):
        run_m.guard(True, False, spec2, tmp_path / 'RESULTS_ml.md', out2)


def test_frozen_files_match_the_spec():
    """The committed frozen files hash to the values in the spec (reads hashes only; no label)."""
    want = run_m.frozen_hashes(run_m.SPEC.read_text())
    for f, h in want.items():
        assert run_m.sha256(run_m.OUT / f) == h


def test_dry_run_flag_parsing_refuses_both(tmp_path):
    out, spec = _frozen_dir(tmp_path)
    with pytest.raises(SystemExit):
        run_m.guard(True, True, spec, tmp_path / 'RESULTS_ml.md', out)


# ---------------- consequences ----------------

def _g(**fail):
    return {k: (k not in fail, '') for k in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8']}


ST = dict(auc=.7, p_auc=.001)


def test_consequences():
    assert run_m.consequence(_g(), ST, False, False).startswith('ADOPT:')
    assert run_m.consequence(_g(), ST, True, False).startswith('TOO FEW')
    assert run_m.consequence(_g(G1=1), dict(auc=.55, p_auc=.3), False, False).startswith('STOP')
    assert run_m.consequence(_g(G5=1), ST, False, False).startswith('DEFENSIVE')
    assert run_m.consequence(_g(G4=1), ST, False, True).startswith('DEFENSIVE')      # worst-fall leg only
    assert run_m.consequence(_g(G4=1), ST, False, False).startswith('NO CHANGE')     # Sharpe leg
    assert run_m.consequence(_g(G8=1), ST, False, False).startswith('ADOPT NOTHING')
    assert run_m.consequence(_g(G1=1), dict(auc=.65, p_auc=.05), False, False).startswith('REPORT ONLY')
    assert run_m.consequence(_g(G3=1), ST, False, False).startswith('REPORT ONLY')
    assert run_m.consequence(_g(G6=1), ST, False, False).startswith('NO CHANGE')


# ---------------- product simulation on a synthetic path ----------------

def _synthetic_market(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    r = rng.standard_t(4, n) * .012 + .0003
    for k in range(500, n, 600):
        r[k:k + 20] -= .012
    idx = pd.bdate_range('1995-01-02', periods=n)
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    ma = px.rolling(200, min_periods=1).mean()
    dmas = pd.Series(np.where(px > ma, 70, 20), index=idx)     # a stand-in for DMAS: invested above the 200-day mean
    return px, dmas


def test_market_sim_normal_bucket_equals_11d_and_low_holds_more():
    px, dmas = _synthetic_market()
    r11, rm, ep = run_m.market_sim(px, dmas, {}, start='1996-01-01')
    assert len(ep) >= 1
    pd.testing.assert_series_equal(r11, rm)
    low = {int(i): 'low' for i in ep.i_entry}
    high = {int(i): 'high' for i in ep.i_entry}
    _, rl, _ = run_m.market_sim(px, dmas, low, start='1996-01-01')
    _, rh, _ = run_m.market_sim(px, dmas, high, start='1996-01-01')
    assert not rl.equals(r11) or not rh.equals(r11)


# ---------------- pooled statistics end to end on synthetic events ----------------

def _synthetic_study(seed=1, planted=True):
    rng = np.random.default_rng(seed)
    n = 320
    t = pd.to_datetime('1980-01-01') + pd.to_timedelta(np.sort(rng.integers(0, 46 * 365, n)), 'D')
    S = pd.DataFrame({'t': t, 'market': rng.choice(['GSPC', 'AEX', 'N225', 'MXX', 'GDAXI'], n)})
    near = np.zeros(n, bool)                                   # keep clusters inside one block (as on the real list)
    for b in va.BLOCKS[1:]:
        near |= (abs((S.t - pd.Timestamp(b.start)).dt.days) < 60).to_numpy()
    S = S[~near].reset_index(drop=True)
    n = len(S)
    S['region'] = S.market.map({'GSPC': 'north_america', 'AEX': 'europe', 'GDAXI': 'europe', 'N225': 'asia_pac_dev', 'MXX': 'emerging'})
    S['seen'] = S.market.isin(['GSPC', 'N225'])
    S['cluster'] = cluster_ids(S.t, 45)
    S['t_embargo'] = S.t + pd.Timedelta(days=275)
    S['event_id'] = [f'e{i}' for i in range(n)]
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    z = (2.0 * X.runup_2y - 1.5 * X.days_le - 1.2) if planted else np.full(n, -1.1)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-z))).astype(float))
    R = pd.Series(np.where(y == 1, -.10, .08) + rng.normal(0, .03, n))
    v = pd.Series(rng.uniform(.15, .6, n))
    return S, X, y, R, v


def test_pooled_stats_find_planted_signal_and_not_noise():
    S, X, y, R, v = _synthetic_study(1, True)
    d = run_m.pooled(S, y, R, v, run_m.fit_blocks(S, X, y, mo.MLogit))
    st = run_m.model_stats(d, n_perm=499)
    assert st['auc'] > .7 and st['p_auc'] < .01 and st['K'] > 0 and st['p_K'] < .01
    S, X, y, R, v = _synthetic_study(2, False)
    d = run_m.pooled(S, y, R, v, run_m.fit_blocks(S, X, y, mo.MLogit))
    st = run_m.model_stats(d, n_perm=499)
    assert st['p_auc'] > .01 or st['auc'] < .6


def test_fit_blocks_equals_validate_walk_forward():
    S, X, y, R, v = _synthetic_study(3, True)
    f = run_m.fit_blocks(S, X, y, mo.MLogit)
    wf = va.walk_forward(S, X, y, mo.MLogit)
    a = pd.concat([b['p'] for b in f.values()]).sort_index()
    np.testing.assert_allclose(a.to_numpy(), wf.p.sort_index().to_numpy())


def test_run_m_without_flag_refuses():
    with pytest.raises(SystemExit):
        run_m.main([])


# ---------------- the whole run script, end to end, on a SYNTHETIC world ----------------

def _world(tmp_path, monkeypatch):
    from research.us_gauge.ml import events as ev, features as fe
    from research.us_gauge.ml.panel import MACRO
    rng = np.random.default_rng(11)
    mk = ['GSPC', 'AEX', 'N225', 'MXX', 'GDAXI']
    P = {}
    for i, m in enumerate(mk):
        start = '1950-01-02' if m == 'GSPC' else '1968-01-01'
        idx = pd.bdate_range(start, '2026-11-11')
        r = rng.standard_t(4, len(idx)) * .011 + .0003
        for k in range(300 + 40 * i, len(idx), 520 + 30 * i):
            r[k:k + 18] -= .013
        P[m] = pd.Series(100 * np.exp(np.cumsum(r)), index=idx, name=m)
    days = pd.bdate_range('1950-01-01', '2026-11-11')
    macro = {}
    for n in MACRO:
        s = pd.Series(np.exp(np.cumsum(rng.normal(0, .004, len(days)))) * 2, index=days)
        if n in ('FRED_BAA', 'FRED_AAA'):
            s = s.resample('MS').mean() + (1. if n == 'FRED_BAA' else 0.)
        macro[n] = s
    dm = lambda m, px: pd.Series(np.where(px > px.rolling(200, min_periods=1).mean(), 70, 20), index=px.index)
    monkeypatch.setattr(ev, 'MARKETS', mk)
    monkeypatch.setattr(ev, 'dmas_cached', dm)
    E, PE = ev.build(P)
    ctx = fe.Context(P, macro)
    entries = {m: list(E.loc[E.market == m, 't']) for m in mk}
    F = fe.features_for(E, P, ctx, entries)
    F.insert(0, 'event_id', E.event_id)
    G = fe.features_for(PE, P, ctx, entries)
    G.insert(0, 'episode_id', PE.episode_id)
    out = tmp_path / 'out'
    out.mkdir()
    monkeypatch.setattr(run_m, 'MARKETS', mk)
    monkeypatch.setattr(run_m, 'dmas_cached', dm)
    monkeypatch.setattr(run_m, 'OUT', out)
    monkeypatch.setattr(run_m, 'RESULTS', tmp_path / 'RESULTS_ml.md')
    monkeypatch.setattr(run_m, 'guard', lambda one, dry: {'features.csv': 'synthetic'})
    monkeypatch.setattr(run_m, 'load_frozen', lambda: (E, F, PE, G))
    monkeypatch.setattr(run_m, 'load_panel', lambda: (P, {m: pd.Series(dtype=float) for m in mk}))
    monkeypatch.setattr(run_m, 'load_macro', lambda: macro)
    monkeypatch.setattr(run_m, 'verify_frozen', lambda *a: (ctx, entries))
    monkeypatch.setattr(run_m, 'g7_core', lambda ctx, entries, fits: dict(ok=True, text='synthetic stub', res={}))
    monkeypatch.setattr(run_m, 'extended_features', lambda S: pd.DataFrame(
        {'loc10_chg': rng.normal(size=len(S)), 'loc_curve': np.nan, 'cli_6m': rng.normal(size=len(S))}, index=S.index))
    return E


def test_run_script_end_to_end_on_a_synthetic_world(tmp_path, monkeypatch):
    E = _world(tmp_path, monkeypatch)
    S = E[E.study]
    assert len(S) > 80 and all((S.t >= b.start).any() for b in va.BLOCKS)
    assert run_m.main(['--dry-run']) == 0
    assert not (tmp_path / 'RESULTS_ml.md').exists()
    assert run_m.main(['--one-run']) == 0
    txt = (tmp_path / 'RESULTS_ml.md').read_text()
    assert '**Verdict (M-L, the gate model):' in txt and '| G8 |' in txt and 'Extended model' in txt
