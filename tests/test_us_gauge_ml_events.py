from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research.us_gauge.ml import events as ev
from research.us_gauge.run_s import shock_episodes

OUT = Path(__file__).resolve().parents[1] / 'research/us_gauge/ml/out'


def _path(*legs, start=100.):
    """Concatenate linear legs (n_days, end_level); first leg starts at `start`."""
    x, lvl = [], start
    for n, end in legs:
        x.extend(np.linspace(lvl, end, n + 1)[1:])
        lvl = end
    return pd.Series(x, index=pd.bdate_range('1990-01-01', periods=len(x)))


def _reference(px):
    """Non-causal reference (the scratch count of 2026-11-18): scan, jump past the event close."""
    p = px.to_numpy(); n = len(p); hi = ev.high_index(p)
    out, k, last_h, last_L = [], 252, -1, np.inf
    while k < n:
        h = hi[k]; l = h + int(p[h:k + 1].argmin()); L = p[l]
        if h != last_h:
            last_h, last_L = h, np.inf
        if L <= .85 * p[h] and l - h <= 63 and p[k] >= 1.10 * L and L < last_L:
            end = min(k + 126, n - 1)
            for j in range(k + 1, min(k + 126, n - 1) + 1):
                if p[j] < L:
                    end = j
                    break
            out.append(k); last_L = L; k = end + 1
            continue
        k += 1
    return out


def test_v_shape_triggers_once():
    px = _path((300, 120), (30, 96), (40, 115), (200, 125))
    e = ev.scan_events(px)
    assert len(e) == 1
    r = e.iloc[0]
    assert r.P_t >= 1.10 * r.L and px.iloc[r.i_t - 1] < 1.10 * r.L
    assert r.L / r.H - 1 == pytest.approx(-.2)


def test_retest_then_new_event_with_lower_low():
    # V up to +10%, then a retest below L (event closes), then a new lower low and a new +10% bounce
    px = _path((300, 120), (30, 96), (10, 106), (10, 90), (20, 101), (150, 100))
    e = ev.scan_events(px)
    assert len(e) == 2
    assert e.L.iloc[1] < e.L.iloc[0] and e.i_h.iloc[0] == e.i_h.iloc[1]


def test_slow_fall_does_not_trigger():
    px = _path((300, 120), (120, 96), (40, 110))
    assert ev.scan_events(px).empty


def test_retrigger_limit_same_high_needs_lower_low():
    # bounce +10%, event closes by time (no retest), second bounce from the same low: no new event
    px = _path((300, 120), (30, 96), (10, 106), (140, 106), (20, 97), (10, 107), (40, 107))
    e = ev.scan_events(px)
    assert len(e) == 1


def test_no_new_event_while_open():
    # a fresh 15% fall from a new high inside the 126-day window, staying above L: blocked
    px = _path((300, 100), (20, 80), (40, 120), (10, 101), (10, 112), (200, 112))
    e = ev.scan_events(px)
    assert len(e) == 1


def test_matches_reference_on_random_paths():
    rng = np.random.default_rng(3)
    for _ in range(40):
        r = rng.standard_t(3, 3000) * .015
        px = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range('1990-01-01', periods=3000))
        assert list(ev.scan_events(px).i_t) == _reference(px)


def test_product_entries_match_run_s_on_synthetic_paths():
    rng = np.random.default_rng(5)
    for _ in range(30):
        r = rng.standard_t(3, 2500) * .015
        px = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range('1990-01-01', periods=2500))
        bw = pd.Series((rng.random(2500) < .4).astype(float), index=px.index).rolling(40).max().fillna(0.)
        bw.iloc[:300] = np.nan
        ref = shock_episodes(px, bw)
        assert list(ev.product_entries(px, bw).i_t) == list(ref.i_entry)


def test_cluster_chaining():
    d = pd.to_datetime(['2020-01-01', '2020-02-10', '2020-03-20', '2020-06-01', '2019-12-31'])
    ids = ev.cluster_ids(d, 45)
    assert ids[0] == ids[1] == ids[2] == ids[4] and ids[3] != ids[0]
    assert len(set(ev.cluster_ids(d, 21))) == 4


def test_events_have_no_outcome_columns():
    E = pd.read_csv(OUT / 'events.csv')
    forbidden = {'end', 'reason', 'i_end', 'y', 'R', 'label', 'stopped', 'ret'}
    assert not forbidden & set(E.columns)
    assert not forbidden & set(pd.read_csv(OUT / 'product_episodes.csv').columns)


def test_counts_match_spec_section_2_3():
    E = pd.read_csv(OUT / 'events.csv', parse_dates=['t'])
    pre = E[E.from_1965]
    assert len(pre) == 261 and pre.cluster45_pre.nunique() == 78 and pre.cluster21_pre.nunique() == 115
    assert (pre.t < '2000-01-01').sum() == 102 and (pre.t >= '2000-01-01').sum() == 159
    assert (pre.market == 'GSPC').sum() == 9
    PE = pd.read_csv(OUT / 'product_episodes.csv', parse_dates=['t'])
    assert (PE.t >= '1965-01-01').sum() == 180 and (PE.t >= '2000-01-01').sum() == 115


@pytest.mark.skipif(not Path('/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/data/market_store').exists(),
                    reason='market store not available')
def test_event_scan_on_store_reproduces_events_csv():
    from research.us_gauge.ml.panel import load_panel
    P, _ = load_panel()
    E = pd.read_csv(OUT / 'events.csv', parse_dates=['t'])
    for m, px in P.items():
        assert list(ev.scan_events(px).t) == list(E[E.market == m].sort_values('t').t)
