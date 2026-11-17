"""Breadth thrust B: synthetic data and the thrust-date check only. No returns of the real rule are computed here."""
import numpy as np
import pandas as pd
import pytest
from research.us_gauge import run_b
from research.us_gauge.run_b import deemer_ratio, thrust_days, zweig_days, thrust_episodes, placebo_spans, gates

IDX = pd.bdate_range('2000-01-03', periods=400)


def _breadth(adv, dec):
    return pd.Series(adv, index=IDX[:len(adv)], dtype=float), pd.Series(dec, index=IDX[:len(dec)], dtype=float)


def test_ratio_is_ten_day_totals():
    adv, dec = _breadth([2000] * 10 + [1000] * 10, [1000] * 20)
    r = deemer_ratio(adv, dec)
    assert r.iloc[:9].isna().all()
    assert r.iloc[9] == pytest.approx(2.0)
    assert r.iloc[10] == pytest.approx(19000 / 10000)


def test_deemer_worked_example_from_his_page():
    # walterdeemer.com/bam.htm: ten-day totals 21000 and 10400 give 2.02, a thrust.
    adv = [2800, 2500, 2050, 2050, 1400, 1400, 2500, 2300, 2100, 1900]
    dec = [500, 700, 1050, 1050, 1700, 1700, 700, 800, 1000, 1200]
    a, d = _breadth([1000] * 10 + adv, [1000] * 10 + dec)
    r = deemer_ratio(a, d)
    assert r.iloc[-1] == pytest.approx(21000 / 10400)
    assert r.iloc[-1] > run_b.THRESHOLD and len(thrust_days(r)) == 1      # one upward crossing in the run-up


def test_thrust_is_a_crossing_not_a_level():
    r = pd.Series([1.5, 1.97, 1.98, 2.2, 1.9, 1.98], index=IDX[:6])
    assert list(thrust_days(r)) == [IDX[2], IDX[5]]          # 1.97 itself is not "greater than 1.97"


def test_zweig_needs_the_low_reading_within_ten_days():
    lo, hi = [300] * 15, [900] * 12
    a, d = _breadth([500] * 30 + lo + hi, [500] * 30 + [700] * 15 + [100] * 12)
    assert len(zweig_days(a, d)) == 1
    a2, d2 = _breadth([500] * 30 + lo + [500] * 15 + hi, [500] * 30 + [700] * 15 + [500] * 15 + [100] * 12)
    assert len(zweig_days(a2, d2)) == 0


def _px(vals):
    return pd.Series(vals, index=IDX[:len(vals)], dtype=float)


def test_episode_only_when_base_in_cash():
    px = _px(np.r_[np.linspace(100, 80, 100), np.linspace(80, 100, 200)])
    t = [px.index[110]]
    assert thrust_episodes(px, pd.Series(1., index=px.index), t).empty
    ep = thrust_episodes(px, pd.Series(0., index=px.index), t)
    assert len(ep) == 1 and ep.iloc[0].reason == 'time'
    assert ep.iloc[0].i_end - ep.iloc[0].i_entry == run_b.MAX_DAYS
    assert ep.iloc[0].L == pytest.approx(px.iloc[110 - run_b.L_WINDOW + 1:111].min())


def test_stop_below_low_and_base_takeover():
    px = _px(np.r_[np.linspace(100, 80, 100), np.linspace(80, 90, 20), np.linspace(90, 70, 40), np.full(100, 70.)])
    ep = thrust_episodes(px, pd.Series(0., index=px.index), [px.index[105]])
    assert ep.iloc[0].reason == 'stop' and px.iloc[ep.iloc[0].i_end] < ep.iloc[0].L
    assert px.iloc[ep.iloc[0].i_end - 1] >= ep.iloc[0].L
    bw = pd.Series(0., index=px.index)
    bw.iloc[110:] = 1.
    ep = thrust_episodes(px, bw, [px.index[105]])
    assert ep.iloc[0].reason == 'base invested' and ep.iloc[0].i_end == 110


def test_thrust_during_an_episode_is_ignored_and_a_later_one_counts():
    px = _px(np.r_[np.linspace(100, 80, 100), np.linspace(80, 120, 300)])
    bw = pd.Series(0., index=px.index)
    t = [px.index[110], px.index[150], px.index[110 + run_b.MAX_DAYS + 5]]
    ep = thrust_episodes(px, bw, t)
    assert list(ep.i_entry) == [110, 110 + run_b.MAX_DAYS + 5]


def test_too_little_history_or_unknown_base_gives_no_episode():
    px = _px(np.linspace(100, 120, 200))
    assert thrust_episodes(px, pd.Series(0., index=px.index), [px.index[10]]).empty
    assert thrust_episodes(px, pd.Series(np.nan, index=px.index), [px.index[100]]).empty


def test_placebo_spans_respect_the_rule():
    rng = np.random.default_rng(0)
    px = _px(100 + np.cumsum(rng.normal(0, 1, 400)))
    bw = pd.Series(0., index=px.index)
    bw.iloc[300:] = 1.
    for e, end in placebo_spans(np.random.default_rng(1), 50, px, bw):
        assert bw.iloc[e] == 0 and e >= run_b.L_WINDOW - 1 and e < end <= e + run_b.MAX_DAYS
        assert end <= 300


def test_gates_run_end_to_end_on_synthetic_data():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range('1990-01-01', periods=1500)
    ret = rng.normal(.0003, .01, len(idx))
    px = pd.Series(100 * np.cumprod(1 + ret), index=idx)
    inp = pd.DataFrame({'px': px, 'eq_ret': px.pct_change(), 'cash_ret': .0001})
    dmas = pd.Series(np.where(px > px.rolling(100, min_periods=1).mean(), 60, 35), index=idx)
    thrusts = idx[rng.choice(np.arange(100, 1400), size=25, replace=False)].sort_values()
    out = gates(inp, dmas, thrusts, start='1990-03-01', h_split='1992-12-31', bears=[], drop=[], n_placebo=20, n_boot=50)
    names = [c[0] for c in out['checks']]
    assert sum(n.startswith('G1') for n in names) == 4 and any(n.startswith('G5') for n in names)
    assert all(isinstance(c[2], (bool, np.bool_)) for c in out['checks'])
    assert (out['rx'].index == out['rb'].index).all()


# Deemer's own table, walterdeemer.com/bam.htm (signals inside the free data window 1965-03 .. 2020-02).
DEEMER = ['1967-01-18', '1970-12-04', '1971-12-08', '1975-01-08', '1976-01-06', '1982-08-23', '1982-10-13', '1985-01-23',
          '1987-01-15', '1991-02-05', '1992-01-06', '2009-03-23', '2009-07-23', '2009-09-16', '2016-07-12', '2019-01-09']


def test_free_breadth_series_reproduces_deemers_table():
    try:
        br = run_b.breadth()
    except Exception:
        pytest.skip('market_store breadth series not available')
    br = br.loc[:'2020-02-10']
    days = br.index
    got = thrust_days(deemer_ratio(br.adv, br.dec))
    pos = days.get_indexer(got)
    for d in DEEMER:                                   # every published signal is found within 2 trading days
        k = days.searchsorted(pd.Timestamp(d))
        assert np.abs(pos - k).min() <= 2, d
    ref = days.searchsorted(pd.DatetimeIndex(DEEMER))
    for k in pos:                                      # and nothing fires outside 30 trading days after a published signal
        gap = k - ref
        assert ((gap >= -2) & (gap <= 30)).any(), days[k]
