import numpy as np
import pandas as pd
from research.us_gauge.run_c import episodes, floor_weights, zscore


def _path(tail):
    rng = np.random.default_rng(3)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, .005, 300)))
    crash = calm[-1] * np.exp(np.cumsum(np.full(10, -.03)))
    p = np.r_[calm, crash, crash[-1] * np.exp(np.cumsum(tail))]
    return pd.Series(p, index=pd.bdate_range('2000-01-03', periods=len(p)))


def test_crash_arms_enters_on_positive_week_and_times_out():
    px = _path(np.full(120, .002))
    assert zscore(px).iloc[309] <= -3
    ep = episodes(px)
    assert len(ep) == 1
    r = ep.iloc[0]
    assert r.i_entry > 309                       # not the low day itself
    assert px.iloc[r.i_entry] > px.iloc[r.i_entry - 5]
    assert r.reason == 'time' and r.i_end == r.i_entry + 63


def test_new_low_stops_episode():
    tail = np.r_[np.full(8, .01), np.full(30, -.02), np.full(40, .0)]
    ep = episodes(_path(tail))
    assert ep.iloc[0].reason == 'stop'
    px = _path(tail)
    assert px.iloc[ep.iloc[0].i_end] < ep.iloc[0].L


def test_floor_earns_from_t_plus_2_through_end_plus_1():
    idx = pd.bdate_range('2001-01-01', periods=20)
    f = floor_weights(idx, [(3, 8)], .5)
    assert (f.iloc[5:10] == .5).all() and f.iloc[4] == 0 and f.iloc[10] == 0
