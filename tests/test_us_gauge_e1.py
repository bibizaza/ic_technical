"""Pre-registered E1 direction, timing, costs and point-in-time contract."""
import numpy as np
import pandas as pd
import pytest
from research.us_gauge import run_e1 as e1


def prices():
    idx = pd.bdate_range('2003-04-30', periods=650)
    return pd.DataFrame({'SPY': 100., 'RSP': 100 * np.exp(np.arange(len(idx)) * .001)}, index=idx)


def test_positive_relative_momentum_selects_rsp_and_zero_selects_spy():
    px = prices()
    sig = e1.signals(px)
    assert sig.rel.gt(0).all()
    assert sig.rsp.eq(1).all()
    px.RSP = 100.
    assert e1.signals(px).rsp.eq(0).all()
    px.RSP = 100 * np.exp(-np.arange(len(px)) * .001)
    assert e1.signals(px).rsp.eq(0).all()
    assert sig.index[0] == pd.Timestamp('2003-11-04')
    assert not sig.index.to_period('M').duplicated().any()


def test_signal_uses_only_data_through_t():
    px = prices()
    full = e1.signals(px)
    for t in full.index[::3]:
        pd.testing.assert_frame_equal(full.loc[:t], e1.signals(px.loc[:t]))
        altered = px.copy()
        altered.loc[altered.index > t, 'RSP'] *= 50
        pd.testing.assert_frame_equal(full.loc[:t], e1.signals(altered).loc[:t])


def test_full_switch_cost_and_t_plus_two_exposure():
    idx = pd.bdate_range('2020-01-01', periods=10)
    px = pd.DataFrame({'SPY': 100., 'RSP': 100.}, index=idx)
    sig = pd.DataFrame({'rsp': [0., 1., 0.]}, index=idx[[0, 3, 6]])
    w, r = e1.simulate(px, sig)
    assert w.iloc[:2].isna().all()
    assert w.iloc[2:5].eq(0).all()
    assert w.iloc[5:8].eq(1).all()
    assert w.iloc[8:].eq(0).all()
    assert r.iloc[[5, 8]].to_numpy() == pytest.approx([-.002, -.002])
    assert r.dropna().sum() == pytest.approx(-.004)
