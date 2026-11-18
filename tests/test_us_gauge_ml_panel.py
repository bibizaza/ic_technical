import socket
import numpy as np
import pandas as pd
import pytest
from research.us_gauge.ml import panel


def _fake_loader(calls):
    def load(name):
        calls.append(name)
        idx = pd.bdate_range('1987-01-02', periods=4000)
        s = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
        s.iloc[5] = np.nan
        s.iloc[6] = 0.
        s.iloc[7] = -1.
        s = pd.concat([s, s.iloc[[10]] * 3])          # duplicate date with a different value
        return s
    return load


def test_clean_drops_nan_nonpositive_duplicates():
    calls = []
    px, big = panel.clean(_fake_loader(calls)('X'), 'AEX', end=None)
    assert px.index.is_unique and px.notna().all() and (px > 0).all()
    assert px.iloc[7] == pytest.approx(np.linspace(100, 200, 4000)[10])   # first copy of the duplicate is kept
    assert len(px) == 4000 - 3


def test_bvsp_cut_and_big_moves_listed_not_removed():
    idx = pd.bdate_range('1992-01-01', '1996-12-31')
    s = pd.Series(100., index=idx)
    s.loc['1993-06-01'] = 200.                        # a +100% day before the cut
    s.loc['1996-03-01':] = 140.                       # a +40% day after the cut
    px, big = panel.clean(s, 'BVSP', end=None)
    assert px.index[0] == pd.Timestamp('1995-01-02')
    assert list(big.index) == [pd.Timestamp('1996-03-01')]
    assert px.loc['1996-03-01'] == 140.               # listed, not removed
    other, big2 = panel.clean(s, 'MXX', end=None)
    assert other.index[0] == idx[0] and len(big2) == 3


def test_data_end_cut():
    s = pd.Series(1., index=pd.bdate_range('2026-11-01', '2026-11-20'))
    px, _ = panel.clean(s, 'AEX')
    assert px.index[-1] <= panel.DATA_END


def test_load_panel_no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('network used')
    monkeypatch.setattr(socket.socket, 'connect', boom)
    calls = []
    P, B = panel.load_panel(['GSPC', 'SXXP', 'BVSP'], end=None, loader=_fake_loader(calls))
    assert calls == ['GSPC.INDX', 'BBG_SXXP', 'BVSP.INDX']
    assert P['BVSP'].index[0] >= panel.BVSP_START


def test_store_loader_is_offline():
    from research.us_gauge import data
    with pytest.raises(ValueError):
        data.load('GSPC.INDX', refresh=True)


def test_panel_definitions():
    assert len(panel.MARKETS) == 21 and len(panel.CLEAN) == 13
    assert set(panel.REGION) == set(panel.MARKETS)
    assert sorted(set(panel.REGION.values())) == ['asia_pac_dev', 'emerging', 'europe', 'north_america']
