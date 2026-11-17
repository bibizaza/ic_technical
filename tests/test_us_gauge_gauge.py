"""Production trend reuse and fixed two-key state/band boundaries."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.momentum import _compute_technical_score, _compute_mars_momentum
from research.us_gauge.data import STORE, load_inputs
from research.us_gauge.gauge import dmas_on, gauge_frame


@pytest.mark.skipif(not STORE.exists(), reason='Dropbox store missing')
def test_dmas_matches_production_formula():
    px = load_inputs().px
    d = pd.Timestamp('2026-11-10')
    t = _compute_technical_score(px, d)
    m = _compute_mars_momentum(px, d, benchmark_close=None)
    assert dmas_on(px, pd.DatetimeIndex([d])).iloc[0] == int((t + m) / 2)


def test_dmas_no_lookahead():
    idx = pd.bdate_range('2000-01-03', periods=2000)
    px = pd.Series(100 * np.exp(np.random.default_rng(3).normal(0, .01, len(idx)).cumsum()), idx)
    for d in idx[[400, 900, 1500]]:
        pd.testing.assert_series_equal(dmas_on(px, pd.DatetimeIndex([d])), dmas_on(px.loc[:d], pd.DatetimeIndex([d])))


def test_gauge_rules():
    got = gauge_frame(pd.Series([50, 40, 40, 25, 40]), pd.Series([-1, .2, -.1, -2, np.nan]))
    assert got.cash_R1.tolist() == [False, False, True, True, True]
    assert got.cash_R0.tolist() == [False, True, True, True, True]
    assert got.G.tolist() == pytest.approx([1/3, .2, -.1, -4/3, -1/3])
    assert got.deep.tolist() == [False, False, False, True, False]


def test_band_edges():
    got = gauge_frame(pd.Series([0]*5), pd.Series([1., .5, 0., -.5, -.51]))
    assert got.band.tolist() == ['G>=1', '0.5-1', '0-0.5', '-0.5-0', 'G<-0.5']
