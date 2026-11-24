import numpy as np
import pandas as pd
from research.us_gauge import run_kr


def test_single_cap_levels():
    i = pd.bdate_range('2010-01-01', periods=300)
    up = pd.Series(np.linspace(100, 130, 300), index=i)
    down = pd.Series(np.linspace(130, 100, 300), index=i)
    assert run_kr.single_cap(up, i).iloc[-1] == 1.
    assert run_kr.single_cap(down, i).iloc[-1] == .5
    assert np.isnan(run_kr.single_cap(up, i).iloc[250])   # needs 252 days


def test_single_cap_uses_no_future_data():
    i = pd.bdate_range('2010-01-01', periods=400)
    s = pd.Series(np.linspace(100, 140, 400), index=i)
    a = run_kr.single_cap(s, i)
    s2 = s.copy()
    s2.iloc[350:] = 1.                                     # crash after day 349
    b = run_kr.single_cap(s2, i)
    pd.testing.assert_series_equal(a.iloc[:350], b.iloc[:350])


def test_pass_bar_is_four_way_bonferroni():
    assert abs(run_kr.P_PASS - .025) < 1e-12
