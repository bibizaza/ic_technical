import numpy as np
import pandas as pd
from research.us_gauge import run_t
from research.us_gauge.run_fast import states


def test_offset_1_equals_month_lock():
    d = pd.bdate_range('2020-01-01', periods=300)
    rng = np.random.default_rng(0)
    dm = pd.Series(rng.integers(0, 100, 300), index=d)
    cash, deep = dm < 45, dm < 30
    a = states(d, cash, deep, dm, None)
    b = run_t.states_offset(d, cash, deep, 1)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_offset_sets_state_on_kth_day():
    d = pd.bdate_range('2020-01-01', '2020-03-31')
    cash = pd.Series(False, index=d)
    cash.loc['2020-02-07':] = True           # 2020-02-07 is the 5th trading day (bdays) of Feb
    s = run_t.states_offset(d, cash, pd.Series(False, index=d), 6)
    assert np.isnan(s.loc['2020-01-07'])     # before the first anchor
    assert s.loc['2020-01-08'] == 1.          # 6th bday of Jan
    assert s.loc['2020-02-07'] == 1.          # locked until the anchor
    assert s.loc['2020-02-10'] == 0.          # 6th bday of Feb


def test_canary_cap_levels():
    i = pd.bdate_range('2010-01-01', periods=300)
    up, down = pd.Series(np.linspace(100, 130, 300), index=i), pd.Series(np.linspace(130, 100, 300), index=i)
    assert run_t.canary_cap(up, up, i).iloc[-1] == 1.
    assert run_t.canary_cap(up, down, i).iloc[-1] == .5
    assert run_t.canary_cap(down, down, i).iloc[-1] == 0.
    assert np.isnan(run_t.canary_cap(up, up, i).iloc[250])


def test_credit_cap_needs_both_conditions():
    i = pd.bdate_range('2000-01-01', periods=700)
    oas = pd.Series(3., index=i) + np.random.default_rng(1).normal(0, .01, 700)
    oas.iloc[-10:] += 2.                     # big spread jump
    px_up = pd.Series(np.linspace(100, 200, 700), index=i)
    px_dn = px_up.copy(); px_dn.iloc[-10:] = 50.
    assert run_t.credit_cap(oas, px_up).iloc[-1] == 1.     # spread jump but price above its 50-day average
    assert run_t.credit_cap(oas, px_dn).iloc[-1] == .5
