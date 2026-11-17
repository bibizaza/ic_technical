"""Monthly decisions, one-way early exits and close execution timing."""
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.us_gauge.trading import decision_dates, sleeve_weights, net_returns


def scenario():
    idx = pd.bdate_range('2024-01-01', '2024-03-20')
    dates = decision_dates(idx, '2024-01-01')
    return idx, dates, pd.Series(False, dates), pd.Series(False, dates)


def test_tuesday_holiday_uses_monday():
    idx = pd.bdate_range('2024-12-16', '2024-12-27').difference(pd.DatetimeIndex(['2024-12-24']))
    dates = decision_dates(idx, '2024-12-16')
    assert list(dates) == [pd.Timestamp('2024-12-17'), pd.Timestamp('2024-12-23')]


def test_monthly_sets_state_and_early_exit_only_when_deep():
    idx, dates, cash, deep = scenario()
    cash.loc['2024-01-16':] = True
    w = sleeve_weights(idx, dates, cash, deep, 0.)
    assert (w.loc['2024-01-04':'2024-02-07'] == 1.).all()
    assert w.loc['2024-02-08'] == 0.
    deep.loc['2024-01-16'] = True
    w = sleeve_weights(idx, dates, cash, deep, 0.)
    assert w.loc['2024-01-17'] == 1.
    assert w.loc['2024-01-18'] == 0.


def test_no_early_reentry():
    idx, dates, cash, deep = scenario()
    cash.loc['2024-01-02'] = True
    w = sleeve_weights(idx, dates, cash, deep, 0.)
    assert (w.loc['2024-01-04':'2024-02-07'] == 0.).all()
    assert w.loc['2024-02-08'] == 1.


def test_execution_lag():
    idx, dates, cash, deep = scenario()
    cash.loc['2024-01-16'] = deep.loc['2024-01-16'] = True
    w = sleeve_weights(idx, dates, cash, deep, 0.)
    assert w.loc[:'2024-01-03'].isna().all()
    assert w.loc['2024-01-16'] == w.loc['2024-01-17'] == 1.
    assert w.loc['2024-01-18'] == 0.


def test_half_version():
    idx, dates, cash, deep = scenario()
    cash.loc['2024-02-06'] = True
    w = sleeve_weights(idx, dates, cash, deep, .5)
    assert set(w.dropna()) == {.5, 1.}


def test_cost_charged_once():
    w = pd.Series([1., 1., 0., 0.])
    zero = pd.Series(0., w.index)
    got = net_returns(w, zero, zero)
    assert got.tolist() == pytest.approx([0., 0., -.001, 0.])
    assert got.sum() == pytest.approx(-.001)


def test_partial_last_month():
    idx, dates, cash, deep = scenario()
    w = sleeve_weights(idx, dates, cash, deep, 0.)
    assert w.iloc[-1] == 1.
    assert dates[-1] == pd.Timestamp('2024-03-19')
