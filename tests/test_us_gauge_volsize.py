import numpy as np
import pandas as pd
import pytest
from research.us_gauge.volsize import sigma, target_weight, daily_weights, turnover_per_year


def test_sigma_truncation_and_cap():
    r = pd.Series(np.random.default_rng(7).normal(0, .02, 100), pd.bdate_range('2000-01-01', periods=100))
    pd.testing.assert_series_equal(sigma(r).iloc[:60], sigma(r.iloc[:60]))
    assert sigma(r).iloc[20] == pytest.approx(r.iloc[:21].std() * np.sqrt(252))
    assert sigma(r).iloc[:20].isna().all()
    assert target_weight(r * .01).dropna().eq(1).all()
    assert target_weight(r).dropna().le(1).all()


def test_thresholds_and_t_plus_two():
    idx = pd.bdate_range('2000-01-04', '2000-02-10')
    dates = pd.to_datetime(['2000-01-04', '2000-01-11', '2000-01-18', '2000-01-25', '2000-02-01', '2000-02-08'])
    v = pd.Series([1., .76, .75, .51, .65, .40], dates)
    w = daily_weights(idx, dates, v)
    assert w.iloc[:2].isna().all()
    for d, expected in zip(dates, [1., 1., .75, .75, .65, .40]):
        assert w.iloc[idx.get_loc(d) + 2] == pytest.approx(expected)


def test_monthly_below_threshold_and_exact_boundary():
    idx = pd.bdate_range('2000-01-04', '2000-03-10')
    dates = pd.to_datetime(['2000-01-04', '2000-02-01', '2000-03-07'])
    w = daily_weights(idx, dates, pd.Series([.5, .59, .6], dates))
    assert w.loc['2000-02-03'] == .5
    assert w.loc['2000-03-09'] == .6


@pytest.mark.parametrize('cash_weight', [0., .5])
def test_v2_combined_threshold_and_forced_state_change(cash_weight):
    idx = pd.bdate_range('2000-01-04', '2000-02-10')
    dates = pd.to_datetime(['2000-01-04', '2000-01-11', '2000-01-18', '2000-01-25'])
    state = pd.Series([1., cash_weight, cash_weight, 1.], dates)
    v = pd.Series([.2, .2, .4, .4], dates)
    w = daily_weights(idx, dates, v, state)
    for d, expected in zip(dates, [.2, cash_weight * .2, cash_weight * .2, .4]):
        assert w.iloc[idx.get_loc(d) + 2] == pytest.approx(expected)


def test_turnover_is_weight_change_not_trade_count():
    w = pd.Series([1., .8, .8, .5], pd.bdate_range('2000-01-01', periods=4))
    assert turnover_per_year(w) == pytest.approx(.5 * 252 / 4)
    assert turnover_per_year(w, w.index[2:]) == pytest.approx(.3 * 252 / 2)
