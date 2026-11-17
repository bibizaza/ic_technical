import pandas as pd
import pytest
from research.us_gauge.run_s2 import mapped_etf, pick_etfs, monthly_returns, tilt_returns


def test_mapping_and_telecom_boundary():
    expected = dict(NoDur='XLP', Durbl='XLY', Manuf='XLI', Enrgy='XLE', Chems='XLB', BusEq='XLK', Utils='XLU', Shops='XLY', Hlth='XLV', Money='XLF', Other=None)
    for industry, etf in expected.items():
        assert mapped_etf(industry, pd.Timestamp('2018-05-31')) == etf
    assert mapped_etf('Telcm', pd.Timestamp('2018-05-31')) == 'IYZ'
    assert mapped_etf('Telcm', pd.Timestamp('2018-06-30')) == 'XLC'


def test_dedupe_and_skip_missing():
    scores = pd.Series([9, 8, 7, 6, 5, 4], index=['Other', 'Durbl', 'Shops', 'Telcm', 'Manuf', 'Hlth'])
    available = pd.Series({'XLY': .01, 'XLC': float('nan'), 'XLI': .02, 'XLV': .03})
    assert pick_etfs(scores, available, pd.Timestamp('2018-06-30')) == ['XLY', 'XLI', 'XLV']


def test_month_end_levels_and_missing_month():
    levels = pd.Series([100., 110., 121.], pd.to_datetime(['2000-01-28', '2000-02-29', '2000-04-28']))
    r = monthly_returns(levels)
    assert r.loc['2000-02-29'] == pytest.approx(.1)
    assert pd.isna(r.loc['2000-04-30'])


def test_prediction_lag_cost_and_return_month():
    idx = pd.date_range('2000-01-31', periods=3, freq='ME')
    pred = pd.DataFrame([[3, 2, 1]] * 3, index=idx, columns=['NoDur', 'Manuf', 'Hlth'])
    etf = pd.DataFrame(.02, index=idx, columns=['SPY','XLP','XLY','XLI','XLE','XLB','XLK','XLU','XLV','XLF'])
    etf.loc[idx[1], 'XLP'] = .1
    got, held = tilt_returns(pred, etf, pd.Series(.001, idx))
    assert got.index[0] == idx[1]
    assert got.S2.iloc[0] == pytest.approx(.7*.02 + .1*(.1+.02+.02) - .001)
    assert got.SPY.iloc[0] == pytest.approx(.019)
    assert got.turnover.iloc[1] == 0
    assert held.XLP.sum() == 2
