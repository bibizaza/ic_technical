import pandas as pd
from research.us_gauge.run_s_live import s_weight


def _setup():
    idx = pd.bdate_range('2020-01-01', periods=12)
    v = pd.Series([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 1., 1.], index=idx)
    ep = pd.DataFrame({'i_entry': [2], 'i_end': [6]})
    return idx, v, ep


def test_frozen_uses_trigger_close_value_on_earn_days():
    idx, v, ep = _setup()
    s = s_weight(idx, ep, v, 'frozen')
    assert list(s.iloc[4:8]) == [.3] * 4
    assert s.iloc[:4].eq(0).all() and s.iloc[8:].eq(0).all()


def test_live_lags_target_by_two_days():
    idx, v, ep = _setup()
    s = s_weight(idx, ep, v, 'live')
    assert list(s.iloc[4:8].round(6)) == [.3, .4, .5, .6]
    assert s.iloc[8:].eq(0).all()


def test_full_is_one_and_clipped_at_end_of_sample():
    idx, v, _ = _setup()
    ep = pd.DataFrame({'i_entry': [9], 'i_end': [11]})
    s = s_weight(idx, ep, v, 'full')
    assert s.iloc[11] == 1. and s.iloc[:11].eq(0).all()
