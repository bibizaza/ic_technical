import numpy as np
import pandas as pd
from research.us_gauge.run_p import base_states, spells, episodes


def test_base_states_match_monthly_rule():
    d = pd.DatetimeIndex(['2001-01-02', '2001-01-09', '2001-01-16', '2001-02-06'])
    cash = pd.Series([False, True, True, True], index=d)
    deep = pd.Series([False, False, True, False], index=d)
    assert base_states(d, cash, deep).tolist() == [1., 1., 0., 0.]


def _setup(stress_vals, px_vals, state_vals):
    idx = pd.bdate_range('2001-01-01', periods=60)
    dates = idx[30::5][:len(state_vals)]
    state = pd.Series(state_vals, index=dates, dtype=float)
    stress = pd.Series(np.nan, index=idx)
    px = pd.Series(100., index=idx)
    vix = pd.Series(20., index=idx)
    for d, s, p in zip(dates, stress_vals, px_vals):
        stress.loc[d] = s
        px.loc[d] = p
    stress = stress.ffill().fillna(10.)
    return idx, state, stress, vix, px


def test_entry_when_stress_quarter_off_peak_and_ends_when_base_invested():
    idx, state, stress, vix, px = _setup([10, 7.4, 7, 6], [90, 95, 96, 97], [0, 0, 0, 1])
    ep = episodes(idx, state, stress, vix, px)
    assert len(ep) == 1 and ep.entry[0] == state.index[1] and ep.reason[0] == 'base invested'


def test_stress_stop_and_no_reentry_in_spell():
    idx, state, stress, vix, px = _setup([10, 7, 11, 5, 5], [90, 95, 96, 97, 98], [0, 0, 0, 0, 0])
    ep = episodes(idx, state, stress, vix, px)
    assert len(ep) == 1 and ep.reason[0] == 'stress stop'
    assert spells(state).nunique() == 1
