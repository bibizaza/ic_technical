import numpy as np
import pandas as pd
from research.us_gauge.run_s import shock_episodes


def _px(fall_days, depth, tail):
    up = np.linspace(100, 120, 300)
    down = np.linspace(120, 120 * (1 - depth), fall_days)
    return pd.Series(np.r_[up, down, tail], index=pd.bdate_range('2000-01-03', periods=300 + fall_days + len(tail)))


def test_fast_shock_triggers_after_10pct_bounce():
    tail = np.linspace(96, 110, 40)          # low 96 (-20%), +10% at 105.6
    px = _px(30, .20, tail)
    ep = shock_episodes(px, pd.Series(0., index=px.index))
    assert len(ep) == 1
    r = ep.iloc[0]
    assert px.loc[r.entry] >= 1.10 * r.L and px.iloc[r.i_entry - 1] < 1.10 * r.L
    assert r.reason == 'time' or r.reason == 'open'


def test_slow_fall_does_not_trigger():
    tail = np.linspace(96, 110, 40)
    px = _px(120, .20, tail)                  # low more than 63 days after the high
    assert shock_episodes(px, pd.Series(0., index=px.index)).empty


def test_no_episode_when_base_already_invested():
    px = _px(30, .20, np.linspace(96, 110, 40))
    assert shock_episodes(px, pd.Series(1., index=px.index)).empty
