import numpy as np
import pandas as pd
from research.us_gauge import core_slide_data as C
from research.us_gauge import run_t

I = pd.bdate_range('2010-01-01', periods=320)
UP = pd.Series(np.linspace(100, 130, 320), index=I)
DOWN = pd.Series(np.linspace(130, 100, 320), index=I)


def test_canary_matches_tested_t2_cap():
    for a, e in ((UP, UP), (UP, DOWN), (DOWN, UP), (DOWN, DOWN)):
        assert C.canary(a, e, I[-1])['canary'] == run_t.canary_cap(a, e, I).iloc[-1]


def test_canary_ignores_data_after_asof():
    e = UP.copy()
    e.iloc[-5:] = 1.                                       # crash after the as-of day
    out = C.canary(UP, e, I[-6])
    assert out['canary'] == 1. and out['canary_asof'] == str(I[-6].date())


def test_canary_reports_oldest_data_date():
    out = C.canary(UP.iloc[:-3], UP, I[-1])
    assert out['canary_asof'] == str(I[-4].date())
