import numpy as np
import pandas as pd
from research.us_gauge import run_q


def q_index(n):
    return pd.date_range('2000-03-31', periods=n, freq='QE')


def test_signals_formula():
    i = q_index(5)
    z1 = {'FA103164105': pd.Series([-4., -4., -4., -4., -8.], index=i),
          'FA653064100': pd.Series([1., 1., 1., 1., 5.], index=i),
          'FA563064100': pd.Series([np.nan, np.nan, 1., 1., 1.], index=i),
          'LM883164115': pd.Series(100., index=i), 'LM893064105': pd.Series(200., index=i)}
    s = run_q.signals(z1)
    assert np.isnan(s.Q1.iloc[2])
    assert abs(s.Q1.iloc[3] - .04) < 1e-12           # -mean(-4) / 100
    assert abs(s.Q1.iloc[4] - .05) < 1e-12           # -mean(-4,-4,-4,-8) / 100
    assert abs(s.Q2.iloc[3] - 1.5 / 200) < 1e-12     # mean(1+0, 1+0, 1+1, 1+1) / 200


def test_trailing_z_excludes_current():
    s = pd.Series([1., 2., 3., 10.])
    z = run_q.trailing_z(s, n=3, n_min=3)
    assert z.iloc[:3].isna().all()
    assert abs(z.iloc[3] - (10 - 2) / 1.) < 1e-12


def test_key_directions():
    z = pd.Series([-1., 0., 1., np.nan])
    assert run_q.key_in(z, 'Q1').tolist()[:3] == [0., 1., 1.]
    assert run_q.key_in(z, 'Q2').tolist()[:3] == [1., 1., 0.]
    assert np.isnan(run_q.key_in(z, 'Q1').iloc[3])


def test_to_daily_waits_120_days():
    k = pd.Series([1., 0.], index=pd.to_datetime(['2001-03-31', '2001-06-30']))
    idx = pd.bdate_range('2001-01-01', '2002-01-31')
    d = run_q.to_daily(k, idx)
    assert np.isnan(d.loc['2001-07-27'])               # 2001-03-31 + 120 days = 2001-07-29 (Sunday)
    assert d.loc['2001-07-30'] == 1.
    assert d.loc['2001-10-26'] == 1. and d.loc['2001-10-29'] == 0.   # 2001-06-30 + 120 = 2001-10-28 (Sunday)
