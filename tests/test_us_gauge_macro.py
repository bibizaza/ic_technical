import numpy as np
import pandas as pd
from research.us_gauge import run_macro


def test_out_key_uses_previous_12_months():
    i = pd.date_range('2000-01-01', periods=14, freq='MS')
    x = pd.Series([5.] * 12 + [5.1, 5.], index=i)
    k = run_macro.out_key(x)
    assert k.iloc[:12].isna().all()
    assert k.iloc[12] == 0.            # 5.1 > mean of 12 x 5.0 -> out
    assert k.iloc[13] == 1.            # 5.0 <= mean(11 x 5.0, 5.1) -> in


def test_to_daily_waits_until_the_10th_of_next_month():
    k = pd.Series([0., 1.], index=pd.to_datetime(['2001-05-01', '2001-06-01']))
    idx = pd.bdate_range('2001-05-01', '2001-08-31')
    d = run_macro.to_daily(k, idx)
    assert np.isnan(d.loc['2001-06-08'])
    assert d.loc['2001-06-11'] == 0.   # 2001-06-10 is a Sunday
    assert d.loc['2001-07-09'] == 0. and d.loc['2001-07-10'] == 1.
