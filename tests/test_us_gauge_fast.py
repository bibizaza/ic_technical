import pandas as pd
from research.us_gauge.run_fast import states, early_reentries


def _d():
    return pd.DatetimeIndex(['2001-01-02', '2001-01-09', '2001-01-16', '2001-01-23', '2001-02-06'])


def test_base_rule_waits_for_month_start():
    d = _d()
    cash = pd.Series([True, True, False, False, False], index=d)
    deep = cash & False
    dmas = pd.Series([40, 40, 62, 65, 65], index=d)
    assert states(d, cash, deep, dmas, None).tolist() == [0, 0, 0, 0, 1]


def test_x_reenters_intra_month_at_60():
    d = _d()
    cash = pd.Series([True, True, False, False, False], index=d)
    dmas = pd.Series([40, 59, 60, 65, 65], index=d)
    st = states(d, cash, cash & False, dmas, 60)
    assert st.tolist() == [0, 0, 1, 1, 1]
    assert early_reentries(st, d) == [d[2]]


def test_forced_placebo_date_replaces_dmas_trigger():
    d = _d()
    cash = pd.Series([True, True, False, False, False], index=d)
    dmas = pd.Series([40, 40, 62, 65, 65], index=d)
    st = states(d, cash, cash & False, dmas, None, forced={d[1]})
    assert st.tolist() == [0, 1, 1, 1, 1]
