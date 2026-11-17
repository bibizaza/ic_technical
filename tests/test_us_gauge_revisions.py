"""Fiscal-year comparators, missing estimates, and past-only normalization."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.us_gauge.data import STORE, load_inputs
from research.us_gauge.revisions import detect_rolls, revisions, rev_zscore


def sample():
    idx = pd.bdate_range('2000-01-03', periods=200)
    return pd.Series(np.r_[np.full(150, 100.), np.full(50, 110.)], idx), pd.Series(np.r_[np.full(150, 110.), np.full(50, 120.)], idx)


@pytest.mark.skipif(not STORE.exists(), reason='Dropbox store missing')
def test_one_roll_per_year_real():
    x = load_inputs()
    rolls = detect_rolls(x.fy1, x.fy2)
    rolls = rolls[(rolls.year >= 1991) & (rolls.year <= 2026)]
    assert list(rolls.year) == list(range(1991, 2027))
    assert set(rolls.month) <= {2, 3}
    assert 2009 in rolls.year


def test_rev_uses_fy2_across_roll():
    a, b = sample()
    assert revisions(a, b, pd.DatetimeIndex([a.index[150]])).iloc[160] == 0


def test_roll_on_window_edge():
    a, b = sample()
    # At the roll today, old FY2 is the comparator.
    assert revisions(a, b, pd.DatetimeIndex([a.index[150]])).iloc[150] == 0
    # A roll exactly 63 rows earlier is excluded; both revisions apply.
    got = revisions(a, b, pd.DatetimeIndex([a.index[97]])).iloc[160]
    assert got == pytest.approx((np.log(110 / 100) + np.log(120 / 110)) / 2)


def test_rev_nan_when_both_missing():
    a, b = sample()
    a.iloc[97] = b.iloc[97] = np.nan
    assert np.isnan(revisions(a, b, pd.DatetimeIndex([])).iloc[160])


def test_rev_uses_whichever_is_defined():
    a, b = sample()
    b.iloc[160] = np.nan
    assert revisions(a, b, pd.DatetimeIndex([])).iloc[160] == pytest.approx(np.log(1.1))


def test_zscore_excludes_today():
    r = pd.Series(np.r_[np.tile([-.01, .01], 630), 10.])
    z = rev_zscore(r)
    assert z.iloc[:756].isna().all()
    assert z.iloc[-1] > 5
    assert z.iloc[-1] == pytest.approx(10 / r.iloc[:1260].std())
