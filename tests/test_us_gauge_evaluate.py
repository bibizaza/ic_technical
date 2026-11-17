"""Known financial paths, paired resampling and forward switch frequencies."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.us_gauge.evaluate import metrics, bootstrap_not_better, switches_per_year, switch_risk_table, worst_drawdowns


def test_metrics_known_series():
    excess = pd.Series(.0004 + np.tile([-.01, .01], 126))
    cash = pd.Series(.0001, excess.index)
    got = metrics(excess + cash, cash)
    assert got['sharpe'] == pytest.approx(.0004 / (.01 * np.sqrt(252/251)) * np.sqrt(252), abs=1e-9)
    r = pd.Series([.1, -.5, 0.])
    assert metrics(r, r * 0)['maxdd'] == pytest.approx(-.5)
    assert metrics(pd.Series([-.1, .0]), pd.Series([0., 0.]))['maxdd'] == pytest.approx(-.1)


def returns():
    r = pd.Series(np.random.default_rng(8).normal(.0003, .01, 1000))
    return r, r * 0


def test_bootstrap_identical_is_one():
    r, cash = returns()
    assert bootstrap_not_better(r, r, cash) == 1.


def test_bootstrap_dominant_is_zero():
    r, cash = returns()
    assert bootstrap_not_better(r + .001, r, cash) == 0.


def test_bootstrap_seed_reproducible():
    r, cash = returns()
    r1 = r + pd.Series(np.random.default_rng(9).normal(0, .005, len(r)))
    assert bootstrap_not_better(r1, r, cash) == bootstrap_not_better(r1, r, cash)


def test_switch_risk_grades():
    states, bands = [], []
    for band, flips in [('G>=1', 1), ('0.5-1', 4), ('0-0.5', 7)]:
        for j in range(10):
            states.extend([False] + [j < flips] * 14)
            bands.extend([band] + [None] * 14)
    gf = pd.DataFrame({'cash_R1': states, 'band': bands}, index=pd.date_range('2000-01-04', periods=len(states), freq='W-TUE'))
    got = switch_risk_table(gf, 'cash_R1').set_index('band')
    assert got.loc[['G>=1', '0.5-1', '0-0.5'], 'grade'].tolist() == ['low', 'medium', 'high']
    assert got.loc[['G>=1', '0.5-1', '0-0.5'], 'share_4w'].tolist() == pytest.approx([.1, .4, .7])


def test_switches_and_drawdown_dates():
    idx = pd.bdate_range('2024-01-01', periods=252)
    w = pd.Series(np.r_[np.ones(100), np.zeros(100), np.ones(52)], idx)
    assert switches_per_year(w) == 2.
    idx = pd.bdate_range('2024-01-01', periods=5)
    dd = worst_drawdowns(pd.Series([.1, -.5, 1., -.2, .1], idx))
    assert dd.iloc[0]['depth'] == pytest.approx(-.5)
    assert dd.iloc[0]['peak'] == idx[0]
    assert dd.iloc[0]['trough'] == idx[1]
    assert dd.iloc[0]['recovery'] == idx[2]
    assert pd.isna(dd.iloc[1]['recovery'])
