"""Pre-registered VIX signal, R4 truth table and step-1c outputs."""
import sys
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.us_gauge import gauge


def test_vix_trailing_mean_ends_at_t_and_prefixes():
    v = pd.Series([24.] * 20 + [45., 0., 100.], pd.bdate_range('2024-01-01', periods=23))
    got = gauge.vix_high(v)
    assert not got.iloc[:20].any()
    assert got.iloc[20]  # (20 * 24 + 45) / 21 == 25
    assert not got.iloc[21]
    for n in range(1, len(v) + 1):
        pd.testing.assert_series_equal(got.iloc[:n], gauge.vix_high(v.iloc[:n]))


def test_vix_boundary_and_nan():
    v = pd.Series([25.] * 21 + [np.nan] + [25.] * 21)
    got = gauge.vix_high(v)
    assert got.iloc[20] and got.iloc[-1]
    assert not got.iloc[21:42].any()
    assert not gauge.vix_high(pd.Series([np.nan] * 30)).any()


@pytest.mark.parametrize('t,r,v', list(product([False, True], repeat=3)))
def test_r4_truth_table(t, r, v):
    got = gauge.r4_cash(pd.Series([t]), pd.Series([r]), pd.Series([v]))
    assert got.iloc[0] == (not t and (not r or v))


def test_step1c_runner_outputs(tmp_path, monkeypatch):
    from research.us_gauge import run_step1c as runner
    from research.us_gauge.trading import decision_dates, sleeve_weights, net_returns
    from research.us_gauge.evaluate import metrics
    days = pd.bdate_range('1993-01-01', '2022-03-16')
    inputs = pd.DataFrame({'eq_ret': np.random.default_rng(1).normal(.0003, .01, len(days)),
                           'cash_ret': .0001, 'vix': np.where(np.arange(len(days)) % 90 < 45, 30., 20.)}, index=days)
    dates = decision_dates(days, '1993-01-01')
    gf = gauge.gauge_frame(pd.Series(np.where(np.arange(len(dates)) % 17 < 8, 25, 60), dates),
                           pd.Series(np.where(np.arange(len(dates)) % 13 < 6, -1., 1.), dates))
    monkeypatch.setattr(runner, 'load_inputs', lambda: inputs)
    monkeypatch.setattr(runner, 'build_signals', lambda inputs: gf.copy())
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    runner.main()
    table = pd.read_csv(tmp_path / 'out/step1c_metrics.csv')
    assert len(table) == 44
    assert set(table.rule) == {'BH', 'R0', 'R1', 'R2', 'R3', 'R4'}
    assert set(table.period) == {'FULL', 'H1', 'H2', 'P2021'}
    assert list(table.columns) == ['rule', 'version', 'period', 'sharpe', 'cagr', 'maxdd', 'calmar', 'vol', 'switches_per_year', 'share_in_cash']
    assert set(p.name for p in (tmp_path / 'out').iterdir()) == {'step1c_metrics.csv', 'step1c_drawdowns.csv'}
    decisions = gf.loc['1993-04-01':]
    cash_signal = ~decisions.T_in & (~decisions.R_in | gauge.vix_high(inputs.vix).reindex(decisions.index))
    for version, cash_weight in [('all', 0.), ('half', .5)]:
        w = sleeve_weights(days, decisions.index, cash_signal, decisions.dmas < 30, cash_weight)
        r = net_returns(w, inputs.eq_ret, inputs.cash_ret).dropna()
        subset = r.loc['2021-01-01':]
        row = table[(table.rule == 'R4') & (table.version == version) & (table.period == 'P2021')].iloc[0]
        for key, value in metrics(subset, inputs.cash_ret.reindex(subset.index)).items():
            assert row[key] == pytest.approx(value)
        assert row.switches_per_year == pytest.approx(w.diff().abs().gt(0).reindex(subset.index).sum() / (len(subset) / 252))
    report = (tmp_path / 'RESULTS_step1c.md').read_text()
    assert 'Verdict:' in report and 'Latest R4 reading' in report and 'Deviations' in report
    assert 'P2021' in report and 'VIX 21d average' in report
