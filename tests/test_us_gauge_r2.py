"""Pre-registered step-1b transitions and effective exposure timing."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.us_gauge.trading import r2_positions, weights_from_positions, sleeve_weights


def run(ds, trend, rev):
    dates = pd.to_datetime(ds)
    return r2_positions(dates, pd.Series(trend, dates), pd.Series(rev, dates))


def test_positive_revisions_cannot_prevent_first_monthly_exit():
    p = run(['2024-01-02', '2024-02-06'], [60, 44], [1, 1])
    assert p.iloc[-1].tolist() == ['Cash', 'trend', False]


def test_positive_exit_does_not_arm_or_reenter():
    p = run(['2024-01-02', '2024-01-09', '2024-01-16'], [60, 29, 40], [1, 1, 1])
    assert p.pos.tolist() == ['Invested', 'Cash', 'Cash']
    assert not p.armed.any()


def test_negative_then_recovered_revisions_enter_earnings_midmonth():
    p = run(['2024-01-02', '2024-01-09', '2024-01-16', '2024-01-23'],
            [60, 29, 40, 40], [1, 1, -1, 0])
    assert p.iloc[-2].tolist() == ['Cash', 'trend', True]
    assert p.iloc[-1].tolist() == ['Invested', 'earnings', False]


def test_earnings_hands_back_to_trend_without_trade():
    p = run(['2024-01-02', '2024-01-09', '2024-01-16'], [40, 40, 45], [-1, 0, -1])
    assert p.pos.tolist() == ['Cash', 'Invested', 'Invested']
    assert p['mode'].tolist() == ['trend', 'earnings', 'trend']


def test_earnings_monthly_exit():
    p = run(['2024-01-02', '2024-01-09', '2024-02-06'], [40, 40, 44], [-1, 0, -1])
    assert p.iloc[-1].tolist() == ['Cash', 'trend', True]


def test_monthly_trend_reentry_has_priority():
    p = run(['2024-01-02', '2024-02-06'], [40, 45], [-1, 0])
    assert p.iloc[-1].tolist() == ['Invested', 'trend', True]


def test_nan_never_arms_or_triggers_reentry():
    p = run(['2024-01-02', '2024-01-09', '2024-01-16', '2024-01-23'],
            [40]*4, [np.nan, 1, -1, np.nan])
    assert p.pos.eq('Cash').all()
    assert p.armed.tolist() == [False, False, True, True]


@pytest.mark.parametrize('trend,rev,expected', [(29, -1, 'Cash'), (30, -1, 'Invested'),
                                               (29, 0, 'Invested'), (29, np.nan, 'Invested')])
def test_earnings_early_exit_boundaries(trend, rev, expected):
    p = run(['2024-01-02', '2024-01-09', '2024-01-16'], [40, 40, trend], [-1, 0, rev])
    assert p.iloc[-1].pos == expected


def test_no_lookahead_every_decision():
    dates = pd.date_range('2024-01-02', periods=90, freq='W-TUE')
    rng = np.random.default_rng(7)
    dmas = pd.Series(rng.choice([29, 30, 44, 45, 60], len(dates)), dates)
    rev = pd.Series(rng.choice([-1, 0, 1, np.nan], len(dates)), dates)
    full = r2_positions(dates, dmas, rev)
    for d in dates:
        pd.testing.assert_series_equal(full.loc[d], r2_positions(dates[dates <= d], dmas.loc[:d], rev.loc[:d]).loc[d])


@pytest.mark.parametrize('cash_weight', [0., .5])
def test_weights_t_plus_two_and_terminal_decision(cash_weight):
    days = pd.bdate_range('2024-01-01', '2024-01-19')
    dates = days[[1, 6, 13]]
    positions = pd.Series([True, False, True], dates)
    w = weights_from_positions(days, positions, cash_weight)
    assert w.iloc[:3].isna().all()
    assert w.iloc[3:8].eq(1.).all()
    assert w.iloc[8:].eq(cash_weight).all()
    expected = sleeve_weights(days, dates, ~positions, pd.Series(True, dates), cash_weight)
    pd.testing.assert_series_equal(w, expected)


def test_step1b_runner_outputs(tmp_path, monkeypatch):
    from research.us_gauge import run_step1b as runner
    from research.us_gauge.trading import decision_dates
    from research.us_gauge.gauge import gauge_frame
    days = pd.bdate_range('1993-01-01', '2011-03-16')
    inputs = pd.DataFrame({'eq_ret': np.random.default_rng(1).normal(.0003, .01, len(days)),
                           'cash_ret': .0001}, index=days)
    dates = decision_dates(days, '1993-01-01')
    gf = gauge_frame(pd.Series(np.where(np.arange(len(dates)) % 17 < 8, 25, 60), dates),
                     pd.Series(np.where(np.arange(len(dates)) % 13 < 6, -1., 1.), dates))
    monkeypatch.setattr(runner, 'load_inputs', lambda: inputs)
    monkeypatch.setattr(runner, 'build_signals', lambda inputs: gf)
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    runner.main()
    table = pd.read_csv(tmp_path / 'out/step1b_metrics.csv')
    assert len(table) == 27
    assert set(table.rule) == {'BH', 'R0', 'R1', 'R2', 'R3'}
    assert set(table.period) == {'FULL', 'H1', 'H2'}
    assert set(p.name for p in (tmp_path / 'out').iterdir()) == {
        'step1b_metrics.csv', 'step1b_positions_r2.csv', 'step1b_drawdowns.csv'}
    pos = pd.read_csv(tmp_path / 'out/step1b_positions_r2.csv', index_col=0, parse_dates=True)
    expected = r2_positions(gf.loc['1993-04-01':].index, gf.dmas, gf.rev_z)
    pd.testing.assert_frame_equal(pos, expected, check_names=False, check_freq=False)
    report = (tmp_path / 'RESULTS_step1b.md').read_text()
    assert 'R2 vs R0' in report and 'R3 vs R0' in report
    # R3 must use monthly revisions alone, with no deep exits.
    decisions = gf.loc['1993-04-01':]
    from research.us_gauge.trading import net_returns
    from research.us_gauge.run_step1 import _metric_rows
    w = sleeve_weights(days, decisions.index, decisions.rev_z < 0,
                       pd.Series(False, decisions.index), 0.)
    r = net_returns(w, inputs.eq_ret, inputs.cash_ret).dropna()
    expected_rows = pd.DataFrame(_metric_rows('R3', 'all', r, w.reindex(r.index), inputs.cash_ret))
    pd.testing.assert_frame_equal(table[(table.rule == 'R3') & (table.version == 'all')].reset_index(drop=True), expected_rows)
