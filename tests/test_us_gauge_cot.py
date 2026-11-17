"""Section 7b timing, warm-up, vote and real-data leakage checks."""
from itertools import product
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.us_gauge.cot import cot_signal
from research.us_gauge.gauge import r5_cash
from research.us_gauge.data import load


def reports(n=200):
    dates = pd.date_range('2006-06-13', periods=n, freq='W-TUE')
    return [pd.Series(values, dates) for values in
            (np.arange(n, dtype=float), np.zeros(n), np.ones(n), np.full(n, 3.))]


def test_release_lag_shutdown_and_latest_report():
    args = reports()
    args[3].iloc[150] = 50
    dates = pd.DatetimeIndex([args[0].index[150] + pd.Timedelta(days=3),
                              args[0].index[158]])
    got = cot_signal(*args, dates)
    assert got.report_date_used.iloc[0] == args[0].index[149]
    # The late arrival of an older report must not replace a newer report.
    assert got.report_date_used.iloc[1] == args[0].index[157]
    exact = args[0].index[149] + pd.Timedelta(days=3)
    assert cot_signal(*args, pd.DatetimeIndex([exact])).report_date_used.iloc[0] == args[0].index[149]


def test_z_excludes_current_uses_previous_156_and_minimum_104():
    args = reports()
    args[0].iloc[180] = 10000.
    got = cot_signal(*args, args[0].index + pd.Timedelta(days=3))
    assert got.C_z.iloc[:104].isna().all()
    assert got.C_in.iloc[:104].isna().all()
    assert got.C_z.iloc[104] == pytest.approx((104 - np.arange(104).mean()) / np.arange(104).std(ddof=1))
    past = args[0].iloc[24:180]
    assert got.C_z.iloc[180] == pytest.approx((10000 - past.mean()) / past.std())
    assert got.net.iloc[180] == 10000


@pytest.mark.parametrize('t,r,c', list(product([False, True], repeat=3)))
def test_r5_truth_table(t, r, c):
    assert r5_cash(pd.Series([t]), pd.Series([r]), pd.Series([c])).iloc[0] == (sum([not t, not r, not c]) >= 2)


def test_real_cot_truncation_no_lookahead():
    args = [load(name, refresh=False) for name in
            ('CFTC_ES_LEV_LONG', 'CFTC_ES_LEV_SHORT', 'CFTC_ES_OI', 'CFTC_ES_RELEASE_LAG_DAYS')]
    dates = pd.date_range('2008-07-01', args[0].index.max(), freq='W-TUE')
    full = cot_signal(*args, dates)
    assert full.C_z.notna().all()
    release = args[3].index + pd.to_timedelta(args[3], unit='D')
    for d in pd.to_datetime(['2008-07-01', '2013-10-15', '2019-01-15', '2020-03-17', '2025-06-03']):
        available = args[3].index[release <= d]
        truncated = cot_signal(*(s.reindex(available) for s in args), dates[dates <= d])
        pd.testing.assert_series_equal(full.loc[d], truncated.loc[d])
    used = pd.DatetimeIndex(full.report_date_used)
    assert ((used + pd.to_timedelta(args[3].reindex(used), unit='D')) <= dates).all()


def test_step2_runner_windows_state_and_outputs(tmp_path, monkeypatch):
    from research.us_gauge import run_step2 as runner
    from research.us_gauge.gauge import gauge_frame
    from research.us_gauge.trading import decision_dates, sleeve_weights, net_returns
    from research.us_gauge.evaluate import metrics
    days = pd.bdate_range('1993-01-01', '2022-03-16')
    inputs = pd.DataFrame({'eq_ret': np.random.default_rng(1).normal(.0003, .01, len(days)),
                           'cash_ret': .0001}, index=days)
    dates = decision_dates(days, '1993-01-01')
    gf = gauge_frame(pd.Series(np.where(np.arange(len(dates)) % 17 < 8, 25, 60), dates),
                     pd.Series(np.where(np.arange(len(dates)) % 13 < 6, -1., 1.), dates))
    args = reports(900)
    names = ('CFTC_ES_LEV_LONG', 'CFTC_ES_LEV_SHORT', 'CFTC_ES_OI', 'CFTC_ES_RELEASE_LAG_DAYS')
    monkeypatch.setattr(runner, 'load_inputs', lambda: inputs)
    monkeypatch.setattr(runner, 'build_signals', lambda inputs: gf.copy())
    monkeypatch.setattr(runner, 'load', lambda name, refresh=False: dict(zip(names, args))[name])
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    runner.main()
    table = pd.read_csv(tmp_path / 'out/step2_metrics.csv')
    assert len(table) == 32
    assert set(table.rule) == {'BH', 'R0', 'R1', 'R5'}
    assert set(table.period) == {'WIN', 'WH1', 'WH2', 'P2021'}
    assert list(table.columns) == ['rule', 'version', 'period', 'sharpe', 'cagr', 'maxdd', 'calmar', 'vol', 'switches_per_year', 'share_in_cash']
    decisions = gf.loc['2008-07-01':].join(cot_signal(*args, gf.loc['2008-07-01':].index))
    start = days[days.get_loc(decisions.index[0]) + 2]
    for rule in ('BH', 'R0', 'R1', 'R5'):
        for version, weight in [('all', 0.), ('half', .5)]:
            if rule == 'BH':
                w, r = pd.Series(1., days), inputs.eq_ret
            else:
                frame = decisions if rule == 'R5' else gf.loc['1993-04-01':]
                cash_signal = r5_cash(frame.T_in, frame.R_in, frame.C_in) if rule == 'R5' else frame[f'cash_{rule}']
                w = sleeve_weights(days, frame.index, cash_signal, frame.deep, weight)
                r = net_returns(w, inputs.eq_ret, inputs.cash_ret)
            for period, lo, hi in [('WIN', start, days[-1]), ('WH1', start, '2017-06-30'),
                                   ('WH2', '2017-07-01', days[-1]), ('P2021', '2021-01-01', days[-1])]:
                subset = r.loc[lo:hi]
                row = table.set_index(['rule', 'version', 'period']).loc[(rule, version, period)]
                for key, value in metrics(subset, inputs.cash_ret.reindex(subset.index)).items():
                    assert row[key] == pytest.approx(value)
                assert row.switches_per_year == pytest.approx(w.diff().abs().gt(0).reindex(subset.index).sum() / (len(subset) / 252))
    signals = pd.read_csv(tmp_path / 'out/step2_signals.csv', index_col=0, parse_dates=True)
    assert signals.index[0] == decisions.index[0]
    assert {'net', 'C_z', 'C_in', 'report_date_used', 'state_R5'} <= set(signals.columns)
    report = (tmp_path / 'RESULTS_step2.md').read_text()
    assert all(label in report for label in ['Verdict:', 'Latest R5 reading', 'Deviations', 'T:', 'R:', 'C:'])
