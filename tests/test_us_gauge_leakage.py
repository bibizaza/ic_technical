"""End-to-end signal invariance to future data removal."""
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.us_gauge.data import STORE, load_inputs
from research.us_gauge.run_step1 import build_signals


@pytest.mark.skipif(not STORE.exists(), reason='Dropbox store missing')
def test_signals_truncation_invariant():
    inputs = load_inputs()
    full = build_signals(inputs)
    for d in pd.to_datetime(['1995-06-06', '2002-06-04', '2009-06-02', '2017-06-06', '2025-06-03']):
        truncated = build_signals(inputs.loc[:d])
        pd.testing.assert_series_equal(full.loc[d], truncated.loc[d])


def test_runner_outputs_and_halves(tmp_path, monkeypatch):
    import numpy as np
    from research.us_gauge import run_step1 as runner
    from research.us_gauge.gauge import gauge_frame
    from research.us_gauge.trading import decision_dates, sleeve_weights, net_returns
    from research.us_gauge.evaluate import metrics
    idx = pd.bdate_range('1993-01-01', '2011-03-16')
    inputs = pd.DataFrame({'px': 100., 'eq_ret': np.random.default_rng(1).normal(.0003, .01, len(idx)),
                           'cash_ret': .0001, 'fy1': 100., 'fy2': 110.}, index=idx)
    def signals(inputs, rev_window=63, z_threshold=0.):
        dates = decision_dates(inputs.index, '1990-01-01')
        dmas = pd.Series(np.where(np.arange(len(dates)) % 17 < 8, 25, 60), dates)
        return gauge_frame(dmas, pd.Series(.1, dates))
    monkeypatch.setattr(runner, 'load_inputs', lambda: inputs)
    monkeypatch.setattr(runner, 'build_signals', signals)
    monkeypatch.setattr(runner, 'HERE', tmp_path)
    runner.main()
    out = tmp_path / 'out'
    assert {p.name for p in out.glob('*.csv')} == {
        'step1_metrics.csv', 'step1_signals.csv', 'step1_switch_risk.csv',
        'step1_drawdowns.csv', 'step1_sensitivity.csv'}
    table = pd.read_csv(out / 'step1_metrics.csv')
    assert len(table) == 15
    assert set(table.period) == {'FULL', 'H1', 'H2'}
    assert 'Verdict:' in (tmp_path / 'RESULTS.md').read_text()
    gf = signals(inputs).loc['1993-04-01':]
    w = sleeve_weights(idx, gf.index, gf.cash_R0, gf.deep, 0.)
    r = net_returns(w, inputs.eq_ret, inputs.cash_ret).dropna()
    expected = metrics(r.loc['2010-01-01':], inputs.cash_ret.reindex(r.loc['2010-01-01':].index))
    row = table[(table.rule == 'R0') & (table.version == 'all') & (table.period == 'H2')].iloc[0]
    assert row.sharpe == pytest.approx(expected['sharpe'])
    sensitivity = pd.read_csv(out / 'step1_sensitivity.csv')
    assert sensitivity.information_only.all()
