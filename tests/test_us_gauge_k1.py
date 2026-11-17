"""K1 preregistered mechanics, written before implementation/results."""
import numpy as np
import pandas as pd
import pytest
from research.us_gauge.run_k1 import combine, v2_weights, reproduce_c5, etf_daily_returns
from research.us_gauge import run_vol


def fixture():
    idx = pd.bdate_range('2000-01-03', '2000-04-10')
    ret = pd.DataFrame(0., index=idx, columns=['SPY', 'XLP', 'XLI', 'XLV', 'XLK'])
    picks = {pd.Period('2000-01'): ['XLP', 'XLI', 'XLV'],
             pd.Period('2000-02'): ['XLK', 'XLI', 'XLV'],
             pd.Period('2000-03'): ['XLK', 'XLI', 'XLV'],
             pd.Period('2000-04'): ['XLP', 'XLI', 'XLV']}
    return ret, picks, pd.Series(.5, idx), pd.Series(.0001, idx)


def test_first_trading_close_and_drift_reset():
    ret, picks, w, cash = fixture()
    ret.loc['2000-01-04', 'XLP'] = .1
    ret.loc['2000-02-01', 'XLK'] = .8
    got, held = combine(ret, picks, w, cash)
    assert held.loc['2000-01-04', 'XLP'] == .1
    assert held.loc['2000-01-05', 'XLP'] == pytest.approx(.11 / 1.01)
    assert held.loc['2000-02-01', 'XLK'] == 0
    assert held.loc['2000-02-01', 'XLP'] == pytest.approx(.11 / 1.01)
    assert held.loc['2000-02-02', 'XLK'] == .1
    assert held.loc['2000-02-02', 'SPY'] == .7
    assert got.loc['2000-02-01', 'sleeve_return'] == 0
    assert held.loc['2000-03-02', ['SPY', 'XLK', 'XLI', 'XLV']].tolist() == [.7, .1, .1, .1]


def test_cost_formula_and_identical_v2_spy_weight():
    ret, picks, w, cash = fixture()
    ret.loc[:, 'SPY'] = .001
    w.loc['2000-02-02':] = .8
    got, held = combine(ret, picks, w, cash)
    date = pd.Timestamp('2000-02-02')
    before = held.loc['2000-02-01'] * (1 + ret.loc['2000-02-01'])
    before /= before.sum()
    turnover = (held.loc[date] - before).abs().sum()
    assert got.loc[date, 'reset_turnover'] == pytest.approx(turnover)
    assert got.loc[date, 'cost'] == pytest.approx(.001 * .3 + .001 * .8 * turnover)
    expected = run_vol.net_returns(w, ret.SPY, cash)
    pd.testing.assert_series_equal(got['V2-SPY'], expected.rename('V2-SPY'))
    gross = w * got.sleeve_return + (1 - w) * cash
    pd.testing.assert_series_equal(got.K1, (gross - got.cost).rename('K1'))


@pytest.mark.parametrize('cut', ['2000-02-01', '2000-02-02', '2000-03-15'])
def test_combined_return_truncation(cut):
    ret, picks, w, cash = fixture()
    ret[:] = np.random.default_rng(7).normal(0, .01, ret.shape)
    full, _ = combine(ret, picks, w, cash)
    available = {m: p for m, p in picks.items() if m.start_time <= pd.Timestamp(cut)}
    short, _ = combine(ret.loc[:cut], available, w.loc[:cut], cash.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut], short)


def test_real_v2_and_combined_truncation():
    # Rebuild the R1 signals and volatility weights from truncated offline inputs.
    from research.us_gauge.run_s1 import read_only_external
    from research.us_gauge.data import load_inputs, load
    from research.us_gauge import run_s2
    with read_only_external():
        inputs = load_inputs()
        reproduced, panel, pred, _ = reproduce_c5()
        assert run_s2.reproduces(reproduced)
        monthly = run_s2.load_etfs()
        _, selected = run_s2.tilt_returns(pred, monthly, panel.r_rf)
        levels = pd.DataFrame({ticker: load(ticker + '.US_TR', refresh=False)
                               for ticker in monthly.columns})
    w, _, _ = v2_weights(inputs)
    gf = run_vol.build_signals(inputs)
    baseline, _ = run_vol._simulate(inputs, gf, 'R1', 0.)
    dates = gf.loc['1993-04-01':].index
    usable = dates[inputs.index.get_indexer(dates) + 2 < len(inputs)]
    expected = run_vol.daily_weights(inputs.index, usable, run_vol.target_weight(inputs.eq_ret),
                                     baseline.shift(-2).reindex(dates))
    pd.testing.assert_series_equal(w, expected)
    idx = inputs.loc['1999-01-01':'2026-08-31'].index
    price_index = inputs.index[inputs.index.get_loc(idx[0]) - 1:inputs.index.get_loc(idx[-1]) + 1]
    ret = etf_daily_returns(levels, price_index).reindex(idx)
    picks = {d.to_period('M'): list(row.index[row.eq(1)]) for d, row in selected.iterrows()}
    full, _ = combine(ret, picks, w, inputs.cash_ret)
    for cut in ['2002-06-04', '2013-01-02', '2025-06-03']:
        short_w, _, _ = v2_weights(inputs.loc[:cut])
        short_ret = etf_daily_returns(levels.loc[:cut], price_index[price_index <= pd.Timestamp(cut)]).reindex(idx[idx <= pd.Timestamp(cut)])
        short, _ = combine(short_ret, {m: p for m, p in picks.items() if m.start_time <= pd.Timestamp(cut)},
                           short_w, inputs.cash_ret.loc[:cut])
        pd.testing.assert_frame_equal(full.loc[:cut], short)


def test_authorized_one_day_gap_books_move_on_next_observation():
    from research.us_gauge.run_k1 import etf_daily_returns
    idx = pd.bdate_range('1999-01-25', periods=3)
    levels = pd.DataFrame({'XLI': [100., np.nan, 110.], 'SPY': [100., 101., 102.]}, index=idx)
    got = etf_daily_returns(levels, idx)
    assert got.loc['1999-01-26', 'XLI'] == 0.
    assert got.loc['1999-01-27', 'XLI'] == pytest.approx(.1)
    pd.testing.assert_frame_equal(got.iloc[:2], etf_daily_returns(levels.iloc[:2], idx[:2]))
    # An absent row is handled identically to an explicit missing value.
    pd.testing.assert_series_equal(got.XLI, etf_daily_returns(levels[['XLI']].dropna(), idx).XLI)


def test_unregistered_etf_gap_still_stops():
    from research.us_gauge.run_k1 import etf_daily_returns
    idx = pd.bdate_range('1999-01-25', periods=3)
    with pytest.raises(ValueError, match='Unregistered ETF gap'):
        etf_daily_returns(pd.DataFrame({'SPY': [100., np.nan, 110.]}, index=idx), idx)


@pytest.mark.parametrize('missing', ['cash', 'weight', 'held_return'])
def test_other_missing_inputs_still_stop(missing):
    ret, picks, w, cash = fixture()
    if missing == 'cash':
        cash.iloc[3] = np.nan
    elif missing == 'weight':
        w.iloc[3] = np.nan
    else:
        ret.iloc[3, ret.columns.get_loc('XLP')] = np.nan
    with pytest.raises(ValueError, match='Missing required daily input'):
        combine(ret, picks, w, cash)
