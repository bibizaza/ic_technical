"""Tests for the HAR 1-week trading range (technical_analysis/har_range.py + spx.get_trading_range)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from technical_analysis.har_range import har_trading_range, har_week_variance  # noqa: E402


def _simulate(n_days, sigma, freq="B", seed=0):
    """Random-walk closes with intraday highs/lows (12 steps a day), daily vol ``sigma``."""
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n_days, 12)) * sigma / np.sqrt(12)
    path = np.log(100) + np.cumsum(steps.ravel())
    path = path.reshape(n_days, 12)
    opens = np.r_[np.log(100), path[:-1, -1]]
    close = np.exp(path[:, -1])
    high = np.exp(np.maximum(path.max(1), opens))
    low = np.exp(np.minimum(path.min(1), opens))
    idx = pd.date_range("2018-01-01", periods=n_days, freq=freq)
    return pd.Series(close, idx), pd.Series(high, idx), pd.Series(low, idx)


@pytest.mark.parametrize("freq,rows_per_week", [("B", 5), ("D", 7)])
def test_weekly_vol_has_the_right_size(freq, rows_per_week):
    """Known daily vol 1%: the 1-week forecast should be about 1% x sqrt(trading rows in 7 days)."""
    price, high, low = _simulate(1600, 0.01, freq)
    var = har_week_variance(price, high, low)
    assert var is not None
    assert np.sqrt(var) == pytest.approx(0.01 * np.sqrt(rows_per_week), rel=0.25)


def test_works_without_highs_and_lows():
    price, _, _ = _simulate(1600, 0.01)
    assert np.sqrt(har_week_variance(price)) == pytest.approx(0.01 * np.sqrt(5), rel=0.25)


def test_range_is_symmetric_around_current_price():
    price, high, low = _simulate(1600, 0.02)
    lo, hi = har_trading_range(price, high, low, current_price=123.0)
    assert (hi - 123.0) == pytest.approx(123.0 - lo)
    assert lo < 123.0 < hi


def test_not_enough_history_returns_none():
    price, high, low = _simulate(200, 0.01)
    assert har_week_variance(price, high, low) is None
    assert har_trading_range(price, high, low) is None


def test_only_past_data_is_used():
    """Changing prices after the as-of date must not change the forecast made at that date."""
    price, high, low = _simulate(1600, 0.01)
    d = price.index[1400]
    a = har_week_variance(price.loc[:d], high, low)
    shocked, hi2, lo2 = price.copy(), high.copy(), low.copy()
    future = shocked.index > d
    shocked[future] *= 3.0
    hi2[future] *= 5.0  # future highs/lows (passed in full) must be ignored too
    lo2[future] *= 0.2
    b = har_week_variance(shocked.loc[:d], hi2, lo2)
    assert a == pytest.approx(b, rel=1e-12)


# --------------------------------------------------------------------------- deck integration
@pytest.fixture(scope="module")
def spx():
    """The deck chart module (needs the project environment: plotly, streamlit, playwright...)."""
    return pytest.importorskip("technical_analysis.equity.spx")


def _write_temp_excel(path, ticker, price, high, low):
    """Same layout as data_loader.create_temp_excel_from_csv: header row, then <t>, <t>.1 (low), <t>.2 (high)."""
    body = pd.DataFrame({"Date": price.index.strftime("%Y-%m-%d"), ticker: price.values,
                         f"{ticker}.1": low.values, f"{ticker}.2": high.values})
    header = pd.DataFrame([{"Date": "DATES", ticker: "#price", f"{ticker}.1": "#price", f"{ticker}.2": "#price"}])
    pd.concat([header, body], ignore_index=True).to_excel(path, sheet_name="data_prices", index=False)


def test_deck_uses_har_range(tmp_path, spx):
    price, high, low = _simulate(1600, 0.015)
    xl = tmp_path / "prices.xlsx"
    _write_temp_excel(xl, "TEST Index", price, high, low)
    p = float(price.iloc[-1])
    df = pd.DataFrame({"Date": price.index, "Price": price.values})
    lo, hi = spx.get_trading_range("TEST Index", p, df, excel_path=str(xl), days_forward=21)
    expected = har_trading_range(price, high, low, current_price=p)
    assert (lo, hi) == pytest.approx(expected)


def test_deck_falls_back_with_short_history(tmp_path, spx):
    price, high, low = _simulate(150, 0.015)
    xl = tmp_path / "prices.xlsx"
    _write_temp_excel(xl, "TEST Index", price, high, low)
    p = float(price.iloc[-1])
    df = pd.DataFrame({"Date": price.index, "Price": price.values})
    got = spx.get_trading_range("TEST Index", p, df, excel_path=str(xl), days_forward=21)
    recent = df["Price"].tail(21)
    assert got == pytest.approx(spx._apply_range_guards(recent.min(), recent.max(), p))


def test_deck_falls_back_for_unknown_ticker(tmp_path, spx):
    price, high, low = _simulate(1600, 0.015)
    xl = tmp_path / "prices.xlsx"
    _write_temp_excel(xl, "TEST Index", price, high, low)
    p = float(price.iloc[-1])
    df = pd.DataFrame({"Date": price.index, "Price": price.values})
    got = spx.get_trading_range("OTHER Index", p, df, excel_path=str(xl), days_forward=21)
    recent = df["Price"].tail(21)
    assert got == pytest.approx(spx._apply_range_guards(recent.min(), recent.max(), p))
