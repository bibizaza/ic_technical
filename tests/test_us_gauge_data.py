"""Calendar, cleaning and offline input contract."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.us_gauge import data

STORE = Path('/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/data/market_store')


def test_cash_daily_switches_to_bil():
    idx = pd.bdate_range('2007-05-25', '2007-06-06')
    bil_idx = idx[idx >= '2007-05-30']
    bil = pd.Series(100 * 1.0001 ** np.arange(len(bil_idx)), bil_idx)
    got = data.cash_daily(pd.Series(5., idx), bil, idx)
    assert got.loc[:'2007-05-30'].to_numpy() == pytest.approx(0.05 / 252)
    pd.testing.assert_series_equal(got.loc['2007-05-31':], bil.pct_change().dropna(), check_names=False)
    assert got.notna().all()


def test_fy2_copy_is_blanked(monkeypatch):
    idx = pd.bdate_range('2007-05-25', periods=8)
    fy1 = pd.Series(100., idx)
    fy2 = pd.Series([110, 100, 100, 100, 110, 110, 110, 110], idx)
    series = {'BBG_SPX': pd.Series(1000., idx), 'BBG_SPX_EPS_FY1': fy1,
              'BBG_SPX_EPS_FY2': fy2, 'FRED_DTB3': pd.Series(5., idx),
              'BIL.US': pd.Series(100., idx), 'SHILLER_SPX_DIV': pd.Series(20., idx),
              'FRED_VIXCLS': pd.Series([20., 30.], [idx[1], idx[4] - pd.Timedelta(hours=12)])}
    def offline(name, refresh):
        assert refresh is False
        return series[name]
    monkeypatch.setattr(data, 'load', offline)
    got = data.load_inputs()
    assert got.fy2.iloc[1:4].isna().all()
    assert got.fy2.iloc[4] == 110
    assert pd.isna(got.vix.iloc[0])
    assert got.vix.iloc[1:4].eq(20.).all()
    assert got.vix.iloc[4:].eq(30.).all()


@pytest.mark.skipif(not STORE.exists(), reason='Dropbox store missing')
def test_load_inputs_real():
    got = data.load_inputs()
    assert got.index[0] == pd.Timestamp('1950-01-03')
    assert got.loc['1954-01-04':, ['eq_ret', 'cash_ret']].notna().all().all()
    price_ret = got.px.pct_change()
    assert (got.eq_ret.loc['2026-04-01':] > price_ret.loc['2026-04-01':]).all()


def test_module_import_without_dropbox(monkeypatch):
    """Missing optional data must not prevent synthetic tests from collecting."""
    import builtins
    import importlib.util
    original = builtins.__import__
    def without_store(name, *args, **kwargs):
        if name == 'market_store':
            raise ModuleNotFoundError('market_store unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_store)
    spec = importlib.util.spec_from_file_location('offline_data_probe', data.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.cash_daily)


def test_store_import_does_not_write_bytecode(tmp_path, monkeypatch):
    """Even Python's import cache must not write into the external store."""
    (tmp_path / 'market_store.py').write_text('def load(name, refresh):\n    return 42\n')
    monkeypatch.setattr(data, 'STORE', tmp_path)
    monkeypatch.delitem(sys.modules, 'market_store', raising=False)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setattr(sys, 'dont_write_bytecode', False)
    assert data.load('sample', refresh=False) == 42
    assert not (tmp_path / '__pycache__').exists()
