"""Offline market-store inputs on the S&P trading calendar."""
from pathlib import Path
import sys

import pandas as pd

STORE = Path('/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/data/market_store')


def load(name: str, refresh=False) -> pd.Series:
    """Defer the optional store import so synthetic tests work without Dropbox."""
    if refresh is not False:
        raise ValueError('Research inputs must be loaded offline')
    if str(STORE) not in sys.path:
        sys.path.insert(0, str(STORE))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from market_store import load as store_load
    finally:
        sys.dont_write_bytecode = previous
    return store_load(name, refresh=False)


def cash_daily(dtb3: pd.Series, bil: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """DTB3 through May 30, 2007; stored BIL close returns thereafter."""
    cash = dtb3.sort_index().reindex(index, method='ffill') / 100 / 252
    bil_daily = bil.sort_index().reindex(index, method='ffill').pct_change(fill_method=None)
    return cash.where(index <= pd.Timestamp('2007-05-30'), bil_daily).rename('cash_ret')


def load_inputs() -> pd.DataFrame:
    series = {name: load(name, refresh=False).sort_index() for name in (
        'BBG_SPX', 'BBG_SPX_EPS_FY1', 'BBG_SPX_EPS_FY2',
        'SHILLER_SPX_DIV', 'FRED_DTB3', 'BIL.US', 'FRED_VIXCLS')}
    for name, s in series.items():
        if s.empty:
            raise ValueError(f'Missing offline series: {name}')
    px = series['BBG_SPX']
    idx = px.index
    fy1, fy2 = series['BBG_SPX_EPS_FY1'], series['BBG_SPX_EPS_FY2']
    fy2 = fy2.mask(fy2.eq(fy1.reindex(fy2.index, method='ffill')))
    div = series['SHILLER_SPX_DIV'].reindex(idx, method='ffill')
    return pd.DataFrame({
        'px': px, 'eq_ret': px.pct_change(fill_method=None) + div / px / 252,
        'cash_ret': cash_daily(series['FRED_DTB3'], series['BIL.US'], idx),
        'vix': series['FRED_VIXCLS'].reindex(idx, method='ffill'),
        'fy1': fy1.reindex(idx, method='ffill'),
        'fy2': fy2.reindex(idx, method='ffill'),
    })
