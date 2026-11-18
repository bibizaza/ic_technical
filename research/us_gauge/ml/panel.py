"""Step 1 (spec 2.2): load and clean the 21 indices, read-only, no network.

Closes only. Cleaning fixed by the spec: drop NaN, non-positive closes and duplicate dates; BVSP from
1995-01-02 only; daily moves above 25% are listed, not removed. All series are cut at DATA_END (the
last date common to the 21 index files on 2026-11-18), so the event list and features can be frozen.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
DATA_END = pd.Timestamp('2026-11-11')
MARKETS = ['GSPC', 'SXXP', 'N225', 'TWII', 'GSPTSE', 'HSI', 'KS11', 'AXJO', 'AEX', 'ATX', 'BFX', 'BSESN',
           'BVSP', 'FCHI', 'GDAXI', 'IBEX', 'MXX', 'NZ50', 'OMXS30', 'SSMI', 'STOXX50E']
SEEN = {'GSPC', 'SXXP', 'N225', 'TWII', 'GSPTSE', 'HSI', 'KS11', 'AXJO'}
CLEAN = [m for m in MARKETS if m not in SEEN]          # 13 never-opened markets (spec 6.2)
REGION = {'GSPC': 'north_america', 'GSPTSE': 'north_america',
          **{m: 'europe' for m in ('SXXP', 'AEX', 'ATX', 'BFX', 'FCHI', 'GDAXI', 'IBEX', 'OMXS30', 'SSMI', 'STOXX50E')},
          **{m: 'asia_pac_dev' for m in ('N225', 'AXJO', 'NZ50', 'HSI')},
          **{m: 'emerging' for m in ('TWII', 'KS11', 'BSESN', 'BVSP', 'MXX')}}
BVSP_START = pd.Timestamp('1995-01-02')
MACRO = ['FRED_DEXJPUS', 'FRED_DEXCAUS', 'FRED_DEXKOUS', 'FRED_DEXUSAL', 'FRED_DEXHKUS', 'FRED_DEXTAUS',
         'FRED_DEXSZUS', 'FRED_DEXUSEU', 'FRED_DEXUSUK', 'FRED_DEXINUS', 'FRED_DEXBZUS', 'FRED_DEXMXUS',
         'FRED_DEXSDUS', 'FRED_DEXUSNZ', 'FRED_DEXDNUS', 'FRED_DGS10', 'FRED_DTB3', 'FRED_DBAA', 'FRED_DAAA',
         'FRED_BAA', 'FRED_AAA']


def source(market: str) -> str:
    return 'BBG_SXXP' if market == 'SXXP' else f'{market}.INDX'


def _store_load(name):
    from research.us_gauge.data import load
    return load(name, refresh=False)


def clean(px: pd.Series, market: str, end=DATA_END):
    """Spec 2.2 cleaning. Returns (closes, daily moves above 25% listed, not removed)."""
    px = px.sort_index(kind='mergesort').astype(float)   # stable: the first copy of a duplicate date is kept
    px = px[px.notna() & (px > 0)]
    px = px[~px.index.duplicated(keep='first')]
    if market == 'BVSP':
        px = px[px.index >= BVSP_START]
    if end is not None:
        px = px[px.index <= pd.Timestamp(end)]
    r = px.pct_change(fill_method=None)
    return px.rename(market), r[r.abs() > .25]


def load_panel(markets=MARKETS, end=DATA_END, loader=None):
    """{market: closes}, {market: big moves}. Market store, refresh=False, inside read_only_external()."""
    from research.us_gauge.run_s1 import read_only_external
    loader = loader or _store_load
    P, B = {}, {}
    with read_only_external():
        for m in markets:
            P[m], B[m] = clean(loader(source(m)), m, end)
    return P, B


def load_macro(end=DATA_END, loader=None) -> dict:
    """FRED series used by the features, as stored (no forward fill), cut at DATA_END."""
    from research.us_gauge.run_s1 import read_only_external
    loader = loader or _store_load
    out = {}
    with read_only_external():
        for n in MACRO:
            s = loader(n).sort_index().astype(float)
            s = s[s.notna()]
            out[n] = s[~s.index.duplicated(keep='first')].loc[:end] if end is not None else s
    return out


def big_moves_lines(B: dict) -> list:
    return [f'- {m}: ' + (', '.join(f'{d.date()} {x:+.0%}' for d, x in B[m].items()) or 'none') for m in B]
