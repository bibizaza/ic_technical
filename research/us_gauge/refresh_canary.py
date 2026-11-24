"""Refresh the warning-light data (spec 11e): AGG and EEM total return, EODHD adjusted close.

Adjusted closes change back in time at each dividend, so the full history is fetched again each time.
Usage: python -m research.us_gauge.refresh_canary
"""
import json
import sys
import urllib.request
from datetime import date
import pandas as pd
from research.us_gauge.data import STORE

SYMBOLS = {'AGG.US': 'AGG.US_TR', 'EEM.US': 'EEM.US_TR'}


def main():
    sys.path.insert(0, str(STORE))
    from market_store import _key, save
    for sym, name in SYMBOLS.items():
        u = f'https://eodhd.com/api/eod/{sym}?api_token={_key()}&fmt=json&from=1990-01-01'
        rows = json.load(urllib.request.urlopen(u, timeout=120))
        s = pd.Series({r['date']: r['adjusted_close'] for r in rows if r.get('adjusted_close') is not None}, dtype=float)
        s.index = pd.to_datetime(s.index)
        s = s[s.index < pd.Timestamp(date.today())].sort_index()   # drop today's partial bar
        if len(s) < 4000:
            raise RuntimeError(f'{sym}: only {len(s)} rows returned; store not changed')
        save(name, s)
        print(f'{name}: {s.index[0].date()} → {s.index[-1].date()} ({len(s)} rows)')


if __name__ == '__main__':
    main()
