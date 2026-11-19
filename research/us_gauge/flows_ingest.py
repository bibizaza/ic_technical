"""Store free US equity flow data in the market store (no Bloomberg).

1. Federal Reserve Z.1 Financial Accounts, table F.223 / L.223 (corporate equities): who buys and sells US
   stocks each quarter, from 1946. Flows are seasonally adjusted annual rates (USD million); levels are market
   values (USD million). Source zip: federalreserve.gov/releases/z1/current/z1_csv_files.zip
2. US Treasury TIC S: foreigners' gross purchases / sales of US corporate stocks, monthly, 1977-01 → 2023-01
   (form S stopped; later months are on form SLT, not stored here). Source: ticdata.treasury.gov s1_globl.txt

Usage: python -m research.us_gauge.flows_ingest <z1 csv folder> <s1_globl.txt>
"""
import csv
import sys
from pathlib import Path
import pandas as pd

STORE = Path.home() / 'Library/CloudStorage/Dropbox/Tools_In_Construction/data/market_store'

Z1_FLOWS = {   # F.223 transactions, SAAR, USD million
    'FA893064105': 'All sectors, net purchases (= net issuance)',
    'FA103164105': 'Nonfinancial corporations, net issuance (negative = net buybacks)',
    'FA153064105': 'Households and nonprofits, net purchases',
    'FA573064105': 'Private pension funds, net purchases',
    'FA343064105': 'Federal government pension funds, net purchases',
    'FA223064145': 'State and local DB pension funds, net purchases',
    'FA543064105': 'Life insurance, net purchases',
    'FA653064100': 'Mutual funds, net purchases',
    'FA553064103': 'Closed-end funds, net purchases',
    'FA563064100': 'ETFs, net purchases',
    'FA623064105': 'Hedge funds (domestic), net purchases net of short sales',
    'FA663064105': 'Broker-dealers, net purchases',
    'FA263064105': 'Rest of the world, net purchases of US equities',
}
Z1_LEVELS = {  # L.223 market value, USD million
    'LM893064105': 'All sectors, corporate equities held',
    'LM153064105': 'Households and nonprofits, held',
    'LM573064105': 'Private pension funds, held',
    'LM223064145': 'State and local DB pension funds, held',
    'LM653064100': 'Mutual funds, held',
    'LM563064100': 'ETFs, held',
    'LM623064105': 'Hedge funds (domestic), held',
    'LM263064105': 'Rest of the world, US equities held',
    'LM883164115': 'All domestic sectors, public corporate equities (liability)',
}


def quarter_end(q: str) -> pd.Timestamp:
    y, n = q.split(':Q')
    return pd.Period(f'{y}Q{n}', freq='Q').end_time.normalize()


def z1_series(folder: Path, table: str, code: str) -> pd.Series:
    df = pd.read_csv(folder / f'{table}.csv', na_values=['ND'])
    s = pd.to_numeric(df[f'{code}.Q'], errors='coerce')
    s.index = df['date'].map(quarter_end)
    return s.dropna()


def tic_us_stocks(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, newline='', encoding='latin-1') as f:
        for r in csv.reader(f, delimiter='\t'):
            if len(r) > 12 and r[0].strip() == 'Grand Total':
                num = lambda x: float(x.replace(',', '').strip() or 'nan')
                rows.append((pd.Period(r[2].strip(), freq='M').end_time.normalize(), num(r[6]), num(r[12])))
    t = pd.DataFrame(rows, columns=['date', 'purch', 'sales']).set_index('date').sort_index()
    t['net'] = t.purch - t.sales
    return t


def write(name: str, s: pd.Series):
    s.rename(name).rename_axis('date').to_frame().to_csv(STORE / f'{name}.csv')
    print(f'{name}: {s.index[0].date()} → {s.index[-1].date()} ({len(s)} rows)')


def main(z1_folder: str, tic_file: str):
    folder = Path(z1_folder)
    for code in Z1_FLOWS:
        write(f'Z1_{code}', z1_series(folder, 'F51_1_t', code))
    for code in Z1_LEVELS:
        write(f'Z1_{code}', z1_series(folder, 'F51_1_s', code))
    t = tic_us_stocks(Path(tic_file))
    write('TIC_US_STOCKS_PURCH', t.purch)
    write('TIC_US_STOCKS_SALES', t.sales)
    write('TIC_US_STOCKS_NET', t.net)


if __name__ == '__main__':
    main(*sys.argv[1:3])
