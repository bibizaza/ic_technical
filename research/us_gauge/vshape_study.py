"""Event study of S&P 500 rebounds: V-shaped vs failed rallies (descriptive, not a backtest).

Events: zigzag on daily closes with a 10% reversal. A decline runs from a swing high to a swing low;
the low is "confirmed" on the first close >= 10% above it (the real-time date).
Kept: declines >= 15% since 1955, and 10-15% since 1990.
Class (fixed before looking at features):
  FAILED = the low is undercut within 252 trading days;
  V      = not undercut and >= 50% of the loss regained 63 trading days after the low;
  SLOW   = the rest.
Usage: python vshape_study.py <fred_csv_dir> <out_csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STORE = '/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/data/market_store'
sys.dont_write_bytecode = True
sys.path.insert(0, STORE)
from market_store import load  # noqa: E402


def fred(folder, name):
    s = pd.read_csv(Path(folder) / f'{name}.csv', index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors='coerce').dropna()


def zigzag(p, rev=.10):
    """Return list of (high_date, low_date, confirm_date)."""
    out, hi, lo, state = [], p.index[0], None, 'up'
    for d, x in p.items():
        if state == 'up':
            if x >= p[hi]:
                hi = d
            elif x <= p[hi] * (1 - rev):
                state, lo = 'down', d
        else:
            if x <= p[lo]:
                lo = d
            elif x >= p[lo] * (1 + rev):
                out.append((hi, lo, d))
                state, hi = 'up', d
    return out


def at(s, d, lag=0):
    s = s[:d]
    return np.nan if len(s) <= lag else s.iloc[-1 - lag]


def main(folder, out_csv):
    p = load('BBG_SPX', refresh=False).dropna().sort_index()
    p = p[p.index >= '1953-01-01']
    days = p.index
    vix = load('FRED_VIXCLS', refresh=False).dropna()
    vix3m = load('CBOE_VIX3M', refresh=False).dropna()
    oas = load('BBG_LF98OAS', refresh=False).dropna()['2000-08-15':]
    bf12 = load('BBG_SPX_EPS_BF12', refresh=False).dropna()
    baa10 = fred(folder, 'BAA10Y')
    baa_m = (fred(folder, 'BAA') - fred(folder, 'GS10')).dropna()  # monthly, long history
    dff = fred(folder, 'DFF').rolling(10).mean()
    walcl, m2 = fred(folder, 'WALCL'), fred(folder, 'M2SL')
    usrec, un = fred(folder, 'USREC'), fred(folder, 'UNRATE')
    sahm = (un.rolling(3).mean() - un.rolling(3).mean().rolling(12).min()).shift(1)  # known next month
    sahm.index = sahm.index + pd.offsets.MonthBegin(1)
    ma200 = p.rolling(200).mean()

    rows = []
    for hi, lo, cf in zigzag(p):
        depth = p[lo] / p[hi] - 1
        if lo < pd.Timestamp('1955-01-01') or depth > -.10 or (depth > -.15 and lo < pd.Timestamp('1990-01-01')):
            continue
        i_hi, i_lo, i_cf = days.get_loc(hi), days.get_loc(lo), days.get_loc(cf)
        fwd = lambda i, n: p.iloc[min(i + n, len(p) - 1)] / p.iloc[i] - 1
        regain = lambda n: (p.iloc[min(i_lo + n, len(p) - 1)] - p[lo]) / (p[hi] - p[lo])
        after = p.iloc[i_lo + 1:i_lo + 253]
        undercut = bool((after < p[lo]).any())
        newhigh = p.iloc[i_lo:][p.iloc[i_lo:] >= p[hi]]
        cls = 'FAILED' if undercut else ('V' if regain(63) >= .5 else 'SLOW')
        if i_lo + 63 >= len(p):
            cls = 'open'
        win = slice(hi, days[min(i_cf + 30, len(days) - 1)])

        def peak_lag(s):
            w = s[win]
            if len(w) < 5:
                return np.nan, np.nan
            pk = w.idxmax()
            return (pk - lo).days, at(s, cf) / w[:cf].max()

        vix_lag, vix_ratio = peak_lag(vix)
        oas_lag, oas_ratio = peak_lag(oas)
        baa_lag, baa_ratio = peak_lag(baa10)
        y1 = p[:lo].iloc[-253:]
        rows.append(dict(
            peak=hi.date(), trough=lo.date(), confirm=cf.date(), cls=cls, depth=depth,
            days_fall=i_lo - i_hi, days_to_confirm=i_cf - i_lo,
            dd_from_2y_high=p[lo] / p.iloc[max(0, i_lo - 504):i_lo + 1].max() - 1,
            days_since_1y_high=i_lo - days.get_loc(y1.idxmax()),
            ma200_slope_at_peak=ma200.iloc[i_hi] / ma200.iloc[i_hi - 63] - 1,
            ma200_slope_at_trough=ma200.iloc[i_lo] / ma200.iloc[i_lo - 63] - 1,
            regain_1m=regain(21), regain_3m=regain(63), regain_6m=regain(126),
            days_to_new_high=(days.get_loc(newhigh.index[0]) - i_lo) if len(newhigh) else np.nan,
            fwd3m_from_confirm=fwd(i_cf, 63), fwd6m_from_confirm=fwd(i_cf, 126),
            min_after_confirm_6m=p.iloc[i_cf:i_cf + 127].min() / p[cf] - 1,
            ff_at_peak=at(dff, hi), ff_chg_peak_to_confirm=at(dff, cf) - at(dff, hi),
            ff_chg_63d_before_peak=at(dff, hi) - dff[:hi].iloc[-64],
            ff_chg_252d_before_peak=at(dff, hi) - dff[:hi].iloc[-253],
            walcl_13w=(at(walcl, cf) / at(walcl, cf, 13) - 1) if cf > pd.Timestamp('2003-06-01') else np.nan,
            m2_3m_ann=((at(m2, cf, 1) / at(m2, cf, 4)) ** 4 - 1) if cf > pd.Timestamp('1960-01-01') else np.nan,
            m2_12m=(at(m2, cf, 1) / at(m2, cf, 13) - 1) if cf > pd.Timestamp('1961-01-01') else np.nan,
            baa_m_at_trough=at(baa_m, lo, 1), baa_m_chg_6m=at(baa_m, lo, 1) - at(baa_m, lo, 7),
            baa10_lag=baa_lag, baa10_ratio=baa_ratio,
            oas_at_confirm=at(oas, cf) if cf > oas.index[0] else np.nan, oas_lag=oas_lag, oas_ratio=oas_ratio,
            vix_max=vix[win].max() if len(vix[win]) else np.nan, vix_lag=vix_lag,
            vix_at_confirm=at(vix, cf) if cf > vix.index[0] else np.nan, vix_ratio=vix_ratio,
            vix_over_3m=(at(vix, cf) / at(vix3m, cf)) if cf > vix3m.index[0] else np.nan,
            eps_chg_peak_to_confirm=(at(bf12, cf) / at(bf12, hi) - 1) if hi > bf12.index[0] else np.nan,
            eps_chg_63d=(at(bf12, cf) / at(bf12, cf, 63) - 1) if cf > pd.Timestamp('1990-06-01') else np.nan,
            fpe_trough=(p[lo] / at(bf12, lo)) if lo > bf12.index[0] else np.nan,
            rec_at_trough=at(usrec, lo), rec_at_peak=at(usrec, hi),
            rec_within_12m_of_peak=float(usrec[hi:hi + pd.DateOffset(months=12)].max()),
            sahm_at_confirm=at(sahm, cf),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 60, 'display.float_format', '{:.2f}'.format)
    print(df.to_string())
    return df


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
