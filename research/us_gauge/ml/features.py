"""Step 3 (spec section 4): the 15 features, all known at the trigger close t.

Timing. Local closes up to t (row i_t of the market's own series). Every other series (other markets,
FX, US rates, credit) uses the last value dated on or before t - 1 business day (`cutoff`). Monthly
Moody's yields (before daily data) use a month's value only from the first business day of the next
month. No price after t, and no other series after the cutoff, is read: see tests/test_us_gauge_ml_leakage.py.
"""
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, MonthBegin

from research.us_gauge.ml.panel import OUT, MARKETS, REGION, load_panel, load_macro

FEATURES = ['depth', 'speed_hl', 'days_le', 'crash_share', 'vol_rel', 'runup_2y', 'attempt', 'glob_breadth',
            'rel_depth', 'fx_shock', 'vol_level', 'region_europe', 'region_asia_pac_dev', 'region_emerging',
            'd_us10y', 'curve_min', 'd_credit']
CORE15 = FEATURES[:11] + ['region'] + FEATURES[14:]       # the 15 features of the spec (region = 3 dummies)
FX_UNITS_PER_USD = {'N225': 'FRED_DEXJPUS', 'TWII': 'FRED_DEXTAUS', 'GSPTSE': 'FRED_DEXCAUS', 'HSI': 'FRED_DEXHKUS',
                    'KS11': 'FRED_DEXKOUS', 'BSESN': 'FRED_DEXINUS', 'BVSP': 'FRED_DEXBZUS', 'MXX': 'FRED_DEXMXUS',
                    'OMXS30': 'FRED_DEXSDUS', 'SSMI': 'FRED_DEXSZUS'}
FX_USD_PER_UNIT = {'AXJO': 'FRED_DEXUSAL', 'NZ50': 'FRED_DEXUSNZ'}
EURO_DKK = {'SXXP', 'STOXX50E', 'GDAXI', 'FCHI', 'AEX', 'BFX', 'ATX'}      # krone proxy before 1999
EURO_START = pd.Timestamp('1999-01-01')
US_BASKET = [('FRED_DEXJPUS', False), ('FRED_DEXSZUS', False), ('FRED_DEXUSUK', True), ('FRED_DEXCAUS', False)]


def sigma21(px: pd.Series) -> pd.Series:
    """As run_vol.sigma: std of the last 21 daily price returns through the row, ddof=1, x sqrt(252)."""
    return px.pct_change(fill_method=None).rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252)


def drawdown252(px: pd.Series) -> pd.Series:
    """Close / highest close of the 252 closes ending the row - 1 (NaN before 252 closes)."""
    return px / px.rolling(252, min_periods=252).max() - 1


def asof(s: pd.Series, d):
    """Last value dated on or before d (NaN if none)."""
    i = s.index.searchsorted(pd.Timestamp(d), side='right') - 1
    return s.iloc[i] if i >= 0 else np.nan


def first_bday(month_start: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(month_start)
    return d if d.weekday() < 5 else d + BDay(1)


class Context:
    """Precomputed, row-causal series. Build it from truncated data and the features must not change."""

    def __init__(self, P: dict, macro: dict):
        self.P = P
        self.sig = {m: sigma21(p) for m, p in P.items()}
        self.dd = {m: drawdown252(p) for m, p in P.items()}
        self.macro = macro
        d = pd.concat([macro['FRED_DBAA'], macro['FRED_DAAA']], axis=1, join='inner').dropna()
        self.credit_daily = (d.iloc[:, 0] - d.iloc[:, 1]).sort_index()
        mo = pd.concat([macro['FRED_BAA'], macro['FRED_AAA']], axis=1, join='inner').dropna()
        mo = mo.iloc[:, 0] - mo.iloc[:, 1]
        mo.index = pd.DatetimeIndex(mo.index).to_period('M').to_timestamp()       # month start
        self.credit_monthly = mo.sort_index()
        c = pd.concat([macro['FRED_DGS10'], macro['FRED_DTB3']], axis=1, join='inner').dropna()
        self.curve = (c.iloc[:, 0] - c.iloc[:, 1]).sort_index()

    # ---- helpers on the other markets (t - 1 rule) ----
    def others_dd(self, market, cutoff):
        out = {}
        for o, dd in self.dd.items():
            if o == market:
                continue
            v = asof(dd, cutoff)
            if not np.isnan(v):
                out[o] = v
        return out

    def fx_series(self, market, h, cutoff):
        """(series of local currency units per USD, ok) chosen for this event, or None (missing)."""
        if market in FX_UNITS_PER_USD:
            return self.macro[FX_UNITS_PER_USD[market]]
        if market in FX_USD_PER_UNIT:
            return 1. / self.macro[FX_USD_PER_UNIT[market]]
        if market in EURO_DKK or market == 'IBEX':
            if cutoff < EURO_START:
                return self.macro['FRED_DEXDNUS'] if market in EURO_DKK else None
            if h >= EURO_START:
                return 1. / self.macro['FRED_DEXUSEU']
            return 'straddle'
        raise KeyError(market)

    def fx_shock(self, market, h, cutoff):
        def chg(s):
            if s is None or len(s) == 0 or s.index[0] > h:
                return np.nan
            return float(np.log(asof(s, cutoff) / asof(s, h)))
        if market == 'GSPC':
            ch = [chg(1. / self.macro[n] if inv else self.macro[n]) for n, inv in US_BASKET]
            return -float(np.mean(ch)) if not np.isnan(ch).any() else np.nan
        s = self.fx_series(market, h, cutoff)
        if isinstance(s, str):           # euro member, h before 1999 and t-1 after: krone to 1998-12-31, then euro
            dkk, eur = self.macro['FRED_DEXDNUS'], 1. / self.macro['FRED_DEXUSEU']
            if market == 'IBEX' or len(dkk) == 0 or dkk.index[0] > h or len(eur) == 0:
                return np.nan
            last98 = pd.Timestamp('1998-12-31')
            return float(np.log(asof(dkk, last98) / asof(dkk, h)) + np.log(asof(eur, cutoff) / eur.iloc[0]))
        return chg(s)

    def d_credit(self, h, cutoff):
        """Baa - Aaa spread: value of the month before h's month -> value usable at t - 1.
        Daily DBAA - DAAA when that month is fully covered by daily data (the month's mean of daily values,
        last daily value at the cutoff); before, monthly BAA - AAA (month M usable from the first business
        day of M + 1) at both ends."""
        m0 = (pd.Timestamp(h).to_period('M') - 1).to_timestamp()
        cd = self.credit_daily
        if len(cd) and cd.index[0] <= m0:
            start = cd.loc[m0:m0 + MonthBegin(1) - pd.Timedelta(days=1)]
            return float(asof(cd, cutoff) - start.mean()) if len(start) else np.nan
        cm = self.credit_monthly
        usable = [m for m in cm.index if first_bday(m + MonthBegin(1)) <= cutoff]
        if m0 not in cm.index or not usable:
            return np.nan
        return float(cm.loc[usable[-1]] - cm.loc[m0])


def event_features(market, px, i_h, i_l, i_t, ctx: Context, prior_entries) -> dict:
    """The 15 features of one trigger. `prior_entries`: entry dates of earlier events in this market."""
    p = px.to_numpy(dtype=float)
    idx = px.index
    h, t = idx[i_h], idx[i_t]
    cutoff = t - BDay(1)
    H, L = p[i_h], p[i_l]
    f = {}
    f['depth'] = L / H - 1
    f['speed_hl'] = i_l - i_h
    f['days_le'] = i_t - i_l
    lr = np.diff(np.log(p[i_h:i_l + 1]))                   # log returns on closes h+1 .. l
    f['crash_share'] = lr.min() / np.log(L / H)
    sig = sigma21(px.iloc[:i_t + 1]) if ctx is None else ctx.sig[market]
    sig_win = sig.iloc[max(0, i_h - 1259):i_h + 1].dropna()
    vol_level = float(sig_win.median()) if len(sig_win) >= 252 else np.nan
    f['vol_rel'] = sig.iloc[i_t] / vol_level
    f['runup_2y'] = float(np.log(H / p[i_h - 504])) if i_h >= 504 else np.nan
    lo = idx[max(0, i_t - 504)]
    f['attempt'] = min(3, sum(1 for d in prior_entries if lo <= pd.Timestamp(d) < t))
    oth = ctx.others_dd(market, cutoff)
    f['glob_breadth'] = float(np.mean([v <= -.15 for v in oth.values()])) if len(oth) >= 3 else np.nan
    own = asof(ctx.dd[market].iloc[:i_t + 1], cutoff) if market in ctx.dd else asof(drawdown252(px.iloc[:i_t + 1]), cutoff)
    f['rel_depth'] = float(own - np.median(list(oth.values()))) if len(oth) >= 1 and not np.isnan(own) else np.nan
    f['fx_shock'] = ctx.fx_shock(market, h, cutoff)
    f['vol_level'] = vol_level
    reg = REGION[market]
    for r in ('europe', 'asia_pac_dev', 'emerging'):
        f[f'region_{r}'] = float(reg == r)
    g10 = ctx.macro['FRED_DGS10']
    f['d_us10y'] = float(asof(g10, cutoff) - asof(g10, h)) if len(g10) and g10.index[0] <= h else np.nan
    cv = ctx.curve.loc[:cutoff]
    f['curve_min'] = float(cv.iloc[-252:].min()) if len(cv) >= 252 else np.nan
    f['d_credit'] = ctx.d_credit(h, cutoff)
    return f


def features_for(rows: pd.DataFrame, P: dict, ctx: Context, entries: dict, px_override=None) -> pd.DataFrame:
    """rows: market, i_h, i_l, i_t (+ an id column). entries: {market: all event entry dates of that market}."""
    out = []
    for r in rows.itertuples():
        px = P[r.market] if px_override is None else px_override
        t = px.index[r.i_t]
        prior = [d for d in entries.get(r.market, []) if pd.Timestamp(d) < t]
        out.append(event_features(r.market, px, int(r.i_h), int(r.i_l), int(r.i_t), ctx, prior))
    return pd.DataFrame(out, columns=FEATURES, index=rows.index)


def build():
    P, _ = load_panel()
    macro = load_macro()
    ctx = Context(P, macro)
    E = pd.read_csv(OUT / 'events.csv', parse_dates=['h', 'l', 't'])
    PE = pd.read_csv(OUT / 'product_episodes.csv', parse_dates=['h', 'l', 't'])
    for D in (E, PE):                                   # the stored indices must point at the stored dates
        for r in D.itertuples():
            ix = P[r.market].index
            assert ix[r.i_h] == r.h and ix[r.i_l] == r.l and ix[r.i_t] == r.t, r
    entries = {m: list(E.loc[E.market == m, 't']) for m in MARKETS}
    F = features_for(E, P, ctx, entries)
    F.insert(0, 'event_id', E.event_id)
    G = features_for(PE, P, ctx, entries)
    G.insert(0, 'episode_id', PE.episode_id)
    return E, F, PE, G


def missing_report(E, F) -> list:
    st = E.study.to_numpy()
    X = F.loc[st, FEATURES]
    L = [f'Study events: {int(st.sum())}. Missing share by feature: ' +
         ', '.join(f'{c} {X[c].isna().mean():.1%}' for c in FEATURES if X[c].isna().any()) + '.']
    nmiss = X.isna().sum(axis=1)
    L.append(f'Events missing more than 2 features: {int((nmiss > 2).sum())} ({(nmiss > 2).mean():.1%}); '
             f'missing 1: {int((nmiss == 1).sum())}, 2: {int((nmiss == 2).sum())}.')
    return L


def main():
    E, F, PE, G = build()
    F.to_csv(OUT / 'features.csv', index=False, float_format='%.10g')
    G.to_csv(OUT / 'product_features.csv', index=False, float_format='%.10g')
    print('\n'.join(missing_report(E, F)))
    X = F.loc[E.study.to_numpy()]
    print(X[FEATURES].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].round(3).to_string())


if __name__ == '__main__':
    main()
