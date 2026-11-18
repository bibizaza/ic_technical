"""Look-ahead tests for the ML features (spec 10, step 3)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay
from research.us_gauge.ml import features as fe
from research.us_gauge.ml.events import scan_events
from research.us_gauge.ml.panel import MACRO

STORE = Path('/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/data/market_store')
MKTS = ['GSPC', 'GDAXI', 'IBEX', 'N225', 'MXX', 'AXJO']


def _walk(rng, n, start, crash_every=700):
    r = rng.standard_t(4, n) * .012 + .0003
    for k in range(400, n, crash_every):
        r[k:k + 20] -= .012                         # a fast 20% fall every few years
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range(start, periods=n))


def _synthetic(seed=1):
    rng = np.random.default_rng(seed)
    P = {m: _walk(rng, 6000, '1984-01-02', 650 + 37 * i) for i, m in enumerate(MKTS)}
    days = pd.bdate_range('1970-01-01', '2008-12-31')
    macro = {}
    for n in MACRO:
        s = pd.Series(np.exp(np.cumsum(rng.normal(0, .005, len(days)))) * 2, index=days)
        if n.startswith('FRED_DEX'):
            s = s.loc['1975-01-01':]
        if n == 'FRED_DEXUSEU':
            s = s.loc['1999-01-04':]
        if n in ('FRED_DBAA', 'FRED_DAAA'):
            s = s.loc['1990-01-02':] + (1. if n == 'FRED_DBAA' else 0.)
        if n in ('FRED_BAA', 'FRED_AAA'):
            s = s.resample('MS').mean() + (1. if n == 'FRED_BAA' else 0.)
        macro[n] = s
    return P, macro


def _all_features(P, macro, market, rows):
    ctx = fe.Context(P, macro)
    ent = {market: list(scan_events(P[market]).t)}
    return fe.features_for(rows.assign(market=market), P, ctx, ent)


def _truncated(P, macro, market, t):
    cut = t - BDay(1)
    Pt = {m: (s.loc[:t] if m == market else s.loc[:cut]) for m, s in P.items()}
    Mt = {n: s.loc[:cut] for n, s in macro.items()}
    return Pt, Mt


def _rows(P, market):
    e = scan_events(P[market])
    return e[e.i_h >= 756]


@pytest.mark.parametrize('market', MKTS)
def test_truncation_gives_identical_features(market):
    P, macro = _synthetic()
    rows = _rows(P, market)
    assert len(rows) >= 2
    full = _all_features(P, macro, market, rows)
    for i, r in rows.iterrows():
        Pt, Mt = _truncated(P, macro, market, r.t)
        one = _all_features(Pt, Mt, market, rows.loc[[i]])
        pd.testing.assert_frame_equal(one, full.loc[[i]], check_exact=False, rtol=1e-12)


@pytest.mark.parametrize('market', ['GSPC', 'GDAXI', 'MXX'])
def test_future_shift_changes_nothing(market):
    P, macro = _synthetic(2)
    rows = _rows(P, market)
    full = _all_features(P, macro, market, rows)
    rng = np.random.default_rng(9)
    for i, r in rows.iterrows():
        cut = r.t - BDay(1)
        Ps = {m: s.where(s.index <= (r.t if m == market else cut), s * rng.uniform(.3, 3., len(s))) for m, s in P.items()}
        Ms = {n: s.where(s.index <= cut, s * 1.7 + 3) for n, s in macro.items()}
        ctx = fe.Context(Ps, Ms)
        prior = [d for d in scan_events(P[market]).t if d < r.t]
        f = fe.event_features(market, Ps[market], int(r.i_h), int(r.i_l), int(r.i_t), ctx, prior)
        np.testing.assert_allclose(pd.Series(f)[fe.FEATURES].to_numpy(float), full.loc[i, fe.FEATURES].to_numpy(float),
                                   rtol=1e-12, equal_nan=True)


def test_monthly_credit_uses_only_finished_months():
    P, macro = _synthetic(3)
    ctx = fe.Context(P, macro)
    h, t = pd.Timestamp('1987-09-15'), pd.Timestamp('1987-11-02')      # before daily credit data (1990 here)
    cut = t - BDay(1)                                                  # 1987-10-30: October not finished
    base = ctx.d_credit(h, cut)
    m2 = dict(macro)
    baa = macro['FRED_BAA'].copy()
    baa.loc['1987-10-01'] += 5.                                        # October value: usable only from 1987-11-02
    baa.loc['1987-11-01'] += 5.
    m2['FRED_BAA'] = baa
    assert fe.Context(P, m2).d_credit(h, cut) == pytest.approx(base)
    assert fe.Context(P, m2).d_credit(h, t) != pytest.approx(base)     # on 1987-11-02 October is usable
    baa2 = macro['FRED_BAA'].copy()
    baa2.loc['1987-08-01'] += 5.                                       # the month before h's month is the start
    m2['FRED_BAA'] = baa2
    assert fe.Context(P, m2).d_credit(h, cut) == pytest.approx(base - 5.)


def test_fx_rules():
    P, macro = _synthetic(4)
    ctx = fe.Context(P, macro)
    h, cut = pd.Timestamp('1990-03-01'), pd.Timestamp('1990-09-03')
    dkk = macro['FRED_DEXDNUS']
    assert ctx.fx_shock('GDAXI', h, cut) == pytest.approx(np.log(fe.asof(dkk, cut) / fe.asof(dkk, h)))
    assert np.isnan(ctx.fx_shock('IBEX', h, cut))                     # Spain before 1999: missing
    assert np.isnan(ctx.fx_shock('GDAXI', pd.Timestamp('1974-01-02'), cut))   # before the FX series starts
    eur = 1 / macro['FRED_DEXUSEU']
    h2, c2 = pd.Timestamp('2001-03-01'), pd.Timestamp('2001-09-03')
    assert ctx.fx_shock('IBEX', h2, c2) == pytest.approx(np.log(fe.asof(eur, c2) / fe.asof(eur, h2)))
    jp, sz, uk, ca = (macro[n] for n in ('FRED_DEXJPUS', 'FRED_DEXSZUS', 'FRED_DEXUSUK', 'FRED_DEXCAUS'))
    lc = lambda s: np.log(fe.asof(s, cut) / fe.asof(s, h))
    assert ctx.fx_shock('GSPC', h, cut) == pytest.approx(-np.mean([lc(jp), lc(sz), -lc(uk), lc(ca)]))
    au = macro['FRED_DEXUSAL']
    assert ctx.fx_shock('AXJO', h, cut) == pytest.approx(-lc(au))     # + = local currency fell


@pytest.mark.skipif(not STORE.exists(), reason='market store not available')
def test_truncation_on_real_events():
    from research.us_gauge.ml.panel import load_panel, load_macro, MARKETS
    P, _ = load_panel()
    macro = load_macro()
    E = pd.read_csv(Path(__file__).resolve().parents[1] / 'research/us_gauge/ml/out/events.csv', parse_dates=['h', 'l', 't'])
    F = pd.read_csv(Path(__file__).resolve().parents[1] / 'research/us_gauge/ml/out/features.csv').set_index('event_id')
    S = E[E.study].sample(40, random_state=0)
    for r in S.itertuples():
        Pt, Mt = _truncated(P, macro, r.market, r.t)
        ctx = fe.Context(Pt, Mt)
        prior = list(scan_events(Pt[r.market]).t)[:-1]               # events found on closes up to t, before t
        f = fe.event_features(r.market, Pt[r.market], r.i_h, r.i_l, r.i_t, ctx, prior)
        np.testing.assert_allclose(pd.Series(f)[fe.FEATURES].to_numpy(float), F.loc[r.event_id, fe.FEATURES].to_numpy(float),
                                   rtol=1e-8, equal_nan=True, err_msg=r.event_id)


def test_other_close_series_of_the_market_gives_same_features():
    """US core episodes are scored on BBG_SPX closes: a series object outside the panel must give the same result."""
    P, macro = _synthetic(5)
    rows = _rows(P, 'GSPC')
    ctx = fe.Context(P, macro)
    prior = list(scan_events(P['GSPC']).t)
    for r in rows.itertuples():
        pr = [d for d in prior if d < r.t]
        a = fe.event_features('GSPC', P['GSPC'], r.i_h, r.i_l, r.i_t, ctx, pr)
        b = fe.event_features('GSPC', P['GSPC'].copy(), r.i_h, r.i_l, r.i_t, ctx, pr)
        np.testing.assert_allclose(pd.Series(a)[fe.FEATURES].to_numpy(float), pd.Series(b)[fe.FEATURES].to_numpy(float),
                                   rtol=1e-12, equal_nan=True)
