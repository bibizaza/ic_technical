"""V-shape rebound gap study (descriptive; US only — STOXX 600 is NOT read here, it stays the holdout).

Run from the worktree root:  python -m research.us_gauge.vgap_study [csv_dir]
Inputs FRED_VXVCLS and CBOE_PUT come from the master data store (market_store). An optional csv_dir with
VXVCLS.csv / cboe_PUT.csv (raw FRED / CBOE downloads) is still accepted.

Parts
 A  Where the product loses the rebound (V2+F+S timing/size weights, 1993+): BH vs product by day bucket after each low.
 B  Long history (S&P 1928+, EODHD GSPC.INDX): forward returns by distance below the 200-day average.
 C  "Turn" signals inside a drawdown (fast-trend re-entry): N-day closing high while 15%+ below the 1-year high.
 D  Volatility estimators for sizing: trailing 21-day vs faster / blended, as forecasts and inside vol sizing.
 E  VIX / VIX3M back below 1 (2007-12+).
 F  Slow/fast momentum states (monthly, 1929+) and the same question inside the base-rule cash state (1955+).
 G  Sizing with VIX instead of / next to trailing vol (1990+).
 H  How front-loaded rebounds are.
 I  S sizing variants and a put-write cash leg (optional cboe_PUT.csv from the free CBOE history file).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from research.us_gauge.data import load, load_inputs  # noqa: E402
from research.us_gauge.run_step1 import build_signals  # noqa: E402
from research.us_gauge.run_p import episodes as f_episodes  # noqa: E402
from research.us_gauge.run_c import floor_weights  # noqa: E402
from research.us_gauge.run_fast import states, to_weights, v2_from_states  # noqa: E402
from research.us_gauge.run_s import shock_episodes  # noqa: E402
from research.us_gauge import run_vol  # noqa: E402
from research.us_gauge.trading import net_returns  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.float_format', '{:.3f}'.format)

STORE = Path.home() / 'Library/CloudStorage/Dropbox/Tools_In_Construction/data/market_store'


def product(inputs):
    """V2+F+S weights exactly as run_s.main builds them (sector layer left out: it does not time)."""
    idx = inputs.index
    gf = build_signals(inputs)
    d1 = gf.loc['1993-04-01':].index
    s1 = states(d1, gf.cash_R1, gf.deep, gf.dmas, None)
    w1 = to_weights(idx, s1)
    w2 = v2_from_states(inputs, s1)
    epf = f_episodes(idx, s1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    w2F = np.maximum(w2, floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)).where(w2.notna())
    ep1 = shock_episodes(inputs.px, w1)
    v = run_vol.target_weight(inputs.eq_ret)
    sw = pd.Series(0., index=idx)
    for r in ep1.itertuples():
        sw.iloc[r.i_entry + 2:min(r.i_end + 2, len(idx))] = v.iloc[r.i_entry]
    w2FS = np.maximum(w2F, sw).where(w2F.notna())
    return pd.DataFrame({'R1': w1, 'V2': w2, 'V2F': w2F, 'V2FS': w2FS, 'voltarget': v})


def part_a(inputs, W):
    ev = pd.read_csv(OUT / 'vshape_events.csv', parse_dates=['peak', 'trough', 'confirm'])
    ev = ev[ev.trough >= '1993-06-01']
    idx = inputs.index
    rp = net_returns(W.V2FS, inputs.eq_ret, inputs.cash_ret)
    r1 = net_returns(W.R1, inputs.eq_ret, inputs.cash_ret)
    rows = []
    for e in ev.itertuples():
        i = idx.get_loc(e.trough)
        row = dict(trough=e.trough.date(), cls=e.cls, depth=e.depth,
                   w_R1_at_low=W.R1.iloc[i], w_prod_at_low=W.V2FS.iloc[i])
        for a, b in ((0, 5), (5, 10), (10, 21), (21, 63), (0, 63), (0, 126)):
            sl = slice(i + 1 + a, min(i + 1 + b, len(idx)))
            bh = (1 + inputs.eq_ret.iloc[sl]).prod() - 1
            pr = (1 + rp.iloc[sl]).prod() - 1
            row[f'bh_{a}_{b}'], row[f'prod_{a}_{b}'] = bh, pr
            row[f'w_{a}_{b}'] = W.V2FS.iloc[sl].mean()
            if (a, b) == (0, 63):
                row['r1_0_63'] = (1 + r1.iloc[sl]).prod() - 1
                row['wR1_0_63'], row['wvol_0_63'] = W.R1.iloc[sl].mean(), W.voltarget.iloc[sl].mean()
        nz = np.flatnonzero(W.R1.iloc[i + 1:].to_numpy() >= 1)
        row['days_to_R1_invested'] = int(nz[0]) + 1 if len(nz) else np.nan
        nf = np.flatnonzero(W.V2FS.iloc[i + 1:].to_numpy() >= .9)
        row['days_to_prod_90pct'] = int(nf[0]) + 1 if len(nf) else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'vgap_capture.csv', index=False)
    print('\n== A. capture by event (V2+F+S vs buy & hold) ==')
    print(t[['trough', 'cls', 'depth', 'w_R1_at_low', 'w_prod_at_low', 'bh_0_5', 'prod_0_5', 'bh_5_10', 'prod_5_10',
             'bh_10_21', 'prod_10_21', 'bh_21_63', 'prod_21_63', 'bh_0_63', 'prod_0_63', 'r1_0_63', 'wR1_0_63',
             'wvol_0_63', 'days_to_R1_invested', 'days_to_prod_90pct']].to_string())
    for name, m in (('all', t.cls.notna()), ('V', t.cls == 'V'), ('FAILED', t.cls == 'FAILED'),
                    ('cash at low (R1=0)', t.w_R1_at_low == 0), ('V & cash at low', (t.cls == 'V') & (t.w_R1_at_low == 0))):
        s = t[m]
        print(f'{name}: n={len(s)} ' + ' '.join(
            f'[{a}-{b}] bh {s[f"bh_{a}_{b}"].mean():+.3f} prod {s[f"prod_{a}_{b}"].mean():+.3f} w {s[f"w_{a}_{b}"].mean():.2f}'
            for a, b in ((0, 5), (5, 10), (10, 21), (21, 63), (0, 63))))
    return t


def fwd(px, n):
    return px.shift(-n) / px - 1


def clusters(mask, gap=63):
    """First day of each run of True separated by at least `gap` days of False."""
    pos = np.flatnonzero(mask.to_numpy())
    if not len(pos):
        return mask.index[:0]
    keep = [pos[0]] + [p for q, p in zip(pos[:-1], pos[1:]) if p - q > gap]
    return mask.index[keep]


def part_b(px):
    ma = px.rolling(200).mean()
    dist = px / ma - 1
    F = pd.DataFrame({n: fwd(px, n) for n in (5, 21, 63, 126)})
    bins = [-1, -.30, -.20, -.15, -.10, -.05, 0, .05, 1]
    lab = ['<-30%', '-30..-20', '-20..-15', '-15..-10', '-10..-5', '-5..0', '0..5', '>5%']
    b = pd.cut(dist, bins, labels=lab)
    rows = []
    for era, a, z in (('1928-2026', None, None), ('1928-1954', None, '1954'), ('1955-1989', '1955', '1989'),
                      ('1990-2026', '1990', None), ('excl. 1929-32', '1933', None)):
        for l in lab:
            m = (b == l).loc[a:z]
            d = F.loc[m.index][m]
            first = clusters(m)
            ff = F.loc[first]
            rows.append(dict(era=era, bucket=l, days=int(m.sum()), episodes=len(first),
                             **{f'mean{n}': d[n].mean() for n in F}, **{f'hit{n}': (d[n] > 0).mean() for n in F},
                             p05_63=d[63].quantile(.05), ep_mean63=ff[63].mean(), ep_hit63=(ff[63] > 0).mean(),
                             ep_worst63=ff[63].min(), ep_mean126=ff[126].mean()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'vgap_ma200_buckets.csv', index=False)
    print('\n== B. forward return by distance from the 200-day average ==')
    print(t.to_string())
    # first crossings below -10/-15/-20: list by episode
    for thr in (-.10, -.15, -.20):
        first = clusters(dist < thr)
        ff = F.loc[first]
        mdd = [px.loc[d:].iloc[:127].min() / px[d] - 1 for d in first]
        e = pd.DataFrame({'date': first.date, 'f21': ff[21].to_numpy(), 'f63': ff[63].to_numpy(),
                          'f126': ff[126].to_numpy(), 'worst_126': mdd})
        e.to_csv(OUT / f'vgap_ma200_first_{int(-thr * 100)}.csv', index=False)
        print(f'first close {thr:.0%} below 200d (63d apart): n={len(e)} mean63 {e.f63.mean():+.3f} hit63 {(e.f63 > 0).mean():.2f} '
              f'mean126 {e.f126.mean():+.3f} worst-path mean {e.worst_126.mean():+.3f} min {e.worst_126.min():+.3f}')
        if thr == -.15:
            print(e.round(3).to_string())
    return dist


def part_c(px):
    """Fast-trend 'turn': first N-day closing high after a new low, while the low is 15%+ below the 252-day high."""
    hi252 = px.rolling(252).max()
    p = px.to_numpy()
    rows = []
    for N in (10, 21, 42, 63):
        hiN = px.rolling(N).max().to_numpy()
        low, armed = np.nan, False
        for i in range(252, len(p)):
            dd = p[i] / hi252.iloc[i] - 1
            if dd <= -.15 and (np.isnan(low) or p[i] < low):
                low, li, armed = p[i], i, True          # new low of the decline: (re)arm
            if dd > -.02:
                low, armed = np.nan, False               # back near the high: decline over
            if armed and p[i] >= hiN[i] and i > li:
                f = lambda n: p[min(i + n, len(p) - 1)] / p[i] - 1
                path = p[i:i + 127]
                rows.append(dict(N=N, date=px.index[i], low_date=px.index[li], off_low=p[i] / low - 1, dd_at_low=low / hi252.iloc[li] - 1,
                                 days_from_low=i - li, f21=f(21), f63=f(63), f126=f(126), worst126=path.min() / p[i] - 1,
                                 trap=bool(path.min() < low)))
                armed = False
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'vgap_turn_signals.csv', index=False)
    print('\n== C. N-day closing high after a 15%+ decline low (one signal per new low) ==')
    for era, a, z in (('1928-2026', '1928', '2026'), ('1928-1954', '1928', '1954'), ('1955-1989', '1955', '1989'), ('1990-2026', '1990', '2026')):
        for N, g in t[(t.date >= a) & (t.date <= z + '-12-31')].groupby('N'):
            print(f'{era} N={N}: n={len(g)} off-low {g.off_low.mean():+.3f} days {g.days_from_low.median():.0f} f63 {g.f63.mean():+.3f} hit {(g.f63 > 0).mean():.2f} '
                  f'f126 {g.f126.mean():+.3f} trap share {g.trap.mean():.2f} worst126 mean {g.worst126.mean():+.3f} min {g.worst126.min():+.3f}')
    base = fwd(px, 63)
    print(f'unconditional 63d: mean {base.mean():+.3f} hit {(base > 0).mean():.2f}')
    return t


def part_d(px_long, inputs, W):
    """Vol estimators: forecast next-21-day realised vol; and vol-sized buy & hold, 1928+ (price only, zero cash)."""
    r = px_long.pct_change()
    ann = np.sqrt(252)
    est = pd.DataFrame({
        's21': r.rolling(21).std() * ann, 's10': r.rolling(10).std() * ann, 's5': r.rolling(5).std() * ann,
        's63': r.rolling(63).std() * ann, 's252': r.rolling(252).std() * ann})
    est['har'] = (est.s5 + est.s21 + est.s252) / 3                       # equal-weight HAR-style (Corsi horizons: week, month + year anchor)
    est['min21_63'] = est[['s21', 's63']].min(axis=1)
    est['max5_21'] = est[['s5', 's21']].max(axis=1)
    est['blend21_252'] = (est.s21 + est.s252) / 2
    real = (r.rolling(21).std() * ann).shift(-21)
    dd = px_long / px_long.rolling(252).max() - 1
    rows = []
    for nm, m in (('all days', dd.notna()), ('in 15%+ drawdown', dd <= -.15),
                  ('within 42d after a 15%+ decline low', None)):
        if m is None:
            low63 = px_long.rolling(63).min()
            is_low = (px_long == low63) & (dd <= -.15)
            pos = np.flatnonzero(is_low.to_numpy())
            flag = np.zeros(len(px_long), bool)
            for q in pos:
                flag[q + 1:q + 43] = True
            flag[pos] = False
            m = pd.Series(flag, index=px_long.index)
        for c in est:
            x = pd.concat([est[c], real], axis=1)[m.reindex(est.index).fillna(False).to_numpy()].dropna()
            ratio = x.iloc[:, 1] / x.iloc[:, 0]
            ql = (x.iloc[:, 1] ** 2 / x.iloc[:, 0] ** 2 - np.log(x.iloc[:, 1] ** 2 / x.iloc[:, 0] ** 2) - 1).mean()
            rows.append(dict(sample=nm, est=c, n=len(x), bias_ratio=ratio.median(), rmse=np.sqrt(((x.iloc[:, 1] - x.iloc[:, 0]) ** 2).mean()), qlike=ql))
    t = pd.DataFrame(rows)
    print('\n== D1. forecast of next-21-day realised vol (bias_ratio = realised / estimate, median) ==')
    print(t.to_string())
    # D2: vol-sized buy and hold, daily, weight = min(1, 15%/est) lagged 2 days, 10 bp
    rows = []
    for era, a, z in (('1928-2026', '1929', None), ('1929-1954', '1929', '1954'), ('1955-1989', '1955', '1989'), ('1990-2026', '1990', None)):
        for c in est:
            w = (.15 / est[c]).clip(upper=1).shift(2)
            ret = (w * r - w.diff().abs() * .001).loc[a:z].dropna()
            eq = (1 + ret).cumprod()
            rows.append(dict(era=era, est=c, sharpe0=ret.mean() / ret.std() * ann, cagr=eq.iloc[-1] ** (252 / len(ret)) - 1,
                             maxdd=(eq / eq.cummax() - 1).min(), avg_w=w.loc[a:z].mean(), turnover=w.diff().abs().loc[a:z].sum() / (len(ret) / 252)))
        ret = r.loc[a:z].dropna()
        eq = (1 + ret).cumprod()
        rows.append(dict(era=era, est='BH', sharpe0=ret.mean() / ret.std() * ann, cagr=eq.iloc[-1] ** (252 / len(ret)) - 1,
                         maxdd=(eq / eq.cummax() - 1).min(), avg_w=1., turnover=0.))
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / 'vgap_vol_estimators.csv', index=False)
    print('\n== D2. vol-sized buy & hold, price-only, zero cash rate (Sharpe not comparable with the product tables) ==')
    print(t2.to_string())
    # D3: weight implied by each estimator after the three named lows
    print('\n== D3. size min(1, 15%/est) on days +5/+10/+21/+42/+63 after the low ==')
    vix = inputs.vix
    for low in ('1998-08-31', '2002-10-09', '2009-03-09', '2011-10-03', '2018-12-24', '2020-03-23', '2022-10-12', '2025-04-08'):
        i = px_long.index.get_loc(pd.Timestamp(low))
        line = {c: [round(float(min(1, .15 / est[c].iloc[i + k])), 2) for k in (5, 10, 21, 42, 63)] for c in ('s21', 's10', 'har', 'blend21_252', 'min21_63')}
        j = inputs.index.get_loc(pd.Timestamp(low))
        line['vix'] = [round(float(min(1, 15 / vix.iloc[j + k])), 2) for k in (5, 10, 21, 42, 63)]
        line['fwd_real21'] = [round(float(real.iloc[i + k]), 2) for k in (5, 10, 21, 42, 63)]
        print(low, line)
    return t, t2


def part_e(inputs, folder):
    v3 = (pd.read_csv(Path(folder) / 'VXVCLS.csv', index_col=0, parse_dates=True).iloc[:, 0] if folder
          else pd.read_csv(STORE / 'FRED_VXVCLS.csv', index_col=0, parse_dates=True).iloc[:, 0])
    v3 = pd.to_numeric(v3, errors='coerce').dropna()
    d = pd.DataFrame({'px': inputs.px, 'vix': inputs.vix, 'v3': v3}).dropna()
    ratio = d.vix / d.v3
    inv = ratio > 1
    rows = []
    run = 0
    pk = 0.
    for i in range(1, len(d)):
        if inv.iloc[i - 1]:
            run += 1
            pk = max(pk, ratio.iloc[i - 1])
        if inv.iloc[i - 1] and not inv.iloc[i]:
            if run >= 5:
                f = lambda n: d.px.iloc[min(i + n, len(d) - 1)] / d.px.iloc[i] - 1
                rows.append(dict(date=d.index[i].date(), days_inverted=run, peak_ratio=pk, vix=d.vix.iloc[i],
                                 f21=f(21), f63=f(63), f126=f(126), worst63=d.px.iloc[i:i + 64].min() / d.px.iloc[i] - 1))
        if not inv.iloc[i - 1] and not inv.iloc[i]:
            if (~inv.iloc[max(0, i - 5):i]).all():
                run, pk = 0, 0.
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'vgap_vix_term.csv', index=False)
    print('\n== E. VIX/VIX3M back below 1 after 5+ inverted days (2007-12+) ==')
    print(t.round(3).to_string())
    print(f'n={len(t)} mean f63 {t.f63.mean():+.3f} hit {(t.f63 > 0).mean():.2f}; unconditional 63d since 2008: {fwd(d.px, 63).mean():+.3f}')
    return t


def part_f(g, inputs):
    """Slow/fast momentum states (Goulding-Harvey-Mazzoleni 2023): sign of 12-month and 1-month return, next month."""
    m = g.resample('ME').last()
    slow, fast, nxt = m.pct_change(12), m.pct_change(1), m.pct_change().shift(-1)
    st = np.select([(slow > 0) & (fast > 0), (slow > 0) & (fast <= 0), (slow <= 0) & (fast <= 0), (slow <= 0) & (fast > 0)],
                   ['Bull', 'Correction', 'Bear', 'Rebound'], default='na')
    d = pd.DataFrame({'st': st, 'nxt': nxt}).dropna()
    d = d[d.st != 'na']
    rows = []
    for era, a, z in (('1929-2026', None, None), ('1929-1954', None, '1954'), ('1955-1989', '1955', '1989'), ('1990-2026', '1990', None)):
        for s_, x in d.loc[a:z].groupby('st'):
            rows.append(dict(era=era, state=s_, months=len(x), share=len(x) / len(d.loc[a:z]), mean=x.nxt.mean(), vol=x.nxt.std(),
                             t=x.nxt.mean() / x.nxt.std() * np.sqrt(len(x)), hit=(x.nxt > 0).mean(), worst=x.nxt.min()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'vgap_momentum_states.csv', index=False)
    print('\n== F1. next-month S&P return by slow (12m) / fast (1m) momentum state ==')
    print(t.to_string())
    # F2: inside the base-rule cash state (R0, DMAS < 45 at the first decision of the month), 1955+
    dm = pd.read_csv(OUT / 'dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0]
    px = inputs.px
    first = px.groupby(px.index.to_period('M')).head(1).index
    first = first[first >= '1955-02-01']
    q = pd.DataFrame({'dmas': dm.reindex(first), 'r21': (px / px.shift(21) - 1).reindex(first), 'r10': (px / px.shift(10) - 1).reindex(first),
                      'r63': (px / px.shift(63) - 1).reindex(first), 'above50': (px > px.rolling(50).mean()).reindex(first),
                      'nxt': (px.shift(-23) / px.shift(-2) - 1).reindex(first)}).dropna()
    cash = q[q.dmas < 45]
    print(f'\n== F2. months that start with DMAS < 45 (R0 cash), 1955+: n={len(cash)}; next-month return (t+2 to t+23) ==')
    rows = []
    for era, a, z in (('1955-2026', None, None), ('1955-1989', None, '1989'), ('1990-2026', '1990', None)):
        c = cash.loc[a:z]
        for nm, mk in (('21d return > 0', c.r21 > 0), ('10d return > 0', c.r10 > 0), ('63d return > 0', c.r63 > 0), ('close > 50d average', c.above50),
                       ('21d return > +5%', c.r21 > .05)):
            for lab, x in (('yes', c[mk]), ('no', c[~mk])):
                rows.append(dict(era=era, cond=nm, val=lab, months=len(x), mean=x.nxt.mean(), vol=x.nxt.std(), hit=(x.nxt > 0).mean(),
                                 worst=x.nxt.min(), t=x.nxt.mean() / x.nxt.std() * np.sqrt(len(x))))
        x = q.loc[a:z][q.loc[a:z].dmas >= 45]
        rows.append(dict(era=era, cond='(DMAS >= 45, invested)', val='-', months=len(x), mean=x.nxt.mean(), vol=x.nxt.std(), hit=(x.nxt > 0).mean(),
                         worst=x.nxt.min(), t=x.nxt.mean() / x.nxt.std() * np.sqrt(len(x))))
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / 'vgap_cash_state_fast.csv', index=False)
    print(t2.to_string())
    bad = cash[(cash.r21 > 0)].nsmallest(12, 'nxt')
    print('worst next months after "cash and 21d return > 0":'); print(bad.round(3).to_string())


def part_g(inputs):
    """Size by min(trailing 21d vol, VIX): 1990+, vol-sized buy & hold with the real cash rate, and forecast bias."""
    r, vix = inputs.eq_ret, inputs.vix / 100
    s21 = r.rolling(21).std() * np.sqrt(252)
    real = s21.shift(-21)
    px = inputs.px
    dd = px / px.rolling(252).max() - 1
    low = (px == px.rolling(63).min()) & (dd <= -.15)
    flag = np.zeros(len(px), bool)
    for q in np.flatnonzero(low.to_numpy()):
        flag[q + 1:q + 43] = True
    est = {'s21': s21, 'vix': vix, 'min(s21,vix)': np.minimum(s21, vix), 'max(s21,vix)': np.maximum(s21, vix), 'mean(s21,vix)': (s21 + vix) / 2}
    rows = []
    for nm, e in est.items():
        x = pd.DataFrame({'e': e, 'real': real}).loc['1990-02-01':]
        for lab, mk in (('all', np.ones(len(x), bool)), ('42d after low', flag[-len(x):])):
            y = x[mk].dropna()
            rows.append(dict(est=nm, sample=lab, n=len(y), bias_ratio=(y.real / y.e).median(),
                             qlike=(y.real ** 2 / y.e ** 2 - np.log(y.real ** 2 / y.e ** 2) - 1).mean()))
        w = (.15 / e).clip(upper=1).shift(2)
        for era, a, z in (('1990-2026', '1990-03-01', None), ('1990-2007', '1990-03-01', '2007'), ('2008-2026', '2008', None)):
            ret = (w * r + (1 - w) * inputs.cash_ret - w.diff().abs() * .001).loc[a:z].dropna()
            ex = ret - inputs.cash_ret.reindex(ret.index)
            eq = (1 + ret).cumprod()
            rows.append(dict(est=nm, sample=era, sharpe=ex.mean() / ex.std() * np.sqrt(252), cagr=eq.iloc[-1] ** (252 / len(ret)) - 1,
                             maxdd=(eq / eq.cummax() - 1).min(), avg_w=w.loc[a:z].mean()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'vgap_vix_sizing.csv', index=False)
    print('\n== G. sizing with VIX (1990+) ==')
    print(t.to_string())


def part_h(g):
    """How front-loaded the rebound is: share of the 63-day move made in the first 5/10/21 days, all 41 events."""
    ev = pd.read_csv(OUT / 'vshape_events.csv', parse_dates=['trough'])
    rows = []
    for e in ev.itertuples():
        i = g.index.get_loc(e.trough)
        f = lambda n: g.iloc[min(i + n, len(g) - 1)] / g.iloc[i] - 1
        rows.append(dict(cls=e.cls, d5=f(5), d10=f(10), d21=f(21), d63=f(63)))
    t = pd.DataFrame(rows)
    print('\n== H. buy & hold after the low, all events 1955+ (mean) ==')
    print(t.groupby('cls').mean().assign(n=t.groupby('cls').size()).to_string())
    print(t[['d5', 'd10', 'd21', 'd63']].mean().to_string())


def part_i(inputs, folder):
    """S sizing variants on the product (3 US episodes on the R1 base: information only) and a put-write cash leg (2007+)."""
    from research.us_gauge.stress_periods import stats
    idx = inputs.index
    gf = build_signals(inputs)
    s1 = states(gf.loc['1993-04-01':].index, gf.cash_R1, gf.deep, gf.dmas, None)
    w1, w2 = to_weights(idx, s1), v2_from_states(inputs, s1)
    epf = f_episodes(idx, s1, inputs.vix, inputs.vix, inputs.px, .5, 30.)
    w2F = np.maximum(w2, floor_weights(idx, list(zip(epf.i_entry, epf.i_end)), .5)).where(w2.notna())
    ep, v = shock_episodes(inputs.px, w1), run_vol.target_weight(inputs.eq_ret)

    def sw(mode):
        s = pd.Series(0., index=idx)
        for r in ep.itertuples():
            a, b = r.i_entry + 2, min(r.i_end + 2, len(idx))
            s.iloc[a:b] = {'frozen': v.iloc[r.i_entry], 'full': 1.}.get(mode, 0.)
            if mode == 'live':
                s.iloc[a:b] = v.shift(2).iloc[a:b].to_numpy()
        return s
    R = lambda w: net_returns(w, inputs.eq_ret, inputs.cash_ret)
    ret = pd.DataFrame({'BH': inputs.eq_ret, 'V2+F': R(w2F), **{f'S {m}': R(np.maximum(w2F, sw(m)).where(w2F.notna()))
                                                                 for m in ('frozen', 'live', 'full')}}).loc['1993-06-01':]
    cash = inputs.cash_ret.reindex(ret.index)
    W = [('FULL 1993+', None, None), ('1993-2009', None, '2009-12-31'), ('2010+', '2010-01-01', None), ('1998-09..12', '1998-09-01', '1998-12-31'),
         ('2008 crisis', '2007-10-09', '2009-03-09'), ('COVID rebound', '2020-03-24', '2020-08-31'), ('LibDay rebound', '2025-04-09', '2025-06-30')]
    t = pd.DataFrame([(n, c, *stats(ret[c].loc[a:b], cash)) for n, a, b in W for c in ret], columns=['window', 'rule', 'total', 'sharpe', 'maxdd'])
    t.to_csv(OUT / 'vgap_s_sizing.csv', index=False)
    print('\n== I1. S sizing on V2+F (episodes: ' + ', '.join(f'{r.entry.date()} v={v.iloc[r.i_entry]:.2f}' for r in ep.itertuples()) + ') ==')
    for m in ('total', 'sharpe', 'maxdd'):
        print(m); print(t.pivot_table(index='window', columns='rule', values=m, sort=False)[list(ret.columns)].to_string())
    f = Path(folder) / 'cboe_PUT.csv' if folder else STORE / 'CBOE_PUT.csv'
    if f.exists():
        p = pd.read_csv(f, index_col=0).iloc[:, 0]
        p.index = pd.to_datetime(p.index, format='mixed')
        p = p.sort_index()['2007-06-01':]
        x = pd.DataFrame({'spx': inputs.eq_ret, 'put': p.pct_change().reindex(idx), 'cash': inputs.cash_ret, 'w1': w1,
                          'vix': inputs.vix.shift(2), 'wF': w2F}).loc['2007-07-01':].dropna()
        base = R(w2F).reindex(x.index)
        print('\n== I2. CBOE PutWrite index as the cash leg (50% of the sleeve while R1 is in cash), 2007-07+ ==')
        print('base V2+F: Sharpe %.3f total %.3f maxDD %.3f' % tuple(np.array(stats(base, x.cash))[[1, 0, 2]]))
        for cond, lab in ((x.w1 == 0, 'whenever R1 cash'), ((x.w1 == 0) & (x.vix >= 30), 'R1 cash and VIX >= 30'), ((x.w1 == 0) & (x.vix < 30), 'R1 cash and VIX < 30')):
            r = base + np.minimum((1 - x.wF).clip(lower=0), .5) * cond * (x.put - x.cash)
            tot, sh, dd = stats(r, x.cash)
            print(f'{lab}: Sharpe {sh:.3f} total {tot:.3f} maxDD {dd:.3f} | 2008 crisis {(1 + r.loc["2007-10-09":"2009-03-09"]).prod() - 1:+.3f} '
                  f'COVID rebound {(1 + r.loc["2020-03-24":"2020-08-31"]).prod() - 1:+.3f} LibDay rebound {(1 + r.loc["2025-04-09":"2025-06-30"]).prod() - 1:+.3f}')


def main(folder):
    inputs = load_inputs()
    W = product(inputs)
    W.to_csv(OUT / 'vgap_weights.csv')
    part_a(inputs, W)
    g = load('GSPC.INDX', refresh=False).dropna().sort_index()
    g = g[g.index >= '1928-01-01']
    part_b(g)
    part_c(g)
    part_d(g, inputs, W)
    part_e(inputs, folder)
    part_f(g, inputs)
    part_g(inputs)
    part_h(g)
    part_i(inputs, folder)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
