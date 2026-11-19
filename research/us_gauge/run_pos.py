"""P: systematic-fund positioning (CTA + vol-target) — pre-registered in spec 6P (commit 06702ef). One run.

P-in  = re-entry when the systematic funds' forced selling is over (exposure flushed, starts rising).
P-out = 50% cap for 21 days when their exposure is crowded and starts falling.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_vol
from research.us_gauge.data import load, load_inputs, cash_daily
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.gauge import dmas_on
from research.us_gauge.run_c import sharpe
from research.us_gauge.evaluate import metrics
from research.us_gauge.run_fast import states, to_weights
from research.us_gauge.run_s import shock_episodes
from research.us_gauge.run_b import breadth, deemer_ratio, thrust_days, thrust_episodes, placebo_spans
from research.us_gauge.revisions import detect_rolls, revisions, rev_zscore
from research.us_gauge.trading import net_returns
from research.us_gauge.core_slide_data import layer, desired_from_states

HERE = Path(__file__).resolve().parent
DMAS_OLD = HERE / 'out/dmas_daily_1928_1954.csv'
SPLIT = '1976-12-31'
CUT, CUT_DAYS = .5, 21
N_PLACEBO = 1000
MIN_EP = 8


# ---------------------------------------------------------------- proxies (spec 6P, fixed)
def proxies(px: pd.Series, ret: pd.Series) -> pd.DataFrame:
    s21 = ret.rolling(21, min_periods=21).std() * np.sqrt(252)
    s63 = ret.rolling(63, min_periods=63).std() * np.sqrt(252)
    vt = (.10 / np.maximum(s21, s63)).clip(upper=1.5)
    lp, d63 = np.log(px), s63 / np.sqrt(252)
    sig = sum(((lp - lp.shift(h)) / (d63 * np.sqrt(h))).clip(-1, 1) for h in (21, 63, 252)) / 3
    cta = sig * (.10 / s63).clip(upper=2.)
    e = cta + vt
    return pd.DataFrame({'vt': vt, 'cta': cta, 'E': e, 'F': e - e.shift(5), 'rank': trailing_rank(e)})


def trailing_rank(e: pd.Series, n=2520, n_min=1260) -> pd.Series:
    """Share of the previous n values (current excluded, at least n_min) strictly below the current value."""
    x, out = e.to_numpy(), np.full(len(e), np.nan)
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        w = x[max(0, i - n):i]
        w = w[~np.isnan(w)]
        if len(w) >= n_min:
            out[i] = (w < x[i]).mean()
    return pd.Series(out, index=e.index)


def in_triggers(P: pd.DataFrame, key_w: pd.Series) -> pd.DatetimeIndex:
    lo = P['rank'].rolling(21, min_periods=1).min()
    c = (key_w == 0) & (lo <= .10) & (P.F > 0) & (P.F.shift(1) <= 0)
    return P.index[c.fillna(False).to_numpy()]


def out_triggers(P: pd.DataFrame, key_w: pd.Series) -> pd.DatetimeIndex:
    hi = P['rank'].rolling(21, min_periods=1).max()
    c = (key_w == 1) & (hi >= .90) & (P.F < 0) & (P.F.shift(1) >= 0)
    return P.index[c.fillna(False).to_numpy()]


def cut_starts(index, trig) -> list:
    """Positions of cut starts; a trigger inside a running cut is ignored."""
    pos, out, free = index.get_indexer(pd.DatetimeIndex(trig)), [], 0
    for k in pos:
        if k >= free:
            out.append(int(k))
            free = k + CUT_DAYS
    return out


def cap_series(index, starts) -> pd.Series:
    """Decision-time cap: desired <= 50% on signal days t .. t+20 (earn days t+2 .. t+22)."""
    c = np.ones(len(index))
    for k in starts:
        c[k:k + CUT_DAYS] = CUT
    return pd.Series(c, index=index)


def random_cuts(rng, n, elig, n_idx):
    """n non-overlapping 21-day cut starts on eligible positions."""
    for _ in range(1000):
        s = np.sort(rng.choice(elig, size=n, replace=False))
        if (np.diff(s) >= CUT_DAYS).all() and s[-1] + CUT_DAYS <= n_idx:
            return list(s)
    raise RuntimeError('no placebo draw without overlap')


# ---------------------------------------------------------------- base skeleton 1929 -> end
def skeleton_inputs() -> pd.DataFrame:
    with read_only_external():
        px = load('GSPC.INDX', refresh=False).sort_index().astype(float)
        div = load('SHILLER_SPX_DIV', refresh=False).sort_index()
        dtb3, bil = load('FRED_DTB3', refresh=False), load('BIL.US_TR', refresh=False)
    px = px[px.notna() & (px > 0)]
    px = px[~px.index.duplicated()]
    d = div.reindex(px.index, method='ffill')
    cash = cash_daily(dtb3, bil, px.index).fillna(0.)
    return pd.DataFrame({'px': px, 'eq_ret': px.pct_change(fill_method=None) + d / px / 252, 'cash_ret': cash})


def daily_dmas(px: pd.Series) -> pd.Series:
    new = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0]
    if DMAS_OLD.exists():
        old = pd.read_csv(DMAS_OLD, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        days = px.index[252:][px.index[252:] < new.index[0]]
        old = dmas_on(px, days)
        old.to_frame('dmas').to_csv(DMAS_OLD)
    return pd.concat([old, new]).reindex(px.index[252:])


def skeleton(df, dm):
    idx = df.index
    s1 = states(dm.index, dm < 45, dm < 30, dm, None)
    w1 = to_weights(idx, s1)
    v = run_vol.target_weight(df.eq_ret)
    half = np.maximum(v.shift(2).fillna(0.), .5)
    ws = layer(idx, shock_episodes(df.px, w1), half)
    desired = pd.concat([s1.reindex(idx).ffill() * v, ws.shift(-2)], axis=1).max(axis=1).where(s1.reindex(idx).ffill().notna())
    return w1, half, desired


def fast_act(index, desired, band=.10) -> pd.Series:
    """Same rule as core_slide_data.act_on_change (weight in force only), in numpy for the placebo loops."""
    d, n = desired.reindex(index).to_numpy(dtype=float), len(index)
    out, held = np.full(n, np.nan), np.nan
    for i in range(n):
        x = d[i]
        if x != x:
            continue
        if held != held or (x == 0) != (held == 0) or abs(x - held) >= band - 1e-12:
            held = x
        if i + 2 < n:
            out[i + 2] = held
    return pd.Series(out, index=index).ffill()


def run_w(idx, desired, df):
    w = fast_act(idx, (desired * 20).round() / 20)
    return net_returns(w, df.eq_ret, df.cash_ret).where(w.notna())


def with_in(idx, desired, ep, half):
    ep = ep[ep.i_entry + 2 < len(idx)]          # an episode with no earn day adds nothing (pandas fails on the empty slice)
    return pd.concat([desired, layer(idx, ep, half).shift(-2)], axis=1).max(axis=1).where(desired.notna())


def with_out(idx, desired, starts):
    return np.minimum(desired, cap_series(idx, starts)).where(desired.notna())


def sh_dd(r, cash):
    r = r.dropna()
    c = cash.reindex(r.index)
    return sharpe(r, c), metrics(r, c)['maxdd']


# ---------------------------------------------------------------- product check (G5)
def product_check(Pfull):
    with read_only_external():
        inputs = load_inputs()
    idx = inputs.index
    br = breadth()
    td = thrust_days(deemer_ratio(br.adv, br.dec))
    v = run_vol.target_weight(inputs.eq_ret)
    half = np.maximum(v.shift(2).fillna(0.), .5)
    full = pd.Series(1., index=idx)
    days = idx[idx >= '1993-04-01']
    dm = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0].reindex(days)
    rz = rev_zscore(revisions(inputs.fy1, inputs.fy2, detect_rolls(inputs.fy1, inputs.fy2), window=63)).reindex(days)
    s1 = states(days, (dm < 45) & (rz < 0), dm < 30, dm, None)
    w1 = to_weights(idx, s1)
    desired = desired_from_states(inputs, s1, v, half, td, full)
    P = Pfull.reindex(idx)
    ep = thrust_episodes(inputs.px, w1, in_triggers(P, w1))
    st = cut_starts(idx, out_triggers(P, w1))
    sl = slice('1999-02-01', None)
    cash = inputs.cash_ret
    base = sh_dd(run_w(idx, desired, inputs).loc[sl], cash)
    pin = sh_dd(run_w(idx, with_in(idx, desired, ep, half), inputs).loc[sl], cash)
    pout = sh_dd(run_w(idx, with_out(idx, desired, st), inputs).loc[sl], cash)
    n_in = int((ep.entry >= '1999-02-01').sum())
    n_out = int(sum(idx[k] >= pd.Timestamp('1999-02-01') for k in st))
    return dict(base=base, pin=pin, pout=pout, n_in=n_in, n_out=n_out)


# ---------------------------------------------------------------- relationship tables (information)
def relationship(df, P, a=None, b=None):
    tr = (1 + df.eq_ret.fillna(0.)).cumprod()
    px = df.px
    f21, f63 = tr.shift(-21) / tr - 1, tr.shift(-63) / tr - 1
    nxt = pd.concat([px.shift(-k) for k in range(1, 22)], axis=1).min(axis=1)
    worst = (nxt / px - 1).clip(upper=0.)
    T = pd.DataFrame({'q': pd.cut(P['rank'], [-.01, .2, .4, .6, .8, 1.], labels=['1 low', '2', '3', '4', '5 high']),
                      'up': P.F > 0, 'f21': f21, 'f63': f63, 'worst': worst}).loc[a:b].dropna()
    g = lambda d: pd.Series({'days': len(d), 'f21': d.f21.mean(), 'pos21': (d.f21 > 0).mean(),
                             'f63': d.f63.mean(), 'worst21': d.worst.mean()})
    by_q = T.groupby('q', observed=True).apply(g)
    by_qf = T.groupby(['q', 'up'], observed=True).apply(g)
    return by_q, by_qf, g(T)


def fmt_rel(name, by_q, by_qf, allr):
    L = [f'### {name}', '', '| Exposure quintile | Flow | Days | Next 21d return | Share up | Next 63d return | Worst fall in next 21d |',
         '|---|---|---:|---:|---:|---:|---:|', f'| all | all | {allr.days:,.0f} | {allr.f21:+.2%} | {allr.pos21:.0%} | {allr.f63:+.2%} | {allr.worst21:.2%} |']
    for q, r in by_q.iterrows():
        L.append(f'| {q} | all | {r.days:,.0f} | {r.f21:+.2%} | {r.pos21:.0%} | {r.f63:+.2%} | {r.worst21:.2%} |')
        for up in (True, False):
            if (q, up) in by_qf.index:
                s = by_qf.loc[(q, up)]
                L.append(f'| {q} | {"rising" if up else "falling"} | {s.days:,.0f} | {s.f21:+.2%} | {s.pos21:.0%} | {s.f63:+.2%} | {s.worst21:.2%} |')
    return L + ['']


# ---------------------------------------------------------------- main (one run)
def main():
    df = skeleton_inputs()
    idx = df.index
    dm = daily_dmas(df.px)
    w1, half, desired = skeleton(df, dm)
    P = proxies(df.px, df.eq_ret)
    first = desired.first_valid_index()
    cash = df.cash_ret
    halves = [('1929–1976', None, SPLIT), ('1977–end', '1977-01-01', None)]
    r_base = run_w(idx, desired, df).loc[first:]
    rng = np.random.default_rng(7)

    # P-in
    ep = thrust_episodes(df.px, w1, in_triggers(P, w1))
    r_in = run_w(idx, with_in(idx, desired, ep, half), df).loc[first:]
    rows = []
    for r in ep.itertuples():
        d_ = idx[r.i_entry + 2:r.i_end + 2]
        eq, cs = float((1 + df.eq_ret.reindex(d_)).prod() - 1), float((1 + cash.reindex(d_)).prod() - 1)
        rows.append(dict(entry=r.entry.date(), end=r.end.date(), reason=r.reason, rank=float(P['rank'].loc[r.entry]),
                         days=len(d_), sp=eq, tb=cs, right=eq > cs))
    Ein = pd.DataFrame(rows, columns=['entry', 'end', 'reason', 'rank', 'days', 'sp', 'tb', 'right'])
    rank_start = P['rank'].first_valid_index()
    pl = []
    for _ in range(N_PLACEBO):
        sp = placebo_spans(rng, len(ep), df.px, w1.where(idx >= rank_start, np.nan))
        fake = pd.DataFrame(sp, columns=['i_entry', 'i_end'])
        pl.append(sh_dd(run_w(idx, with_in(idx, desired, fake, half), df).loc[first:], cash)[0])
    p_in = float((np.array(pl) >= sh_dd(r_in, cash)[0]).mean())
    pl_in_med = float(np.median(pl))

    # P-out
    st = cut_starts(idx, out_triggers(P, w1))
    r_out = run_w(idx, with_out(idx, desired, st), df).loc[first:]
    rows = []
    for k in st:
        d_ = idx[k + 2:k + 2 + CUT_DAYS]
        eq = float((1 + df.eq_ret.reindex(d_)).prod() - 1)
        rows.append(dict(start=idx[k].date(), rank=float(P['rank'].iloc[k]), sp=eq, right=eq < 0))
    Eout = pd.DataFrame(rows, columns=['start', 'rank', 'sp', 'right'])
    elig = np.flatnonzero(((w1 == 1) & (idx >= rank_start)).to_numpy())
    elig = elig[elig + CUT_DAYS <= len(idx)]
    pl = []
    for _ in range(N_PLACEBO):
        fake = random_cuts(rng, len(st), elig, len(idx))
        pl.append(sh_dd(run_w(idx, with_out(idx, desired, fake), df).loc[first:], cash)[0])
    p_out = float((np.array(pl) >= sh_dd(r_out, cash)[0]).mean())
    pl_out_med = float(np.median(pl))

    prod = product_check(P)

    def gates(name, r_x, E, p_val, pl_med, g5):
        sb, db = sh_dd(r_base, cash)
        sx, dx = sh_dd(r_x, cash)
        hs = [(n, sh_dd(r_base.loc[a:b], cash)[0], sh_dd(r_x.loc[a:b], cash)[0]) for n, a, b in halves]
        n = len(E)
        G = [('G1 Sharpe better, full and both halves', f'full {sx:.3f} vs {sb:.3f}; ' + '; '.join(f'{h} {x:.3f} vs {b_:.3f}' for h, b_, x in hs),
              sx > sb and all(x > b_ for _, b_, x in hs)),
             ('G2 worst fall not deeper by > 2 pts', f'{dx:.1%} vs {db:.1%}', dx >= db - .02),
             ('G3 placebo p <= 0.10', f'p = {p_val:.3f} (placebo median Sharpe {pl_med:.3f})', p_val <= .10),
             ('G4 episodes right > half', f'{int(E.right.sum()) if n else 0} of {n}', n > 0 and E.right.mean() > .5),
             ('G5 real product (1999-02 → end) not worse', f'{g5[1][0]:.3f} vs {g5[0][0]:.3f} (worst fall {g5[1][1]:.1%} vs {g5[0][1]:.1%}; {g5[2]} episodes)',
              g5[1][0] >= g5[0][0])]
        if n < MIN_EP:
            v = 'TOO FEW TO CONCLUDE'
        else:
            v = 'PASS' if all(g[2] for g in G) else 'FAIL'
        L = [f'## {name}: **{v}**', '', '| Gate | Evidence | Result |', '|---|---|---|']
        L += [f'| {g} | {e} | {"PASS" if p else "FAIL"} |' for g, e, p in G]
        L += ['', f'| {r_base.index[0].date()} → {r_base.index[-1].date()} | Base | Base + {name.split()[0]} |', '|---|---:|---:|',
              f'| Sharpe | {sb:.2f} | {sx:.2f} |', f'| Return a year | {metrics(r_base.dropna(), cash.reindex(r_base.dropna().index))["cagr"]:.1%} | '
              f'{metrics(r_x.dropna(), cash.reindex(r_x.dropna().index))["cagr"]:.1%} |', f'| Worst fall | {db:.1%} | {dx:.1%} |', '']
        return v, L

    v_in, L_in = gates('P-in (forced selling over → buy)', r_in, Ein, p_in, pl_in_med, (prod['base'], prod['pin'], prod['n_in']))
    v_out, L_out = gates('P-out (crowded, selling starts → cut to 50%)', r_out, Eout, p_out, pl_out_med, (prod['base'], prod['pout'], prod['n_out']))

    L = ['# P: systematic-fund positioning, CTA + vol-target proxies (pre-registered, spec 6P, one run)', '',
         f'**P-in: {v_in}. P-out: {v_out}.** Consequence (declared before the run): PASS → proposed to the user; FAIL → rejected list, no change.', '',
         f'S&P 500 {idx[0].date()} → {idx[-1].date()}. Price-only skeleton of the product (DMAS keys daily with month lock, vol size, S at max(live, 50%), '
         '5-point sizes, trade next day if gap ≥ 10 points). Cash: 0% before 1954, T-bill after. 10 bp. Proxies rebuilt from S&P prices (no BNPP data). Code: `run_pos.py`.', '']
    L += L_in
    L += ['### P-in episodes', '', '| Entry | End | Reason | Exposure rank | Days | S&P | Cash |', '|---|---|---|---:|---:|---:|---:|']
    L += [f'| {r.entry} | {r.end} | {r.reason} | {r.rank:.2f} | {r.days} | {r.sp:+.1%} | {r.tb:+.1%} |' for r in Ein.itertuples()]
    L += [''] + L_out
    L += ['### P-out cuts', '', '| Start | Exposure rank | S&P over the cut |', '|---|---:|---:|']
    L += [f'| {r.start} | {r.rank:.2f} | {r.sp:+.1%} |' for r in Eout.itertuples()]
    L += ['', '## Relationship tables (information only, not traded)', '',
          'Exposure = CTA + vol-target proxy, ranked against its own previous 10 years. Flow = 5-day change. '
          'Forward windows overlap, so the day counts overstate the independent evidence.', '']
    for name, a, b in [('All years', None, None)] + halves:
        L += fmt_rel(name, *relationship(df, P, a, b))
    (HERE / 'RESULTS_pos.md').write_text('\n'.join(L))
    P.to_csv(HERE / 'out/pos_proxies.csv')
    print('\n'.join(L))


if __name__ == '__main__':
    if '--one-run' not in sys.argv:
        sys.exit('Pre-registered single run: pass --one-run.')
    main()
