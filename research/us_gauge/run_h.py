"""H: HTEI re-entry layer — pre-registered in spec section 6h (commit 7d6d063). One run.

Product (spec 11b) vs product + H, from the first HTEI date. Gates G1-G5, placebo p.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from research.us_gauge import run_vol
from research.us_gauge.data import load_inputs
from research.us_gauge.run_s1 import read_only_external
from research.us_gauge.revisions import detect_rolls, revisions, rev_zscore
from research.us_gauge.run_fast import states, to_weights
from research.us_gauge.run_b import breadth, deemer_ratio, thrust_days, thrust_episodes, placebo_spans
from research.us_gauge.trading import net_returns
from research.us_gauge.evaluate import metrics
from research.us_gauge.run_c import sharpe
from research.us_gauge.core_slide_data import layer, act_on_change, desired_from_states

HERE = Path(__file__).resolve().parent
HTEI = Path.home() / 'Library/CloudStorage/Dropbox/Tools_In_Construction/HTEI/daily_signals/signals_history.csv'
BUY = 35.
N_PLACEBO = 1000


def htei_score() -> pd.Series:
    h = pd.read_csv(HTEI, parse_dates=['date']).set_index('date').sort_index()
    return h.us_score.dropna()


def run_product(idx, desired, inputs):
    w = act_on_change(idx, (desired * 20).round() / 20)[0]
    return w, net_returns(w, inputs.eq_ret, inputs.cash_ret)


def main():
    with read_only_external():
        inputs = load_inputs()
    idx = inputs.index
    br = breadth()
    td = thrust_days(deemer_ratio(br.adv, br.dec))
    v = run_vol.target_weight(inputs.eq_ret)
    live = v.shift(2)
    days = idx[idx >= '1993-04-01']
    dm = pd.read_csv(HERE / 'out/dmas_daily_1955.csv', index_col=0, parse_dates=True).iloc[:, 0].reindex(days)
    rz = rev_zscore(revisions(inputs.fy1, inputs.fy2, detect_rolls(inputs.fy1, inputs.fy2), window=63)).reindex(days)
    s1 = states(days, (dm < 45) & (rz < 0), dm < 30, dm, None)
    w1 = to_weights(idx, s1)
    desired = desired_from_states(inputs, s1, v, live, td)

    score = htei_score()
    sc = score.reindex(idx)
    trig = idx[(sc < BUY).to_numpy()]
    ep = thrust_episodes(inputs.px, w1, trig)
    wh = layer(idx, ep, live)

    start = score.index[0]
    w_p, r_p = run_product(idx, desired, inputs)
    w_h, r_h = run_product(idx, pd.concat([desired, wh.shift(-2)], axis=1).max(axis=1), inputs)
    sl = slice(start, None)
    rp, rh = r_p.loc[sl].dropna(), r_h.loc[sl].dropna()
    rh = rh.reindex(rp.index)
    cash = inputs.cash_ret.reindex(rp.index)
    mp, mh = metrics(rp, cash), metrics(rh, cash)
    shp, shh = sharpe(rp, cash), sharpe(rh, cash)

    # placebo: episodes moved to random cash days, same mechanics, same sizing and execution
    rng = np.random.default_rng(7)
    ep_in = ep[ep.entry >= start]
    pl = []
    for _ in range(N_PLACEBO):
        sp = placebo_spans(rng, len(ep_in), inputs.px.loc[:], w1.where(idx >= start, np.nan))
        fake = pd.DataFrame(sp, columns=['i_entry', 'i_end'])
        wpl = layer(idx, fake, live)
        _, r_ = run_product(idx, pd.concat([desired, wpl.shift(-2)], axis=1).max(axis=1), inputs)
        pl.append(sharpe(r_.reindex(rp.index), cash))
    pl = np.array(pl)
    p_val = float((pl >= shh).mean())

    # episodes
    rows = []
    for r in ep_in.itertuples():
        d_ = idx[r.i_entry + 2:r.i_end + 2]
        eq = float((1 + inputs.eq_ret.reindex(d_)).prod() - 1)
        cs = float((1 + inputs.cash_ret.reindex(d_)).prod() - 1)
        rows.append(dict(entry=r.entry.date(), end=r.end.date(), reason=r.reason, htei=float(sc.loc[r.entry]),
                         days=len(d_), sp500=eq, tbill=cs, beat=eq > cs))
    E = pd.DataFrame(rows)
    halves = [('2010-2017', None, '2017-12-31'), ('2018-2026', '2018-01-01', None)]
    hs = [(n, sharpe(rp.loc[a:b], cash.loc[a:b]), sharpe(rh.loc[a:b], cash.loc[a:b])) for n, a, b in halves]
    gates = [('G1 Sharpe product+H > product', f'{shh:.3f} vs {shp:.3f}', shh > shp),
             ('G2 worst fall not worse by > 2 pts', f'{mh["maxdd"]:.1%} vs {mp["maxdd"]:.1%}', mh['maxdd'] >= mp['maxdd'] - .02),
             ('G3 placebo p <= 0.10', f'p = {p_val:.3f} (placebo median {np.median(pl):.3f})', p_val <= .10),
             ('G4 more than half of episodes beat cash', f'{int(E.beat.sum()) if len(E) else 0} of {len(E)}', len(E) > 0 and E.beat.mean() > .5),
             ('G5 both halves not worse', '; '.join(f'{n}: {b:.3f} vs {a:.3f}' for n, a, b in hs), all(b >= a for _, a, b in hs))]
    verdict = all(g[2] for g in gates)
    # where H changed the weight in force
    diff_days = int((w_h.loc[sl] - w_p.loc[sl]).abs().gt(1e-9).sum())
    L = ['# H: HTEI re-entry (pre-registered, spec 6h, one run)', '',
         f'**Verdict: {"PASS" if verdict else "FAIL"}**', '',
         f'Sample {rp.index[0].date()} → {rp.index[-1].date()}. HTEI US score < {BUY:g} while the keys are in cash. '
         'Episode exits as B. Size = live vol target. Product = spec 11b (daily keys, 5-pt sizes, trade next day if gap >= 10 pts). '
         f'10 bp costs. HTEI history: backtest to 2026-04-12, live after, so mostly in-sample. Code: `run_h.py`.', '',
         '| Gate | Evidence | Result |', '|---|---|---|']
    L += [f'| {n} | {e} | {"PASS" if p else "FAIL"} |' for n, e, p in gates]
    L += ['', '| Since 2010 | Product | Product + H |', '|---|---:|---:|',
          f'| Sharpe | {shp:.2f} | {shh:.2f} |', f'| Return a year | {mp["cagr"]:.1%} | {mh["cagr"]:.1%} |',
          f'| Worst fall | {mp["maxdd"]:.1%} | {mh["maxdd"]:.1%} |', f'| Days with a different weight | | {diff_days} |', '',
          '## Episodes', '', '| Entry | End | Reason | HTEI at entry | Days | S&P 500 | T-bill |', '|---|---|---|---:|---:|---:|---:|']
    L += [f'| {r.entry} | {r.end} | {r.reason} | {r.htei:.1f} | {r.days} | {r.sp500:+.1%} | {r.tbill:+.1%} |' for r in E.itertuples()]
    L += ['', f'Cash days with HTEI score available: {int(((w1 == 0) & sc.notna()).loc[sl].sum())}. '
          f'Of those, HTEI < {BUY:g}: {int(((w1 == 0) & (sc < BUY)).loc[sl].sum())}.', '']
    (HERE / 'RESULTS_h.md').write_text('\n'.join(L))
    E.to_csv(HERE / 'out/h_episodes.csv', index=False)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
