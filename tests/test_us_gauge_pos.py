import numpy as np
import pandas as pd
from research.us_gauge import run_pos
from research.us_gauge.core_slide_data import act_on_change


def test_trailing_rank_excludes_current_and_needs_minimum():
    e = pd.Series([1., 2., 3., 0.5, 4.])
    r = run_pos.trailing_rank(e, n=3, n_min=2)
    assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[1])
    assert r.iloc[2] == 1.0            # 3 > 1, 2
    assert r.iloc[3] == 0.0            # 0.5 < 1, 2, 3
    assert r.iloc[4] == 1.0            # window = 2, 3, 0.5


def test_proxies_caps_and_signs():
    idx = pd.bdate_range('2000-01-03', periods=400)
    rng = np.random.default_rng(0)
    ret = pd.Series(.002 + rng.normal(0, .002, 400), index=idx)    # strong uptrend, very low vol
    px = 100 * (1 + ret).cumprod()
    P = run_pos.proxies(px, ret)
    last = P.iloc[-1]
    assert last.vt == 1.5                                          # 0.10 / tiny vol, capped
    assert abs(last.cta - 2.0) < 1e-9                              # all three trends clipped at +1, leverage capped at 2
    assert abs(last.E - 3.5) < 1e-9


def test_cut_starts_ignore_triggers_inside_a_cut():
    idx = pd.bdate_range('2000-01-03', periods=100)
    st = run_pos.cut_starts(idx, idx[[5, 10, 25, 26, 60]])
    assert st == [5, 26, 60]
    c = run_pos.cap_series(idx, st)
    assert c.iloc[5] == .5 and c.iloc[25] == .5 and c.iloc[4] == 1. and c.iloc[47] == 1. and c.iloc[46] == .5


def test_fast_act_equals_product_rule():
    idx = pd.bdate_range('2000-01-03', periods=600)
    rng = np.random.default_rng(3)
    d = pd.Series((rng.uniform(0, 1, 600) * 20).round() / 20, index=idx)
    d.iloc[rng.choice(600, 60)] = 0.
    d.iloc[:7] = np.nan
    a = act_on_change(idx, d)[0]
    b = run_pos.fast_act(idx, d)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_random_cuts_do_not_overlap():
    rng = np.random.default_rng(1)
    s = run_pos.random_cuts(rng, 5, np.arange(0, 500), 600)
    assert len(s) == 5 and (np.diff(s) >= 21).all()


def test_with_in_ignores_episode_on_last_day():
    idx = pd.bdate_range('2000-01-03', periods=50)
    d = pd.Series(0., index=idx)
    ep = pd.DataFrame({'i_entry': [10, 49], 'i_end': [20, 49]})
    out = run_pos.with_in(idx, d, ep, pd.Series(.5, index=idx))
    assert out.iloc[10:20].eq(.5).all() and out.iloc[20:].eq(0.).all()
