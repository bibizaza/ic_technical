from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score
from research.us_gauge.ml import validate as va
from research.us_gauge.ml.events import cluster_ids
from research.us_gauge.ml.models import BaseRate, MLogit

OUT = Path(__file__).resolve().parents[1] / 'research/us_gauge/ml/out'


def _noise_frame(rng, n_clusters=45):
    rows = []
    blocks = [b.name for b in va.BLOCKS]
    for c in range(n_clusters):
        b = blocks[c % 4]
        k = rng.integers(1, 6)
        p = rng.uniform(.05, .6)
        for _ in range(k):
            rows.append(dict(cluster=c, block=b, p=np.clip(p + rng.normal(0, .05), .01, .99), pi=.25,
                             v=rng.uniform(.1, 1.), y=int(rng.random() < .25), R=rng.normal(0, .1)))
    return va.size_events(pd.DataFrame(rows))


def test_wauc_equals_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.round(rng.random(300), 2)            # ties
    w = rng.uniform(.1, 1, 300)
    assert va.wauc(y, p, w) == pytest.approx(roc_auc_score(y, p, sample_weight=w))


def test_calibration_slope_and_brier():
    rng = np.random.default_rng(1)
    p = rng.uniform(.05, .95, 20000)
    y = (rng.random(20000) < p).astype(int)
    w = np.ones_like(p)
    assert va.calibration_slope(y, p, w) == pytest.approx(1., abs=.06)
    q = 1 / (1 + np.exp(-2 * np.log(p / (1 - p))))   # over-confident forecast -> slope near 0.5
    assert va.calibration_slope(y, q, w) == pytest.approx(.5, abs=.05)
    assert va.brier(y, p, w) < va.brier(y, np.full_like(p, y.mean()), w)


def test_cluster_weights():
    np.testing.assert_allclose(va.cluster_weights([3, 3, 5, 3]), [1 / 3, 1 / 3, 1., 1 / 3])


def test_K_definition_by_hand():
    df = pd.DataFrame(dict(cluster=[0, 0, 1, 2], block=['a', 'a', 'a', 'b'], dw=[.5, .3, 0., -.2], R=[.1, .3, -.1, .05]))
    # cluster 0: dw .4, R .2, cost 2*.001*.4; cluster 1: dw 0, R -.1; block a mean dw .2; cluster 2 alone in block b
    k = np.mean([(.4 - .2) * .2 - .0008, (0 - .2) * -.1 - 0., (-.2 + .2) * .05 - .0004])
    assert va.K_stat(df) == pytest.approx(k)


def test_holding_more_everywhere_scores_zero_before_costs():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(dict(cluster=np.arange(40), block=np.repeat(['a', 'b'], 20), dw=.5, R=rng.normal(.05, .1, 40)))
    assert va.K_stat(df) == pytest.approx(-.001)


def test_permutation_p_uniform_on_noise():
    """200 synthetic noise runs: the K and AUC permutation p-values are uniform."""
    pk, pa = [], []
    for s in range(200):
        df = _noise_frame(np.random.default_rng(100 + s))
        pk.append(va.perm_K(df, n=199, seed=s)[1])
        pa.append(va.perm_auc(df, n=199, seed=s)[2])
    for p in (np.array(pk), np.array(pa)):
        assert stats.kstest(p, 'uniform').pvalue > .01
        assert .01 <= (p <= .05).mean() <= .10
        assert .4 <= p.mean() <= .6


def test_permutation_finds_planted_sizing_skill():
    rng = np.random.default_rng(3)
    df = _noise_frame(rng, 60)
    df['R'] = np.where(df.bucket == 'low', .08, np.where(df.bucket == 'high', -.08, 0.)) + rng.normal(0, .02, len(df))
    k, p, _ = va.perm_K(df, n=999)
    assert k > 0 and p < .01


def _synthetic_events(seed=0):
    rng = np.random.default_rng(seed)
    t = pd.to_datetime('1980-01-01') + pd.to_timedelta(np.sort(rng.integers(0, 46 * 365, 300)), 'D')
    E = pd.DataFrame({'t': t})
    E['cluster'] = cluster_ids(E.t, 45)
    E['t_embargo'] = E.t + pd.Timedelta(days=275)          # about 189 trading days
    return E


def test_walk_forward_no_cluster_in_both_and_embargo():
    E = _synthetic_events()
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(len(E), 6)),
                     columns=['runup_2y', 'attempt', 'days_le', 'vol_rel', 'fx_shock', 'd_credit'])
    y = pd.Series((np.random.default_rng(2).random(len(E)) < .25).astype(float))
    wf = va.walk_forward(E, X, y, MLogit)
    for b in va.BLOCKS:
        tr, te = va.split(E, b, y.notna())
        assert not set(E.cluster[tr]) & set(E.cluster[te])
        assert (E.t_embargo[tr] < pd.Timestamp(b.start)).all()
        assert (E.t[tr] <= pd.Timestamp(b.train_cut)).all()
        assert set(wf.index[wf.block == b.name]) == set(E.index[te])
    assert wf.index.is_unique and (wf.p > 0).all()


def test_embargo_violation_is_caught():
    E = _synthetic_events()
    E.loc[E.t <= '1999-03-31', 't_embargo'] = pd.Timestamp('2000-02-01')
    y = pd.Series(0., index=E.index)
    y[::4] = 1.
    with pytest.raises(AssertionError):
        va.walk_forward(E, pd.DataFrame({'a': np.zeros(len(E))}), y, BaseRate)


def test_shared_cluster_is_caught():
    E = _synthetic_events()
    b = va.BLOCKS[0]
    tr, te = va.split(E, b, np.ones(len(E), bool))
    E.loc[np.flatnonzero(te)[0], 'cluster'] = E.cluster[tr].iloc[0]
    with pytest.raises(AssertionError):
        va.assert_split(E, tr, te, b)


def test_blocks_on_the_real_event_list():
    """Spec 6.1 table: test events and clusters per block, embargo and cluster separation on the frozen list."""
    E = pd.read_csv(OUT / 'events.csv', parse_dates=['t', 't_embargo'])
    S = E[E.study].reset_index(drop=True)
    expect = {'2000–2004': (28, 10), '2005–2009': (29, 7), '2010–2015': (31, 10), '2016–2026': (71, 18)}
    for b in va.BLOCKS:
        tr, te = va.split(S, b, np.ones(len(S), bool))
        va.assert_split(S, tr, te, b)
        assert (te.sum(), S.cluster[te].nunique()) == expect[b.name]
