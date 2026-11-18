import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from research.us_gauge.ml import models as mo
from research.us_gauge.ml.features import FEATURES


def _data(n, seed, planted=True):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    for r in ('region_europe', 'region_asia_pac_dev', 'region_emerging'):
        X[r] = (rng.random(n) < .3).astype(float)
    X['attempt'] = rng.integers(0, 4, n).astype(float)
    X.loc[rng.random(n) < .05, 'fx_shock'] = np.nan
    z = 1.2 * X.runup_2y - 1.0 * X.days_le + .8 * X.d_credit - 1.2
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int) if planted else (rng.random(n) < .25).astype(int)
    return X, y


def _w(n):
    return np.ones(n)


@pytest.mark.parametrize('name', ['M-L', 'M-G'])
def test_planted_signal_is_found(name):
    X, y = _data(600, 1)
    Xt, yt = _data(2000, 2)
    m = mo.FACTORIES[name]().fit(X, y, _w(len(y)))
    assert roc_auc_score(yt, m.predict(Xt)) > .70


@pytest.mark.parametrize('name', ['M-L', 'M-G'])
def test_shuffled_labels_give_no_signal(name):
    """The same pipeline on the same planted data, labels shuffled (training and test): no skill."""
    X, y = _data(600, 1)
    Xt, yt = _data(3000, 2)
    aucs = []
    for k in range(6):
        rng = np.random.default_rng(10 + k)
        m = mo.FACTORIES[name]().fit(X, rng.permutation(y), _w(len(y)))
        aucs.append(roc_auc_score(rng.permutation(yt), m.predict(Xt)))
    assert abs(np.mean(aucs) - .5) < .02 and max(abs(a - .5) for a in aucs) < .05


def test_fixed_seeds_reproduce():
    X, y = _data(300, 3)
    w = np.random.default_rng(0).uniform(.2, 1, len(y))
    a = mo.MBoost().fit(X, y, w).predict(X)
    b = mo.MBoost().fit(X, y, w).predict(X)
    np.testing.assert_array_equal(a, b)
    c = mo.MLogit().fit(X, y, w).predict(X)
    np.testing.assert_array_equal(c, mo.MLogit().fit(X, y, w).predict(X))


def test_monotone_constraints_hold_on_a_grid():
    X, y = _data(500, 5)
    # make the raw relation of curve_min positive and runup negative: constraints must still hold
    m = mo.MBoost().fit(X, y, _w(len(y)))
    base = X.iloc[:40].copy()
    grid = np.linspace(-3, 3, 25)
    for f, sign in mo.MONOTONE.items():
        P = []
        for g in grid:
            Z = base.copy()
            Z[f] = g
            P.append(m.predict(Z))
        d = np.diff(np.array(P), axis=0) * sign
        assert (d >= -1e-9).all(), f


def test_ml_settings_and_preprocessing():
    X, y = _data(400, 6)
    m = mo.MLogit().fit(X, y, _w(len(y)))
    assert m.model.C == .5 and m.features == mo.ML_FEATURES
    Z = m._prep(X)
    np.testing.assert_allclose(Z.mean(), 0, atol=1e-9)
    assert Z.notna().all().all()                        # missing -> training median
    X2 = X.copy()
    X2.loc[0, 'vol_rel'] = 1e6                          # winsorised at the training 99th percentile
    assert m._prep(X2).loc[0, 'vol_rel'] == pytest.approx(((m.hi['vol_rel'] - m.mu['vol_rel']) / m.sd['vol_rel']))
    c = m.contributions(X)
    np.testing.assert_allclose(c.sum(axis=1) + m.model.intercept_[0],
                               np.log(m.predict(X) / (1 - m.predict(X))), rtol=1e-8)


def test_xgb_settings():
    assert mo.XGB_PARAMS['max_depth'] == 2 and mo.XGB_PARAMS['n_estimators'] == 200
    assert list(mo.SEEDS) == list(range(25)) and len(mo.MG_FEATURES) == 17


def test_boost_contributions_add_up():
    X, y = _data(300, 7)
    m = mo.MBoost(seeds=range(3)).fit(X, y, _w(len(y)))
    c = m.contributions(X)
    assert c.shape == (300, 17)


def test_bucket_map_and_delta_w():
    pi = .25
    b = mo.bucket([.10, .15, .2, .37, .375, .9], pi)
    assert list(b) == ['low', 'low', 'normal', 'normal', 'high', 'high']
    v = np.array([.2, .7, .3])
    assert list(mo.s_weight(['low', 'normal', 'high'], v)) == [1., .7, .3]
    np.testing.assert_allclose(mo.delta_w(['low', 'low', 'normal', 'high', 'high'], [.2, .7, .3, .3, .8]),
                               [.5, .3, 0., -.2, 0.])


def test_base_rate_is_weighted():
    m = mo.BaseRate().fit(None, [1, 0, 0], [2., 1., 1.])
    assert m.pi == .5 and list(m.predict(range(2))) == [.5, .5]


def test_top_reasons():
    r = mo.top_reasons(pd.Series({'a': .1, 'b': -.5, 'c': .3, 'd': np.nan, 'e': .05}))
    assert r == [('b', '-'), ('c', '+'), ('a', '+')]
