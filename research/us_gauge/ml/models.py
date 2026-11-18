"""Step 4 (spec section 5): M-L (gate), M-G (challenger), benchmarks, bucket map, contributions. No tuning.

Every model is fitted with cluster weights w = 1 / (events in the cluster) passed as sample weights.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.us_gauge.ml.features import FEATURES

ML_FEATURES = ['runup_2y', 'attempt', 'days_le', 'vol_rel', 'fx_shock', 'd_credit']
MG_FEATURES = list(FEATURES)                       # the 15 features; region = 3 dummies (17 columns)
MONOTONE = {'runup_2y': 1, 'd_credit': 1, 'curve_min': -1}
XGB_PARAMS = dict(max_depth=2, n_estimators=200, learning_rate=0.03, min_child_weight=3, subsample=0.7,
                  colsample_bytree=0.7, reg_lambda=5, objective='binary:logistic', eval_metric='logloss')
SEEDS = range(25)
LOW, HIGH = .6, 1.5                                 # bucket multipliers of pi, fixed by the spec
EXPECTED_SIGNS = {'runup_2y': '+', 'attempt': '+', 'days_le': '-', 'fx_shock': '+', 'd_credit': '+', 'vol_rel': 'open'}
SINGLE_RULES = ['days_le', 'vol_rel', 'runup_2y']


class MLogit:
    """Median impute -> winsorise at the training 1st/99th percentile -> standardise -> L2 logistic, C = 0.5."""

    def __init__(self, features=ML_FEATURES, C=.5):
        self.features, self.C = list(features), C

    def _prep(self, X):
        Z = X[self.features].astype(float).fillna(self.med)
        Z = Z.clip(self.lo, self.hi, axis=1)
        return (Z - self.mu) / self.sd

    def fit(self, X, y, w):
        A = X[self.features].astype(float)
        self.med = A.median()
        A = A.fillna(self.med)
        self.lo, self.hi = A.quantile(.01), A.quantile(.99)
        A = A.clip(self.lo, self.hi, axis=1)
        self.mu, self.sd = A.mean(), A.std(ddof=0).replace(0., 1.)
        self.model = LogisticRegression(C=self.C, solver='lbfgs', max_iter=10_000, tol=1e-10)
        self.model.fit(((A - self.mu) / self.sd).to_numpy(), np.asarray(y, float), sample_weight=np.asarray(w, float))
        return self

    def predict(self, X):
        return self.model.predict_proba(self._prep(X).to_numpy())[:, 1]

    @property
    def coef(self):
        return pd.Series(self.model.coef_[0], index=self.features)

    def contributions(self, X):
        """Coefficient x standardised value (log-odds), one column per feature."""
        return self._prep(X) * self.coef


class MBoost:
    """xgboost, all features, monotone where the economics is clear, mean of 25 seeds; native missing values."""

    def __init__(self, features=MG_FEATURES, seeds=SEEDS):
        self.features, self.seeds = list(features), list(seeds)

    def fit(self, X, y, w):
        from xgboost import XGBClassifier
        mono = tuple(MONOTONE.get(f, 0) for f in self.features)
        A = X[self.features].astype(float).to_numpy()
        self.models = []
        for s in self.seeds:
            m = XGBClassifier(**XGB_PARAMS, monotone_constraints=mono, random_state=s, n_jobs=1)
            m.fit(A, np.asarray(y, int), sample_weight=np.asarray(w, float))
            self.models.append(m)
        return self

    def predict(self, X):
        A = X[self.features].astype(float).to_numpy()
        return np.mean([m.predict_proba(A)[:, 1] for m in self.models], axis=0)

    def contributions(self, X):
        """Tree SHAP (pred_contribs, log-odds), mean over seeds; last column (bias) dropped."""
        from xgboost import DMatrix
        D = DMatrix(X[self.features].astype(float).to_numpy())
        c = np.mean([m.get_booster().predict(D, pred_contribs=True) for m in self.models], axis=0)
        return pd.DataFrame(c[:, :-1], columns=self.features, index=X.index)


class BaseRate:
    """Training base rate (cluster-weighted stop share), no features."""

    def fit(self, X, y, w):
        self.pi = float(np.average(np.asarray(y, float), weights=np.asarray(w, float)))
        return self

    def predict(self, X):
        return np.full(len(X), self.pi)


def single_rule(feature):
    """One-feature benchmark: the M-L pipeline on that feature alone."""
    return MLogit([feature])


FACTORIES = {'M-L': MLogit, 'M-G': MBoost, 'base rate': BaseRate,
             **{f'only {f}': (lambda f=f: single_rule(f)) for f in SINGLE_RULES}}


def train_pi(y, w) -> float:
    """pi = stop share in the training events (cluster-weighted, like everything else)."""
    return float(np.average(np.asarray(y, float), weights=np.asarray(w, float)))


def bucket(p, pi):
    p = np.asarray(p, float)
    return np.where(p <= LOW * pi, 'low', np.where(p >= HIGH * pi, 'high', 'normal'))


def s_weight(b, v):
    """S weight during the episode: low 100%; normal max(v, 50%) (spec 11d); high the live size v."""
    v = np.asarray(v, float)
    b = np.asarray(b)
    return np.where(b == 'low', 1., np.where(b == 'high', v, np.maximum(v, .5)))


def delta_w(b, v):
    """Model S weight - 11d S weight (spec 6.3)."""
    v = np.asarray(v, float)
    return s_weight(b, v) - np.maximum(v, .5)


def top_reasons(contrib_row: pd.Series, k=3) -> list:
    """The k largest contributions to the score (absolute size), with their sign."""
    c = contrib_row.dropna()
    top = c.abs().sort_values(ascending=False).index[:k]
    return [(f, '+' if c[f] > 0 else '-') for f in top]
