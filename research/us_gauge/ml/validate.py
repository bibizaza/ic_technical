"""Step 5 (spec section 6): walk-forward blocks, purge/embargo checks, cluster weights, K, AUC, calibration,
Brier, permutation tests. The cluster (45-day chaining) is the unit everywhere."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from research.us_gauge.ml.models import bucket, delta_w, train_pi

COST_BP = 10.
N_PERM, PERM_SEED = 5000, 7
N_BOOT, BOOT_SEED = 5000, 7


@dataclass(frozen=True)
class Block:
    name: str
    start: str
    end: str
    train_cut: str          # training events: entry <= train_cut (126 + 63 trading days before the start)


BLOCKS = [Block('2000–2004', '2000-01-01', '2004-12-31', '1999-03-31'),
          Block('2005–2009', '2005-01-01', '2009-12-31', '2004-03-31'),
          Block('2010–2015', '2010-01-01', '2015-12-31', '2009-03-31'),
          Block('2016–2026', '2016-01-01', '2026-12-31', '2015-03-31')]


def cluster_weights(cluster) -> np.ndarray:
    """Weight of an event = 1 / (events of its cluster among the events passed)."""
    c = pd.Series(np.asarray(cluster))
    return (1. / c.map(c.value_counts())).to_numpy()


def block_of(t) -> pd.Series:
    t = pd.to_datetime(pd.Series(t))
    out = pd.Series(None, index=t.index, dtype=object)
    for b in BLOCKS:
        out[(t >= b.start) & (t <= b.end)] = b.name
    return out


def split(E: pd.DataFrame, block: Block, labelled):
    """Boolean masks (training, test) over E (study events). Training needs a label; test is every study event
    with an entry in the block (open events are dropped later, when labels exist)."""
    t = E.t
    labelled = np.asarray(labelled, bool)
    tr = (t <= block.train_cut).to_numpy() & labelled
    te = ((t >= block.start) & (t <= block.end)).to_numpy()
    return tr, te


def assert_split(E, tr, te, block: Block):
    """Spec 6.1: no cluster in both sets; every training event's label window (126) + 63 trading days closed
    before the block starts; training entries before the cut-off."""
    a, b = set(E.cluster[tr]), set(E.cluster[te])
    assert not a & b, f'{block.name}: clusters in training and test: {sorted(a & b)}'
    emb = pd.to_datetime(E.t_embargo[tr])
    assert emb.notna().all() and (emb < pd.Timestamp(block.start)).all(), f'{block.name}: embargo violated'
    assert (E.t[tr] <= pd.Timestamp(block.train_cut)).all()
    assert not (tr & te).any()


def walk_forward(E: pd.DataFrame, X: pd.DataFrame, y: pd.Series, factory) -> pd.DataFrame:
    """Pooled out-of-sample predictions. E, X, y aligned on the same index (study events); y NaN = open."""
    labelled = y.notna().to_numpy()
    out = []
    for b in BLOCKS:
        tr, te = split(E, b, labelled)
        assert_split(E, tr, te, b)
        w = cluster_weights(E.cluster[tr])
        m = factory().fit(X[tr], y[tr].astype(int), w)
        pi = train_pi(y[tr], w)
        out.append(pd.DataFrame({'p': m.predict(X[te]), 'pi': pi, 'block': b.name, 'n_train': int(tr.sum())},
                                index=E.index[te]))
    return pd.concat(out)


# ---------------- statistics ----------------

def wauc(y, p, w) -> float:
    """AUC with sample weights (ties count one half); equals sklearn roc_auc_score(sample_weight=w)."""
    y, p, w = (np.asarray(a, float) for a in (y, p, w))
    u, inv = np.unique(p, return_inverse=True)
    wp = np.bincount(inv, weights=w * (y == 1), minlength=len(u))
    wn = np.bincount(inv, weights=w * (y == 0), minlength=len(u))
    below = np.cumsum(wn) - wn
    return float((wp * (below + .5 * wn)).sum() / (wp.sum() * wn.sum()))


def calibration_slope(y, p, w) -> float:
    """Weighted logistic regression of y on logit(p), no penalty (Newton steps); returns the slope."""
    y, p, w = (np.asarray(a, float) for a in (y, p, w))
    x = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    A = np.c_[np.ones_like(x), x]
    beta = np.zeros(2)
    for _ in range(100):
        q = 1 / (1 + np.exp(-A @ beta))
        g = A.T @ (w * (y - q))
        Hm = (A * (w * q * (1 - q))[:, None]).T @ A
        step = np.linalg.solve(Hm, g)
        beta += step
        if np.abs(step).max() < 1e-12:
            break
    return float(beta[1])


def brier(y, p, w) -> float:
    y, p, w = (np.asarray(a, float) for a in (y, p, w))
    return float(np.average((p - y) ** 2, weights=w))


def logloss(y, p, w) -> float:
    y, p, w = (np.asarray(a, float) for a in (y, p, w))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def sizing_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Event level -> cluster level. df: cluster, block, dw, R. Cost = 10 bp per unit of weight traded on the
    weight difference, in at t+1 and out at x+1 (2 x |dw|)."""
    d = df.assign(cost=2 * COST_BP / 1e4 * df.dw.abs())
    c = d.groupby('cluster').agg(block=('block', 'first'), dw=('dw', 'mean'), R=('R', 'mean'), cost=('cost', 'mean'), n=('dw', 'size'))
    assert d.groupby('cluster').block.nunique().max() == 1, 'a cluster spans two blocks'
    return c.reset_index().sort_values(['block', 'cluster'], kind='stable').reset_index(drop=True)


def K_from_clusters(c: pd.DataFrame, dw=None) -> float:
    dw = c.dw.to_numpy() if dw is None else dw
    m = pd.Series(dw).groupby(c.block.to_numpy()).transform('mean').to_numpy()
    return float(np.mean((dw - m) * c.R.to_numpy() - c.cost.to_numpy()))


def K_stat(df: pd.DataFrame) -> float:
    """Spec 6.3: mean over clusters of (dw_c - mean dw of its block) x R_c, net of costs."""
    return K_from_clusters(sizing_frame(df))


def perm_K(df: pd.DataFrame, n=N_PERM, seed=PERM_SEED):
    """Cluster-level dw (with its cost) shuffled across clusters within each block. p = share K_perm >= K."""
    c = sizing_frame(df)
    real = K_from_clusters(c)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(c.block.to_numpy() == b) for b in sorted(c.block.unique())]
    dw, cost, R = c.dw.to_numpy(), c.cost.to_numpy(), c.R.to_numpy()
    m = pd.Series(dw).groupby(c.block.to_numpy()).transform('mean').to_numpy()
    null = np.empty(n)
    for i in range(n):
        perm = np.arange(len(c))
        for g in groups:
            perm[g] = rng.permutation(g)
        null[i] = np.mean((dw[perm] - m) * R - cost[perm])
    return real, float((null >= real - 1e-15).mean()), null


def cluster_level_p(df):
    return df.groupby('cluster').p.transform('mean').to_numpy()


def perm_auc(df: pd.DataFrame, n=N_PERM, seed=PERM_SEED):
    """Cluster-level p shuffled across clusters within each block; events take their cluster's p.
    Returns (event-level weighted AUC, cluster-level-p AUC used as the permutation comparator, p-value)."""
    d = df.reset_index(drop=True)
    w = cluster_weights(d.cluster)
    auc_event = wauc(d.y, d.p, w)
    pc = d.groupby('cluster').agg(p=('p', 'mean'), block=('block', 'first'))
    cl = pc.index.to_numpy()
    pos = pd.Series(np.arange(len(cl)), index=cl)[d.cluster].to_numpy()
    pcv = pc.p.to_numpy()
    real = wauc(d.y, pcv[pos], w)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(pc.block.to_numpy() == b) for b in sorted(pc.block.unique())]
    null = np.empty(n)
    for i in range(n):
        perm = np.arange(len(cl))
        for g in groups:
            perm[g] = rng.permutation(g)
        null[i] = wauc(d.y, pcv[perm][pos], w)
    return auc_event, real, float((null >= real - 1e-15).mean())


def bootstrap_logloss(df: pd.DataFrame, col_a='p', col_b='p_g', n=N_BOOT, seed=BOOT_SEED) -> float:
    """One-sided cluster bootstrap: share of draws where B's log-loss is not lower than A's."""
    d = df.reset_index(drop=True)
    cl = d.cluster.unique()
    rng = np.random.default_rng(seed)
    rows = {c: np.flatnonzero(d.cluster.to_numpy() == c) for c in cl}
    worse = 0
    for _ in range(n):
        pick = rng.choice(cl, size=len(cl), replace=True)
        ix = np.concatenate([rows[c] for c in pick])
        s = d.iloc[ix]
        w = cluster_weights(np.concatenate([[k] * len(rows[c]) for k, c in enumerate(pick)]))
        worse += logloss(s.y, s[col_b], w) >= logloss(s.y, s[col_a], w)
    return worse / n


def size_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add bucket and dw from p, pi (training base rate of the block) and v (live vol size at t)."""
    b = bucket(df.p.to_numpy(), df.pi.to_numpy())
    return df.assign(bucket=b, dw=delta_w(b, df.v.to_numpy()))


def too_few(df: pd.DataFrame):
    """Spec 7: fewer than 35 out-of-sample clusters with a label or fewer than 8 failed clusters (cluster-weighted
    count of stopped events = sum over clusters of the cluster's stop share)."""
    n_cl = df.cluster.nunique()
    failed = float(df.groupby('cluster').y.mean().sum())
    return (n_cl < 35 or failed < 8), n_cl, failed


def loyo(E: pd.DataFrame, X: pd.DataFrame, y: pd.Series, factory) -> pd.Series:
    """Report only: leave-one-calendar-year-out prediction for every labelled event (trains on later years too)."""
    lab = y.notna()
    yr = E.t.dt.year
    out = pd.Series(np.nan, index=E.index)
    for Y in sorted(yr[lab].unique()):
        tr = (lab & (yr != Y)).to_numpy()
        te = (lab & (yr == Y)).to_numpy()
        m = factory().fit(X[tr], y[tr].astype(int), cluster_weights(E.cluster[tr]))
        out[te] = m.predict(X[te])
    return out


def coef_sign_stability(E, X, y, factory, n=1000, seed=PERM_SEED) -> pd.DataFrame:
    """Report only: M-L fitted on all labelled events and on cluster bootstrap samples; share of samples with
    the same sign as the full fit."""
    lab = y.notna().to_numpy()
    Ei, Xi, yi = E[lab], X[lab], y[lab].astype(int)
    full = factory().fit(Xi, yi, cluster_weights(Ei.cluster)).coef
    cl = Ei.cluster.unique()
    rows = {c: np.flatnonzero(Ei.cluster.to_numpy() == c) for c in cl}
    rng = np.random.default_rng(seed)
    same = np.zeros(len(full))
    for _ in range(n):
        pick = rng.choice(cl, size=len(cl), replace=True)
        ix = np.concatenate([rows[c] for c in pick])
        cid = np.concatenate([[k] * len(rows[c]) for k, c in enumerate(pick)])
        m = factory().fit(Xi.iloc[ix], yi.iloc[ix], cluster_weights(cid))
        same += np.sign(m.coef.to_numpy()) == np.sign(full.to_numpy())
    return pd.DataFrame({'coef': full, 'same_sign_share': same / n})
