"""Small statistics used by the audit: Wilson intervals, MCC from confusion matrices (vectorised over bootstrap
replicates), the train-optimal single-feature threshold, quantile-bin rules and fold assignments."""
import numpy as np


def wilson(k, n, z=1.959964):
    """Wilson 95% interval for a proportion k / n."""
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def share(mask):
    m = np.asarray(mask, bool)
    n = len(m)
    k = int(m.sum())
    return dict(value=float(k / n) if n else None, ci=wilson(k, n), k=k, n=int(n))


def mcc_from_confusion(C):
    """Matthews correlation (multiclass, Gorodkin) of confusion matrices C (..., K, K) with rows = truth."""
    C = np.asarray(C, np.float64)
    t = C.sum(-1)
    p = C.sum(-2)
    c = np.trace(C, axis1=-2, axis2=-1)
    s = C.sum((-2, -1))
    num = c * s - (t * p).sum(-1)
    den = np.sqrt(np.maximum(s * s - (p * p).sum(-1), 0) * np.maximum(s * s - (t * t).sum(-1), 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    return out


def encode_labels(y, classes):
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut.get(v, -1) for v in y], np.int64)


def confusion(yi, pi, K):
    ok = (yi >= 0) & (pi >= 0)
    return np.bincount(yi[ok] * K + pi[ok], minlength=K * K).reshape(K, K)


def score(yi, pi, K, metric='mcc'):
    """Main metric of class-index predictions: mcc, acc or f1 (macro)."""
    if len(yi) == 0:
        return None
    C = confusion(yi, pi, K)
    return float(_metric_from_C(C[None], metric)[0])


def _metric_from_C(C, metric):
    if metric == 'mcc':
        return mcc_from_confusion(C)
    tot = C.sum((-2, -1))
    if metric == 'acc':
        return np.trace(C, axis1=-2, axis2=-1) / np.maximum(tot, 1)
    if metric == 'f1':
        tp = np.diagonal(C, axis1=-2, axis2=-1).astype(float)
        fp = C.sum(-2) - tp
        fn = C.sum(-1) - tp
        with np.errstate(invalid='ignore', divide='ignore'):
            f = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
        present = (C.sum(-1) + C.sum(-2)) > 0
        return (f * present).sum(-1) / np.maximum(present.sum(-1), 1)
    raise ValueError(metric)


def bootstrap_ci(yi, pi, K, metric='mcc', n_boot=1000, seed=0, batch=100):
    """Percentile 95% interval of the metric over bootstrap resamples of items."""
    n = len(yi)
    if n == 0 or n_boot <= 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    cat = yi * K + pi
    ok = (yi >= 0) & (pi >= 0)
    cat = np.where(ok, cat, K * K)
    vals = []
    for b0 in range(0, n_boot, batch):
        B = min(batch, n_boot - b0)
        idx = rng.integers(0, n, size=(B, n))
        flat = (np.arange(B)[:, None] * (K * K + 1) + cat[idx]).ravel()
        cnt = np.bincount(flat, minlength=B * (K * K + 1)).reshape(B, K * K + 1)[:, :K * K].reshape(B, K, K)
        vals.append(_metric_from_C(cnt, metric))
    v = np.concatenate(vals)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def best_threshold(x, y):
    """Train-optimal rule 'positive iff d * x > d * t' for binary y (0/1), by MCC over every cut point.
    Returns (t, d, mcc)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.int64)
    n = len(x)
    P = int(y.sum())
    N = n - P
    if n == 0 or P == 0 or N == 0:
        return 0.0, 1, 0.0
    o = np.argsort(x, kind='stable')
    xs, ys = x[o], y[o]
    last = np.r_[np.flatnonzero(xs[1:] != xs[:-1]), n - 1]   # last index of each distinct value
    cpos = np.cumsum(ys)[last]          # positives with x <= value
    cneg = (last + 1) - cpos
    # d = +1: predict positive iff x > value
    tp = P - cpos
    fp = N - cneg
    fn = cpos
    tn = cneg
    m_up = _mcc_counts(tp, fp, fn, tn)
    # d = -1: predict positive iff x < next value, i.e. x <= value
    m_dn = _mcc_counts(cpos, cneg, P - cpos, N - cneg)
    i_up = int(np.argmax(m_up))
    i_dn = int(np.argmax(m_dn))
    vals = xs[last]
    if m_up[i_up] >= m_dn[i_dn]:
        return float(vals[i_up]), 1, float(m_up[i_up])
    nxt = vals[i_dn + 1] if i_dn + 1 < len(vals) else vals[i_dn] + 1.0
    return float((vals[i_dn] + nxt) / 2.0), -1, float(m_dn[i_dn])


def _mcc_counts(tp, fp, fn, tn):
    tp, fp, fn, tn = (np.asarray(a, np.float64) for a in (tp, fp, fn, tn))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)


def apply_threshold(x, t, d):
    x = np.asarray(x, np.float64)
    return (x > t).astype(np.int64) if d == 1 else (x < t).astype(np.int64)


def fit_bins(x, yi, K, nb=64):
    """Quantile bins of x with the majority class per bin (multiclass one-feature rule). A point mass at the maximum
    (the two highest quantiles equal) gets a bin of its own (rules v1.2.1; before, it shared the last bin with the
    values below it, so a feature with two values gave a single bin)."""
    q = np.quantile(x, np.linspace(0, 1, nb + 1))
    edges = np.unique(q)
    if q[-2] == q[-1]:
        edges = np.append(edges, np.inf)
    b = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, max(len(edges) - 2, 0))
    glob = int(np.argmax(np.bincount(yi, minlength=K)))
    nbins = max(len(edges) - 1, 1)
    cnt = np.zeros((nbins, K), np.int64)
    np.add.at(cnt, (b, yi), 1)
    lab = np.where(cnt.sum(1) > 0, cnt.argmax(1), glob)
    return edges, lab


def apply_bins(x, edges, lab):
    b = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, max(len(edges) - 2, 0))
    return lab[b]


def auroc(y, s):
    """AUROC of score s for binary y (0/1) with midranks; None if one class is missing."""
    y = np.asarray(y, np.int64)
    s = np.asarray(s, np.float64)
    P = int(y.sum())
    N = len(y) - P
    if P == 0 or N == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def auroc_ci(y, s, n_boot=1000, seed=0):
    """Percentile 95% interval of the AUROC over bootstrap resamples of items (None when one class is missing)."""
    y = np.asarray(y, np.int64)
    s = np.asarray(s, np.float64)
    if n_boot <= 0 or y.min() == y.max():
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        a = auroc(y[i], s[i])
        if a is not None:
            vals.append(a)
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def group_folds(groups, k=5, seed=0):
    """Fold of every item; items with the same group share a fold; groups dealt by a seeded permutation of the
    sorted unique group labels (explicit order, independent of library tie rules)."""
    g = np.asarray(groups)
    labels, inv = np.unique(g, return_inverse=True)
    perm = np.random.default_rng(seed).permutation(len(labels))
    return (perm % k)[inv.ravel()].astype(np.int64)
