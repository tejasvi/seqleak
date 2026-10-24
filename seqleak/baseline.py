"""Position-only baseline models (row zero) with genome-grouped cross-validation.

The default model is a logistic regression on B-spline bases of the position features, with separate splines for
same-strand and opposite-strand pairs; ``model='gbm'`` uses a histogram gradient-boosting classifier on the raw
features instead. Model scores can be appended (``extra``) to fit the position + model model whose gain over the
position model is :func:`seqleak.metrics.gain_over_position`.

Specs (``spec=``)
-----------------
``'adjacent'``: neighbouring pairs. [same_strand, spline(clip(gap, -150, 1500), 12 knots) x same, x opposite];
    knots refitted on each training fold; C = 10; probability output.
``'pair'``: any pairs. Splines of log1p(genes_apart) (10 knots), log1p(min(max(gap, 0), 1e6)) (8) and
    log1p(max_intervening_gap) (8), each x same and x opposite strand, one-hot strand switches (0, 1, 2, 3, 4+),
    same_strand; knots fitted once on every 50th item (features only, no labels); C = 1; each training fold
    subsampled to every 4th item; decision-function output.
``'operon_v1'``, ``'ppi_v1'``: the same designs fitted with the v1 solver settings (lbfgs, tol 1e-4); they reproduce
    the v1 code bit for bit on a given machine, but that fit stops before convergence on this ill-conditioned basis
    and moves by up to ~4e-4 AUROC between machines; ``'adjacent'`` / ``'pair'`` use newton-cholesky with tol 1e-8.
``'gene'``: single genes. Splines of the numeric gene features present (``length``, ``dist_origin``, ``density``,
    ``gap_up``, ``gap_down``, ``operon_rule``, ``directon_size``) and one-hot ``strand``, ``replication_strand``,
    ``directon_pos``. Gene length is a property of the gene, not of its layout: this preset includes it as a
    gene-level covariate, so row zero, the gain over position and the position share then count length as
    position. For layout only, pass the columns without length (generic splines, prognostic strata; the columns
    must have no missing values):
    ``row_zero(task, position_columns=['strand', 'replication_strand', 'dist_origin', 'directon_pos',
    'directon_size', 'operon_rule', 'density'])``.
``'auto'``: from the columns (``genes_apart`` varies -> pair; neighbouring pairs -> adjacent; else gene).
A list of terms (:class:`Linear`, :class:`OneHot`, :class:`Spline`) gives a custom design.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .bootstrap import factorize_sorted

TRANSFORMS = {
    'identity': lambda x: x,
    'log1p': np.log1p,
    'log1p_gap': lambda x: np.log1p(np.minimum(np.maximum(x, 0), 10 ** 6)),
    'slog1p': lambda x: np.sign(x) * np.log1p(np.abs(x)),
    'clip_gap': lambda x: np.clip(x, -150, 1500),
}


class Linear:
    def __init__(self, col):
        self.col, self.fit_on = col, None

    def columns(self, d, rows):
        return [d[self.col][rows].astype(np.float64)[:, None]]


class OneHot:
    """Indicators of ``levels``, plus one column for any other value (``other=True``)."""

    def __init__(self, col, levels, other=True):
        self.col, self.levels, self.other, self.fit_on = col, list(levels), other, None

    def columns(self, d, rows):
        v = d[self.col][rows]
        cols = [v == lv for lv in self.levels]
        if self.other:
            cols.append(~np.isin(v, self.levels))
        return [np.stack(cols, 1).astype(np.float64)]


class Spline:
    """Cubic B-spline basis (sklearn ``SplineTransformer``, quantile knots) of a transformed column, optionally
    split by a 0/1 column (``by``) into basis * by and basis * (1 - by).

    ``fit_on='train'`` refits the knots on each training fold; ``'all'`` fits them once on every ``stride``-th item
    of the whole table (features only)."""

    def __init__(self, col, n_knots, transform='identity', by=None, fit_on='train', stride=1, extrapolation='constant'):
        self.col, self.n_knots, self.transform, self.by = col, n_knots, transform, by
        self.fit_on, self.stride, self.extrapolation = fit_on, stride, extrapolation

    def _v(self, d, rows):
        f = TRANSFORMS[self.transform] if isinstance(self.transform, str) else self.transform
        return f(d[self.col][rows].astype(np.float64)).reshape(-1, 1)

    def fit(self, d, rows):
        self.spl = SplineTransformer(n_knots=self.n_knots, degree=3, knots='quantile',
                                     extrapolation=self.extrapolation).fit(self._v(d, rows[::self.stride]))
        return self

    def columns(self, d, rows):
        B = self.spl.transform(self._v(d, rows))
        if self.by is None:
            return [B]
        s = d[self.by][rows].astype(np.float64).reshape(-1, 1)
        return [B * s, B * (1 - s)]


def _spec(spec, features):
    """(terms, defaults) for a named spec."""
    cols = set(features.columns)
    if spec == 'auto':
        if 'genes_apart' in cols and features['genes_apart'].nunique() > 1:
            spec = 'pair'
        elif {'gap', 'same_strand'} <= cols:
            spec = 'adjacent'
        else:
            spec = 'gene'
    exact = dict(solver='lbfgs', tol=1e-4)              # the v1 settings (reproduce the v1 code exactly)
    converged = dict(solver='newton-cholesky', tol=1e-8)  # second-order, converged: machine-independent to ~1e-8
    if spec in ('adjacent', 'operon_v1'):
        terms = [Linear('same_strand'), Spline('gap', 12, 'clip_gap', by='same_strand', fit_on='train')]
        return terms, dict(C=10.0, max_iter=5000, output='proba', train_stride=1,
                           **(exact if spec == 'operon_v1' else converged))
    if spec in ('pair', 'ppi_v1'):
        terms = [Spline('genes_apart', 10, 'log1p', by='same_strand', fit_on='all', stride=50),
                 Spline('gap', 8, 'log1p_gap', by='same_strand', fit_on='all', stride=50),
                 Spline('max_intervening_gap', 8, 'log1p', by='same_strand', fit_on='all', stride=50),
                 OneHot('strand_switches', [0, 1, 2, 3]), Linear('same_strand')]
        return terms, dict(C=1.0, max_iter=3000, output='decision', train_stride=4,
                           **(exact if spec == 'ppi_v1' else converged))
    if spec == 'gene':
        terms = []
        for c, nk, tf in [('length', 8, 'log1p'), ('dist_origin', 6, 'identity'), ('density', 6, 'log1p'),
                          ('gap_up', 8, 'slog1p'), ('gap_down', 8, 'slog1p'), ('operon_rule', 5, 'identity'),
                          ('directon_size', 5, 'log1p')]:
            if c in cols and features[c].notna().all() and features[c].nunique() > nk:
                terms.append(Spline(c, nk, tf, fit_on='train'))
        for c, lv in [('strand', [1]), ('replication_strand', [1]), ('directon_pos', [0, 1, 2])]:
            if c in cols and features[c].notna().all() and features[c].nunique() > 1:
                terms.append(OneHot(c, lv, other=True))
        if not terms:
            raise ValueError('no usable per-gene position features')
        return terms, dict(C=1.0, max_iter=3000, output='decision', train_stride=1, **converged)
    if spec == 'generic':
        return generic_terms(features), dict(C=1.0, max_iter=3000, output='decision', train_stride=1, **converged)
    if isinstance(spec, (list, tuple)):
        return list(spec), dict(C=1.0, max_iter=3000, output='decision', train_stride=1, **converged)
    raise ValueError(f'unknown spec {spec!r}')


def generic_terms(features, max_levels=10, n_knots=8):
    """Terms for arbitrary position columns: indicators for columns with <= max_levels values (bool, categorical,
    small integers), cubic splines for the other numeric columns (log1p for non-negative long-tailed columns, signed
    log1p for long-tailed columns with negative values). Columns with missing values are not supported."""
    terms = []
    for c in features.columns:
        v = features[c]
        if v.isna().any():
            raise ValueError(f'position column {c!r} has missing values')
        u = pd.unique(v)
        if len(u) <= 1:
            continue
        if len(u) <= max_levels or not np.issubdtype(v.dtype, np.number) or v.dtype == bool:
            levels = sorted(u.tolist(), key=str)
            terms.append(OneHot(c, levels[:-1], other=True))
            continue
        x = v.to_numpy(np.float64)
        q1, q50, q99 = np.quantile(x, [0.01, 0.5, 0.99])
        tf = 'identity'
        if x.min() >= 0 and q99 > 10 * max(q50, 1e-9):
            tf = 'log1p'
        elif x.min() < 0 and max(abs(q1), abs(q99)) > 10 * max(abs(q50), 1e-9):
            tf = 'slog1p'
        terms.append(Spline(c, n_knots, tf, fit_on='train'))
    if not terms:
        raise ValueError('no varying position columns')
    return terms


def make_folds(groups, folds=5, seed=0, order=None):
    """Fold id of every item, grouped by genome.

    ``folds`` = int k: genomes are assigned to folds by ``numpy.random.default_rng(seed).permutation(G) % k`` over the
    sorted genome labels; ``'logo'``: one fold per genome; ``'contigK'`` (e.g. ``'contig5'``, for single-genome
    tasks): each genome's items, in genome order (``order``, a permutation; default the item order), are cut into K
    contiguous chunks; an array: used as is."""
    if not isinstance(folds, (int, np.integer, str)):
        return np.asarray(folds)
    g, labels = factorize_sorted(groups)
    if folds == 'logo':
        return g
    if isinstance(folds, str) and folds.startswith('contig'):
        K = int(folds[6:] or 5)
        order = np.arange(len(g)) if order is None else np.asarray(order)
        f = np.empty(len(g), np.int64)
        go = g[order]
        for k in range(len(labels)):
            idx = order[go == k]
            for q, part in enumerate(np.array_split(idx, K)):
                f[part] = q
        return f
    rng = np.random.default_rng(seed)
    return (rng.permutation(len(labels)) % int(folds))[g]


def position_model(features, y, groups, *, spec='auto', extra=None, folds=5, seed=0, train_mask=None,
                   model='logistic', C=None, max_iter=None, train_stride=None, max_train=None, output=None,
                   solver=None, tol=None, chunk=20_000_000, order=None, return_info=False, verbose=False):
    """Out-of-fold predictions of a position-only model (or position + ``extra`` columns), folds grouped by genome.

    Parameters
    ----------
    features : DataFrame with the position feature columns of the spec.
    y : labels (0/1; NaN or other values = unlabelled, never used for training).
    groups : genome of each item (folds never split a genome).
    spec : see module docstring.
    extra : optional array (n,) or (n, m) appended to the design (model scores). Items with non-finite extra values
        are not used for training and get NaN predictions.
    folds : int k (grouped k-fold), ``'logo'``, ``'contigK'`` (contiguous chunks of each genome along ``order``),
        or an array of fold ids.
    train_mask : optional boolean array; if given, items outside it are never used for training (e.g. a benchmark's
        test split evaluated by cross-validation, or ``split == 'train'`` with ``folds=None`` to fit once on the
        training split and predict every item).
    model : ``'logistic'`` or ``'gbm'`` (``HistGradientBoostingClassifier`` on the raw features + extra).
    C, max_iter, train_stride, output, solver, tol : override the spec defaults (``output`` 'proba' or
        'decision'; ``solver``/``tol`` of ``LogisticRegression``: the v1 presets use lbfgs with tol 1e-4 as the v1
        code did, all other specs newton-cholesky with tol 1e-8, which converges to the same optimum on any machine).
    max_train : cap on training items per fold (random subsample after the stride).

    Returns an array of predictions (and, with ``return_info``, a dict with the fitted models per fold)."""
    n = len(features)
    y = np.asarray(y, np.float64)
    terms, dflt = _spec(spec, features)
    C = dflt['C'] if C is None else C
    max_iter = dflt['max_iter'] if max_iter is None else max_iter
    output = dflt['output'] if output is None else output
    train_stride = dflt['train_stride'] if train_stride is None else train_stride
    solver = dflt['solver'] if solver is None else solver
    tol = dflt['tol'] if tol is None else tol
    need = sorted({t.col for t in terms} | {t.by for t in terms if getattr(t, 'by', None)})
    d = {c: features[c].to_numpy() for c in need}
    if extra is not None:
        extra = np.asarray(extra, np.float64)
        extra = extra.reshape(n, -1)
    usable = np.isin(y, (0, 1))
    if extra is not None:
        ok_extra = np.isfinite(extra).all(1)
        usable &= ok_extra
    else:
        ok_extra = np.ones(n, bool)
    if train_mask is not None:
        usable &= np.asarray(train_mask, bool)
    for t in terms:
        if getattr(t, 'fit_on', None) == 'all':
            t.fit(d, np.arange(n))
    if folds is None:
        fold = np.zeros(n, np.int64)
        fold_list = [None]
    else:
        fold = make_folds(groups, folds, seed, order)
        fold_list = list(np.unique(fold))
    rng = np.random.default_rng(seed)
    pred = np.full(n, np.nan)
    info = []
    for q in fold_list:
        if q is None:
            tr = np.flatnonzero(usable)
            te = np.flatnonzero(ok_extra)
        else:
            tr = np.flatnonzero((fold != q) & usable)
            te = np.flatnonzero((fold == q) & ok_extra)
        sub = tr[::train_stride]
        if max_train is not None and len(sub) > max_train:
            sub = np.sort(rng.choice(sub, max_train, replace=False))
        if len(np.unique(y[sub])) < 2 or len(te) == 0:
            continue
        if model == 'gbm':
            from sklearn.ensemble import HistGradientBoostingClassifier
            raw = np.column_stack([d[c].astype(np.float64) for c in need])
            Xtr = raw[sub] if extra is None else np.hstack([raw[sub], extra[sub]])
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
            clf.fit(Xtr, y[sub].astype(int))
            for c0 in range(0, len(te), chunk):
                rows = te[c0:c0 + chunk]
                X = raw[rows] if extra is None else np.hstack([raw[rows], extra[rows]])
                pred[rows] = clf.predict_proba(X)[:, 1]
            info.append(dict(fold=q, n_train=len(sub), model=clf))
            continue
        for t in terms:
            if getattr(t, 'fit_on', None) == 'train':
                t.fit(d, tr)
        Xtr = _design(terms, d, sub, extra)
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=max_iter, C=C, solver=solver, tol=tol).fit(sc.transform(Xtr), y[sub].astype(int))
        del Xtr
        for c0 in range(0, len(te), chunk):
            rows = te[c0:c0 + chunk]
            X = sc.transform(_design(terms, d, rows, extra))
            pred[rows] = lr.decision_function(X) if output == 'decision' else lr.predict_proba(X)[:, 1]
        if verbose:
            print(f'fold {q}: trained on {len(sub)} items, predicted {len(te)}', flush=True)
        info.append(dict(fold=q, n_train=len(sub), coef=lr.coef_.ravel().copy(), intercept=float(lr.intercept_[0])))
    return (pred, info) if return_info else pred


def _design(terms, d, rows, extra):
    cols = []
    for t in terms:
        cols += t.columns(d, rows)
    if extra is not None:
        cols.append(extra[rows])
    return np.hstack(cols)
