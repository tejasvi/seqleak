"""Rank metrics with gene position held fixed, gains over a position model, and the position share.

Conventions
-----------
* ``y``: binary labels (1 positive, 0 negative). ``score``: higher = more likely positive. Items with a
  non-finite score are dropped (and counted in ``info['n_dropped']``).
* ``groups``: genome of each item. ``strata``: position stratum of each item (``seqleak.strata``).
* Comparisons: a positive and a negative are compared only if they share the stratum and (default
  ``compare='within_group'``) the genome. ``compare='across_groups'`` also compares items of different genomes.
* Aggregation: ``aggregate='mean'`` (default; per-genome metric averaged over genomes, the BacBench convention) or
  ``'pooled'`` (one ratio over all genomes, i.e. genomes weighted by their positive x negative pairs).
* Within-stratum AUROC = sum over strata of Mann-Whitney U / sum over strata of n_pos x n_neg, i.e. the per-stratum
  AUROCs averaged with weights n_pos x n_neg. Ties count 1/2. With one stratum it is the ordinary AUROC.

Bootstrap
---------
Intervals come from a :class:`seqleak.bootstrap.Resample`. Statistics are computed from sufficient statistics, not
by re-scoring items:

* cluster bootstrap with within-genome comparisons: per-(genome, stratum) U and n_pos x n_neg, aggregated per genome;
  a replicate is a weighted sum over genomes (exact; O(genomes) per replicate);
* block bootstrap (or comparisons across genomes): for every comparison domain, the matrices
  M[u, v] = sum over strata of U(positives of unit u, negatives of unit v) and D[u, v] = sum over strata of
  n_pos(u) n_neg(v); a replicate with unit multiplicities c has numerator c'Mc and denominator c'Dc, exactly the value
  obtained by re-scoring the resampled items (copies of a block are compared with each other, as when the resampled
  items are pooled).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .bootstrap import Resample, factorize_sorted, percentile_ci

__all__ = ['MetricResult', 'within_auc', 'auroc', 'auprc', 'within_auprc', 'gain_over_position', 'explained_share',
           'explained_share_ovr', 'position_share', 'position_share_bound', 'equalized_score',
           'position_share_result', 'within_auc_ovr', 'concordance', 'mann_whitney_cells']


@dataclass
class MetricResult:
    name: str
    estimate: float
    ci: tuple = (np.nan, np.nan)
    replicates: np.ndarray = None
    per_group: pd.DataFrame = None
    n_items: int = 0
    n_pos: int = 0
    n_groups: int = 0
    info: dict = field(default_factory=dict)

    @property
    def lo(self):
        return self.ci[0]

    @property
    def hi(self):
        return self.ci[1]

    def to_dict(self):
        d = dict(name=self.name, estimate=self.estimate, ci_low=self.ci[0], ci_high=self.ci[1], n_items=self.n_items,
                 n_pos=self.n_pos, n_groups=self.n_groups)
        d.update({k: v for k, v in self.info.items() if np.isscalar(v) or isinstance(v, (list, tuple, str))})
        return d

    def __str__(self):
        return f'{self.name} {self.estimate:.4f} [{self.ci[0]:.4f}, {self.ci[1]:.4f}] (n={self.n_items}, groups={self.n_groups})'


# ----------------------------------------------------------------------------------------------- sufficient statistics
def mann_whitney_cells(y, score, cell, n_cells):
    """Per-cell positives P, negatives N and Mann-Whitney U (positives over negatives, ties 1/2).

    ``cell``: int array (n,) with values in 0..n_cells-1. O(n log n) (one lexsort)."""
    n = len(y)
    if n == 0:
        z = np.zeros(n_cells)
        return z, z.copy(), z.copy()
    order = np.lexsort((score, cell))
    c = cell[order]
    s = score[order]
    yy = y[order].astype(np.float64)
    brk = np.empty(n, bool)
    brk[0] = True
    brk[1:] = c[1:] != c[:-1]
    starts = np.flatnonzero(brk)
    sizes = np.diff(np.append(starts, n))
    rank = np.arange(1, n + 1, dtype=np.float64) - np.repeat(starts, sizes)
    tie = brk.copy()
    tie[1:] |= s[1:] != s[:-1]
    rs = np.flatnonzero(tie)
    re = np.append(rs[1:], n) - 1
    mid = 0.5 * (rank[rs] + rank[re])
    r = np.repeat(mid, re - rs + 1)
    P = np.bincount(c, weights=yy, minlength=n_cells)
    T = np.bincount(c, minlength=n_cells).astype(np.float64)
    R = np.bincount(c, weights=r * yy, minlength=n_cells)
    U = R - P * (P + 1) / 2
    return P, T - P, U


def _unit_matrices(y, score, stratum, unit_local, n_local):
    """M[u, v] and D[u, v] for one comparison domain (see module docstring). unit_local in 0..n_local-1."""
    k, _ = factorize_sorted(stratum)
    K = int(k.max()) + 1 if len(k) else 1
    yb = y.astype(bool)
    idx = unit_local * K + k
    Pk = np.bincount(idx, weights=yb, minlength=n_local * K).reshape(n_local, K)
    Tk = np.bincount(idx, minlength=n_local * K).reshape(n_local, K).astype(np.float64)
    D = Pk @ (Tk - Pk).T
    r, _ = factorize_sorted(score)
    R = int(r.max()) + 1 if len(r) else 1
    key = k * R + r
    kp, up, base = key[yb], unit_local[yb], k[yb] * R
    nk, nu = key[~yb], unit_local[~yb]
    o = np.lexsort((nk, nu))
    nk, nu = nk[o], nu[o]
    bounds = np.searchsorted(nu, np.arange(n_local + 1))
    M = np.zeros((n_local, n_local))
    for v in range(n_local):
        seg = nk[bounds[v]:bounds[v + 1]]
        if len(seg) == 0 or len(kp) == 0:
            continue
        lo = np.searchsorted(seg, kp, 'left')
        hi = np.searchsorted(seg, kp, 'right')
        b0 = np.searchsorted(seg, base, 'left')
        M[:, v] = np.bincount(up, weights=(lo - b0) + 0.5 * (hi - lo), minlength=n_local)
    return M, D


def _domain_replicates(y, score, stratum, unit, rs, compare):
    """Numerators and denominators (B, n_domains) of the replicates, by quadratic forms over units."""
    dom_of_unit = rs.unit_group if compare == 'within_group' else np.zeros(rs.n_units, np.int64)
    item_dom = dom_of_unit[unit]
    C = rs.counts
    nums, dens, labels = [], [], []
    order = np.argsort(item_dom, kind='stable')
    bounds = np.searchsorted(item_dom[order], np.arange(dom_of_unit.max() + 2))
    for d in range(int(dom_of_unit.max()) + 1):
        units_d = np.flatnonzero(dom_of_unit == d)
        ii = order[bounds[d]:bounds[d + 1]]
        if len(units_d) == 0:
            continue
        local = np.full(rs.n_units, -1, np.int64)
        local[units_d] = np.arange(len(units_d))
        M, D = _unit_matrices(y[ii], score[ii], stratum[ii], local[unit[ii]], len(units_d))
        Cd = C[:, units_d].astype(np.float64)
        nums.append(np.einsum('bu,bu->b', Cd @ M, Cd))
        dens.append(np.einsum('bu,bu->b', Cd @ D, Cd))
        labels.append(d)
    return np.stack(nums, 1), np.stack(dens, 1)


def _aggregate_rep(num, den, aggregate):
    if aggregate == 'pooled':
        with np.errstate(invalid='ignore', divide='ignore'):
            return num.sum(1) / den.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        a = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    with np.errstate(invalid='ignore'):
        cnt = np.isfinite(a).sum(1)
        return np.where(cnt > 0, np.nansum(a, 1) / np.maximum(cnt, 1), np.nan)


# ------------------------------------------------------------------------------------------------------------- helpers
def _prepare(y, score, strata, groups, resample, n_boot, method, block_size, seed, order=None):
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    n = len(y)
    if score.shape[0] != n:
        raise ValueError('y and score have different lengths')
    groups = np.zeros(n, np.int64) if groups is None else np.asarray(groups)
    strata = np.zeros(n, np.int64) if strata is None else np.asarray(strata)
    if resample is None and n_boot:
        resample = Resample(groups, method=method, n_boot=n_boot, block_size=block_size, seed=seed, order=order)
    if resample is not None:
        if len(resample.group_of_item) != n:
            raise ValueError('resample was built for a different number of items')
        g_code, g_labels = resample.group_of_item, resample.group_labels
    else:
        g_code, g_labels = factorize_sorted(groups)
    ok = np.isfinite(score) & np.isin(y, (0, 1))
    return y, score, strata, g_code, g_labels, resample, ok


def _cells_auc(y, s, k, g, G):
    """Per-group U and W = sum_k P N over (group, stratum) cells, plus per-group counts."""
    k_code, k_lab = factorize_sorted(k)
    K = len(k_lab) if len(k_lab) else 1
    key = g * K + k_code
    cell, cell_keys = pd.factorize(key)
    P, N, U = mann_whitney_cells(y, s, cell.astype(np.int64), len(cell_keys))
    cg = (np.asarray(cell_keys) // K).astype(np.int64)
    U_g = np.bincount(cg, weights=U, minlength=G)
    W_g = np.bincount(cg, weights=P * N, minlength=G)
    n_g = np.bincount(g, minlength=G)
    p_g = np.bincount(g, weights=y.astype(np.float64), minlength=G)
    used = np.bincount(cg, weights=((P > 0) & (N > 0)).astype(np.float64), minlength=G)
    return U_g, W_g, n_g, p_g, used, (P, N, U, cell_keys, K)


# -------------------------------------------------------------------------------------------------------------- AUROC
def within_auc(y, score, strata=None, groups=None, *, aggregate='mean', compare='within_group', resample=None,
               n_boot=0, method='cluster', block_size=50, seed=0, level=0.95, order=None, name=None):
    """AUROC with position held fixed: positives are compared only with negatives of the same stratum (and genome).

    Weighted by n_pos x n_neg per stratum. ``strata=None`` gives the ordinary AUROC. Returns a :class:`MetricResult`
    whose ``per_group`` table has one row per genome (n, n_pos, compared pairs, value). Bootstrap: pass a shared
    ``resample`` (preferred, makes intervals of derived quantities paired) or ``n_boot`` > 0."""
    if aggregate not in ('mean', 'pooled'):
        raise ValueError("aggregate must be 'mean' or 'pooled'")
    if compare not in ('within_group', 'across_groups'):
        raise ValueError("compare must be 'within_group' or 'across_groups'")
    name = name or ('within_auroc' if strata is not None else 'auroc')
    y, s, k, g, g_lab, rs, ok = _prepare(y, score, strata, groups, resample, n_boot, method, block_size, seed, order)
    G = len(g_lab)
    yy, ss, kk, gg = y[ok].astype(np.int8), s[ok], k[ok], g[ok]
    if compare == 'within_group':
        U_g, W_g, n_g, p_g, used, _ = _cells_auc(yy, ss, kk, gg, G)
        with np.errstate(invalid='ignore', divide='ignore'):
            A_g = np.where(W_g > 0, U_g / np.where(W_g > 0, W_g, 1), np.nan)
        valid = W_g > 0
        est = float(np.mean(A_g[valid])) if aggregate == 'mean' and valid.any() else \
            float(U_g.sum() / W_g.sum()) if W_g.sum() > 0 else np.nan
        per = pd.DataFrame(dict(group=g_lab, n=n_g, n_pos=p_g.astype(np.int64), cells=used.astype(np.int64),
                                pairs=W_g, value=A_g))
    else:
        U_g, W_g, n_g, p_g, used, (P, N, U, ck, K) = _cells_auc(yy, ss, kk, np.zeros(len(yy), np.int64), 1)
        est = float(U.sum() / (P * N).sum()) if (P * N).sum() > 0 else np.nan
        per = None
    rep = None
    ci = (np.nan, np.nan)
    if rs is not None:
        if rs.method == 'cluster' and compare == 'within_group':
            Cf = rs.counts.astype(np.float64)
            if aggregate == 'mean':
                with np.errstate(invalid='ignore', divide='ignore'):
                    rep = (Cf @ np.where(valid, A_g, 0.0)) / (Cf @ valid.astype(np.float64))
            else:
                with np.errstate(invalid='ignore', divide='ignore'):
                    rep = (Cf @ U_g) / (Cf @ W_g)
        else:
            unit = rs.unit_of_item[ok]
            num, den = _domain_replicates(yy, ss, kk, unit, rs, compare)
            rep = _aggregate_rep(num, den, aggregate if compare == 'within_group' else 'pooled')
        ci = percentile_ci(rep, level)
    info = dict(aggregate=aggregate, compare=compare, n_dropped=int((~ok).sum()), n_valid_groups=int(np.sum(W_g > 0)),
                method=None if rs is None else rs.method, n_boot=0 if rs is None else rs.n_boot)
    return MetricResult(name, est, ci, rep, per, int(ok.sum()), int(yy.sum()), int(len(np.unique(gg))), info)


def auroc(y, score, groups=None, **kw):
    """Ordinary AUROC (per genome, averaged over genomes by default); same options as :func:`within_auc`."""
    return within_auc(y, score, None, groups, **kw)


# -------------------------------------------------------------------------------------------------------------- AUPRC
def _ap_sorted(yy, ww, gg, G, ss):
    """Average precision per group for items sorted by (group, -score); weights ww (multiplicities).
    sklearn semantics: precision at each distinct threshold, weighted by the recall increment."""
    n = len(yy)
    if n == 0:
        return np.full(G, np.nan)
    tp = yy * ww
    fp = (1 - yy) * ww
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    gstart = np.flatnonzero(np.r_[True, gg[1:] != gg[:-1]])
    gid = gg[gstart]
    sizes = np.diff(np.append(gstart, n))
    base_tp = np.where(gstart > 0, ctp[gstart - 1], 0.0)
    base_fp = np.where(gstart > 0, cfp[gstart - 1], 0.0)
    off_tp = np.repeat(base_tp, sizes)
    off_fp = np.repeat(base_fp, sizes)
    TP = ctp - off_tp
    FP = cfp - off_fp
    end = np.r_[(gg[1:] != gg[:-1]) | (ss[1:] != ss[:-1]), True]
    e = np.flatnonzero(end)
    TPe, FPe, ge = TP[e], FP[e], gg[e]
    prev = np.r_[0.0, TPe[:-1]]
    first_in_group = np.r_[True, ge[1:] != ge[:-1]]
    prev[first_in_group] = 0.0
    dtp = TPe - prev
    with np.errstate(invalid='ignore', divide='ignore'):
        prec = np.where(TPe + FPe > 0, TPe / np.where(TPe + FPe > 0, TPe + FPe, 1), 0.0)
    num = np.bincount(ge, weights=dtp * prec, minlength=G)
    Pg = np.zeros(G)
    Pg[gid] = TP[np.append(gstart[1:], n) - 1]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(Pg > 0, num / np.where(Pg > 0, Pg, 1), np.nan)


def _cells(g, fold):
    """Comparison cells: the group, or (group, fold) when ``fold`` is given (out-of-fold predictions of different fold
    models are never compared). Returns (cell code per item, number of cells, group of each cell)."""
    if fold is None:
        G = int(g.max()) + 1 if len(g) else 0
        return g, G, np.arange(G)
    f = factorize_sorted(fold)[0]
    F = int(f.max()) + 1 if len(f) else 1
    cell, keys = pd.factorize(g * F + f)
    return cell.astype(np.int64), len(keys), (np.asarray(keys) // F).astype(np.int64)


def _ap_to_groups(ap_c, pos_c, neg_c, cell_group, G):
    """Per-group AP from per-cell AP: cells with both classes, weighted by their positives."""
    valid = np.isfinite(ap_c) & (pos_c > 0) & (neg_c > 0)
    num = np.bincount(cell_group, weights=np.where(valid, ap_c * pos_c, 0.0), minlength=G)
    den = np.bincount(cell_group, weights=np.where(valid, pos_c, 0.0), minlength=G)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _ap_groups_sorted(y_o, w, c_o, s_o, nc, cg, G):
    """Per-group AP for items sorted by (cell, -score) with weights w."""
    ap_c = _ap_sorted(y_o, w, c_o, nc, s_o)
    pos_c = np.bincount(c_o, weights=w * y_o, minlength=nc)
    neg_c = np.bincount(c_o, weights=w * (1 - y_o), minlength=nc)
    return _ap_to_groups(ap_c, pos_c, neg_c, cg, G)


def _ap_per_cell(y, s, cell, n_cells, w=None):
    order = np.lexsort((-s, cell))
    yy = y[order].astype(np.float64)
    ww = np.ones(len(yy)) if w is None else w[order].astype(np.float64)
    return _ap_sorted(yy, ww, cell[order], n_cells, s[order])


def auprc(y, score, groups=None, *, fold=None, resample=None, n_boot=0, method='cluster', block_size=50, seed=0,
          level=0.95, order=None, name='auprc'):
    """Average precision (sklearn's step-wise definition) per genome, averaged over genomes, with bootstrap CI.

    ``fold`` (out-of-fold predictions whose folds split genomes): AP per (genome, fold), averaged within the genome
    with weights = positives, so that predictions of different fold models are never ranked against each other.
    Cluster bootstrap resamples the per-genome values; block bootstrap re-scores each replicate with multiplicity
    weights (exact, O(n) per replicate)."""
    y, s, k, g, g_lab, rs, ok = _prepare(y, score, None, groups, resample, n_boot, method, block_size, seed, order)
    G = len(g_lab)
    yy, ss, gg = y[ok].astype(np.int8), s[ok], g[ok]
    cell, nc, cg = _cells(gg, None if fold is None else np.asarray(fold)[ok])
    o = np.lexsort((-ss, cell))
    y_o, c_o, s_o = yy[o].astype(np.float64), cell[o], ss[o]
    ap = _ap_groups_sorted(y_o, np.ones(len(o)), c_o, s_o, nc, cg, G)
    valid = np.isfinite(ap)
    est = float(np.nanmean(ap)) if valid.any() else np.nan
    per = pd.DataFrame(dict(group=g_lab, n=np.bincount(gg, minlength=G),
                            n_pos=np.bincount(gg, weights=yy, minlength=G).astype(np.int64), value=ap))
    rep, ci = None, (np.nan, np.nan)
    if rs is not None:
        if rs.method == 'cluster':
            Cf = rs.counts.astype(np.float64)
            with np.errstate(invalid='ignore', divide='ignore'):
                rep = (Cf @ np.where(valid, ap, 0.0)) / (Cf @ valid.astype(np.float64))
        else:
            u_o = rs.unit_of_item[ok][o]
            rep = np.empty(rs.n_boot)
            for b in range(rs.n_boot):
                a = _ap_groups_sorted(y_o, rs.counts[b, u_o].astype(np.float64), c_o, s_o, nc, cg, G)
                rep[b] = np.nanmean(np.where(valid, a, np.nan))
        ci = percentile_ci(rep, level)
    info = dict(n_dropped=int((~ok).sum()), method=None if rs is None else rs.method, n_boot=0 if rs is None else rs.n_boot,
                fold_stratified=fold is not None)
    return MetricResult(name, est, ci, rep, per, int(ok.sum()), int(yy.sum()), int(valid.sum()), info)


def within_auprc(y, score, strata, groups=None, *, normalize=True, resample=None, n_boot=0, method='cluster',
                 block_size=50, seed=0, level=0.95, name='within_auprc'):
    """Average precision with position held fixed (experimental; AUROC is the recommended within-position metric).

    Per (genome, stratum) cell: AP and prevalence pi. With ``normalize=True`` each cell contributes the AP lift
    (AP - pi) / (1 - pi) (0 = chance, 1 = perfect), so that strata with different prevalence are comparable; cells
    are averaged with weights n_pos x n_neg within a genome, then genomes are averaged. Cluster bootstrap only."""
    y, s, k, g, g_lab, rs, ok = _prepare(y, score, strata, groups, resample, n_boot, 'cluster', block_size, seed)
    if rs is not None and rs.method != 'cluster':
        raise NotImplementedError('within_auprc supports the cluster bootstrap only')
    G = len(g_lab)
    yy, ss, kk, gg = y[ok].astype(np.int8), s[ok], k[ok], g[ok]
    k_code, k_lab = factorize_sorted(kk)
    K = max(len(k_lab), 1)
    cell, cell_keys = pd.factorize(gg * K + k_code)
    cell = cell.astype(np.int64)
    nc = len(cell_keys)
    ap = _ap_per_cell(yy, ss, cell, nc)
    P = np.bincount(cell, weights=yy, minlength=nc)
    T = np.bincount(cell, minlength=nc).astype(np.float64)
    N = T - P
    w = P * N
    pi = np.where(T > 0, P / np.maximum(T, 1), np.nan)
    val = (ap - pi) / (1 - pi) if normalize else ap
    val = np.where(w > 0, val, 0.0)
    cg = (np.asarray(cell_keys) // K).astype(np.int64)
    num = np.bincount(cg, weights=val * w, minlength=G)
    den = np.bincount(cg, weights=w, minlength=G)
    with np.errstate(invalid='ignore', divide='ignore'):
        A = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    valid = den > 0
    est = float(np.nanmean(A)) if valid.any() else np.nan
    rep, ci = None, (np.nan, np.nan)
    if rs is not None:
        Cf = rs.counts.astype(np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            rep = (Cf @ np.where(valid, A, 0.0)) / (Cf @ valid.astype(np.float64))
        ci = percentile_ci(rep, level)
    per = pd.DataFrame(dict(group=g_lab, value=A, pairs=den))
    return MetricResult(name, est, ci, rep, per, int(ok.sum()), int(yy.sum()), int(valid.sum()),
                        dict(normalize=normalize, chance=0.0 if normalize else np.nan))


# ---------------------------------------------------------------------------------------------- gain over position
def gain_over_position(y, pos_pred, pos_plus_model_pred, groups=None, *, metric='auroc', aggregate='mean', fold=None,
                       resample=None, n_boot=0, method='cluster', block_size=50, seed=0, level=0.95, order=None):
    """Change in the per-genome metric when the model score is added to the position model.

    ``pos_pred`` and ``pos_plus_model_pred``: out-of-fold predictions (genome-grouped CV) of the position model and
    of the position + model model, aligned with ``y``. Returns a :class:`MetricResult` with estimate = mean over
    genomes of metric(pos + model) - metric(pos), bootstrap CI (paired: both predictions see the same replicates),
    ``info['p_wilcoxon']`` (Wilcoxon signed-rank test over genomes, scipy defaults), ``info['groups_improved']``.

    ``fold``: fold id of every item when the cross-validation folds split genomes (e.g. contiguous chunks of one
    genome). Out-of-fold predictions from different fold models are then compared only within their fold: the
    per-genome AUROC is the within-fold AUROC weighted by positives x negatives (AUPRC: within-fold AP weighted by
    positives). Pooling predictions of different fold models biases the AUROC down (a fold rich in positives was
    trained on fewer) and the bias is larger for weak, coarse predictors such as a position model, which inflates the
    gain. With genome-grouped folds ``fold`` changes nothing."""
    if metric not in ('auroc', 'auprc'):
        raise ValueError("metric must be 'auroc' or 'auprc'")
    pos_pred = np.asarray(pos_pred, np.float64)
    pe = np.asarray(pos_plus_model_pred, np.float64)
    both = np.where(np.isfinite(pos_pred) & np.isfinite(pe), 0.0, np.nan)
    a = pos_pred + both
    b = pe + both
    y_arr = np.asarray(y)
    if resample is None and n_boot:
        resample = Resample(np.zeros(len(y_arr), np.int64) if groups is None else groups, method=method, n_boot=n_boot,
                            block_size=block_size, seed=seed, order=order)
    if metric == 'auroc':
        ra = within_auc(y_arr, a, fold, groups, aggregate=aggregate, resample=None)
        rb = within_auc(y_arr, b, fold, groups, aggregate=aggregate, resample=None)
    else:
        ra = auprc(y_arr, a, groups, fold=fold)
        rb = auprc(y_arr, b, groups, fold=fold)
    d_g = rb.per_group['value'].to_numpy() - ra.per_group['value'].to_numpy()
    valid = np.isfinite(d_g)
    if aggregate == 'pooled' and metric == 'auroc':
        est = rb.estimate - ra.estimate
    else:
        est = float(np.mean(d_g[valid])) if valid.any() else np.nan
    rep, ci = None, (np.nan, np.nan)
    if resample is not None:
        if resample.method == 'cluster' and aggregate == 'mean':
            Cf = resample.counts.astype(np.float64)
            with np.errstate(invalid='ignore', divide='ignore'):
                rep = (Cf @ np.where(valid, d_g, 0.0)) / (Cf @ valid.astype(np.float64))
        elif metric == 'auroc' and aggregate == 'mean':
            rep = _paired_mean_diff_auc(y_arr, a, b, groups, resample, fold)
        elif metric == 'auroc':
            rep = (within_auc(y_arr, b, fold, groups, aggregate='pooled', resample=resample).replicates
                   - within_auc(y_arr, a, fold, groups, aggregate='pooled', resample=resample).replicates)
        else:
            rep = _paired_mean_diff_ap(y_arr, a, b, groups, resample, fold)
        ci = percentile_ci(rep, level)
    try:   # with <= 5 genomes no two-sided exact p-value can fall below 0.0625: not reported
        p = float(wilcoxon(d_g[valid]).pvalue) if valid.sum() >= 6 and np.any(d_g[valid] != 0) else np.nan
    except ValueError:
        p = np.nan
    per = ra.per_group[['group', 'n', 'n_pos']].copy()
    per['position'] = ra.per_group['value'].to_numpy()
    per['position_plus_model'] = rb.per_group['value'].to_numpy()
    per['delta'] = d_g
    info = dict(metric=metric, p_wilcoxon=p, groups_improved=int((d_g[valid] > 0).sum()), n_valid_groups=int(valid.sum()),
                position=ra.estimate, position_plus_model=rb.estimate, method=None if resample is None else resample.method,
                fold_stratified=fold is not None)
    return MetricResult(f'gain_{metric}', est, ci, rep, per, ra.n_items, ra.n_pos, int(valid.sum()), info)


def _paired_mean_diff_auc(y, a, b, groups, rs, fold=None):
    """Block (or cross-group) bootstrap of mean over genomes of AUROC_g(b) - AUROC_g(a), same replicates."""
    ok = np.isfinite(a) & np.isfinite(b) & np.isin(y, (0, 1))
    yy = y[ok].astype(np.int8)
    unit = rs.unit_of_item[ok]
    z = np.zeros(len(yy), np.int64) if fold is None else factorize_sorted(np.asarray(fold)[ok])[0]
    na, da = _domain_replicates(yy, a[ok], z, unit, rs, 'within_group')
    nb, db = _domain_replicates(yy, b[ok], z, unit, rs, 'within_group')
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(da > 0, nb / np.where(db > 0, db, 1) - na / np.where(da > 0, da, 1), np.nan)
    cnt = np.isfinite(d).sum(1)
    return np.where(cnt > 0, np.nansum(d, 1) / np.maximum(cnt, 1), np.nan)


def _paired_mean_diff_ap(y, a, b, groups, rs, fold=None):
    ok = np.isfinite(a) & np.isfinite(b) & np.isin(y, (0, 1))
    yy = y[ok].astype(np.float64)
    g = rs.group_of_item[ok]
    G = len(rs.group_labels)
    unit = rs.unit_of_item[ok]
    cell, nc, cg = _cells(g, None if fold is None else np.asarray(fold)[ok])
    oa = np.lexsort((-a[ok], cell))
    ob = np.lexsort((-b[ok], cell))
    rep = np.empty(rs.n_boot)
    for i in range(rs.n_boot):
        apa = _ap_groups_sorted(yy[oa], rs.counts[i, unit[oa]].astype(np.float64), cell[oa], a[ok][oa], nc, cg, G)
        apb = _ap_groups_sorted(yy[ob], rs.counts[i, unit[ob]].astype(np.float64), cell[ob], b[ok][ob], nc, cg, G)
        rep[i] = np.nanmean(apb - apa)
    return rep


# ------------------------------------------------------------------------------------------------- position share
def position_share(raw, within, chance=0.5, min_margin=0.02):
    """Position share = 1 - (within - chance) / (raw - chance) = (raw - within) / (raw - chance).

    The share of a model's above-chance performance that disappears when gene position is held fixed: 0 = none,
    1 = all. Values above 1 mean the model ranks worse than chance at fixed position; values below 0 mean it ranks
    better at fixed position than overall. Defined (else NaN) only when raw - chance > min_margin. Works on scalars
    and arrays (e.g. bootstrap replicates)."""
    raw = np.asarray(raw, dtype=np.float64)
    within = np.asarray(within, dtype=np.float64)
    m = raw - chance
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(m > min_margin, (raw - within) / np.where(m != 0, m, np.nan), np.nan)
    return float(out) if out.ndim == 0 else out


def position_share_result(raw, within, chance=0.5, min_margin=0.02, level=0.95):
    """Position share with a paired bootstrap interval from two :class:`MetricResult` computed on the same Resample.

    The point estimate follows :func:`position_share`. The interval is the percentile interval of the replicate
    shares (raw* - within*) / (raw* - chance). If more than (1 - level) / 2 of the replicates have raw* <= chance the
    denominator is not bounded away from zero and no interval is given (flag ``ci_unbounded``). Flags:
    ``raw_near_chance`` (raw - chance <= min_margin, share not reported), ``within_below_chance`` (share > 1),
    ``within_above_raw`` (share < 0)."""
    est = position_share(raw.estimate, within.estimate, chance, min_margin)
    flags = []
    if not raw.estimate - chance > min_margin:
        flags.append('raw_near_chance')
    else:
        if within.estimate < chance:
            flags.append('within_below_chance')
        if within.estimate > raw.estimate:
            flags.append('within_above_raw')
    ci, rep = (np.nan, np.nan), None
    if np.isfinite(est) and raw.replicates is not None and within.replicates is not None:
        r, w = raw.replicates, within.replicates
        if len(r) != len(w):
            raise ValueError('raw and within were not computed on the same Resample')
        m = r - chance
        if np.mean(~(m > 0)) > (1 - level) / 2:
            flags.append('ci_unbounded')
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                rep = np.where(m > 0, (r - w) / np.where(m > 0, m, 1), np.where(r - w >= 0, np.inf, -np.inf))
            ci = percentile_ci(rep, level)
    return MetricResult('position_share', est, ci, rep, None, raw.n_items, raw.n_pos, raw.n_groups,
                        dict(chance=chance, min_margin=min_margin, flags=';'.join(flags)))


# ------------------------------------------------------------------------------- headline: position-explained share
def _group_midranks(s, g, w=None, order=None):
    """Midrank of every item's score among the items of its group (with multiplicities w: ranks in the expanded
    data, every copy of an item gets the item's midrank; items of weight 0 do not count). ``order``: a precomputed
    ``numpy.lexsort((s, g))``. Returns ranks aligned with s."""
    n = len(s)
    order = np.lexsort((s, g)) if order is None else order
    gg, ss = g[order], s[order]
    ww = np.ones(n) if w is None else w[order].astype(np.float64)
    new_g = np.r_[True, gg[1:] != gg[:-1]]
    cw = np.cumsum(ww)
    gstart = np.flatnonzero(new_g)
    base = np.repeat(np.where(gstart > 0, cw[gstart - 1], 0.0), np.diff(np.append(gstart, n)))
    cwg = cw - base                                   # inclusive cumulative weight within the group
    run = new_g | np.r_[True, ss[1:] != ss[:-1]]
    rid = np.cumsum(run) - 1
    rs = np.flatnonzero(run)
    re = np.append(rs[1:], n) - 1
    before = np.where(new_g[rs], 0.0, cwg[np.maximum(rs - 1, 0)])
    mid = before + (cwg[re] - before + 1) / 2.0
    R = np.empty(n)
    R[order] = mid[rid]
    return R


def _share_parts(y, s, q_cell, g, G, w=None, q_fixed=None, continuous=False, order=None):
    """Per-group E (AUROC excess), E_pos (position part) and W (P x N, or for continuous labels the Gini normaliser
    sum_{k<l} |y_k - y_l|), optionally with multiplicities w (0 allowed) and a precomputed score order."""
    ww = np.ones(len(y)) if w is None else w.astype(np.float64)
    R = _group_midranks(s, g, None if w is None else ww, order)
    n_g = np.bincount(g, weights=ww, minlength=G)
    c = R - (n_g[g] + 1) / 2.0
    P = np.bincount(g, weights=ww * y, minlength=G)
    if continuous:
        if w is not None:
            raise NotImplementedError('continuous labels: cluster bootstrap only')
        o = np.lexsort((y, g))
        gs = np.flatnonzero(np.r_[True, g[o][1:] != g[o][:-1]])
        r = np.arange(1, len(y) + 1) - np.repeat(gs, np.diff(np.append(gs, len(y))))
        W = np.bincount(g[o], weights=y[o] * (2 * r - n_g[g[o]] - 1), minlength=G)
    else:
        W = P * (n_g - P)
    if q_fixed is not None:
        q = q_fixed
    else:
        cell, cell_n = q_cell
        Pc = np.bincount(cell, weights=ww * y, minlength=cell_n)
        nc = np.bincount(cell, weights=ww, minlength=cell_n)
        with np.errstate(invalid='ignore', divide='ignore'):
            q_c = (Pc[cell] - y) / (nc[cell] - 1)
            q_g = (P[g] - y) / (n_g[g] - 1)
        q = np.where(nc[cell] > 1, q_c, np.where(n_g[g] > 1, q_g, 0.0))
    q = np.where(ww > 0, q, 0.0)
    e = np.bincount(g, weights=ww * y * c, minlength=G)
    ep = np.bincount(g, weights=ww * q * c, minlength=G)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(W > 0, e / np.where(W > 0, W, 1), np.nan), np.where(W > 0, ep / np.where(W > 0, W, 1), np.nan), W


def explained_share(y, score, strata=None, groups=None, *, q=None, aggregate='mean', resample=None, n_boot=0,
                    method='cluster', block_size=50, seed=0, level=0.95, order=None, min_margin=0.02,
                    labels='binary', name='position_share'):
    """Position share (the headline): the share of a model's AUROC excess over chance that is carried by the part
    of the labels that gene position predicts.

    Per genome g, with R_k the midrank of item k's score among the genome's items and c_k = R_k - (n_g + 1) / 2,

        E_g     = sum_k y_k c_k / (P_g N_g)            exactly AUROC_g - 1/2
        E_pos,g = sum_k q_k c_k / (P_g N_g)            the same with each label replaced by q_k
        share   = mean_g E_pos,g / mean_g E_g

    where q_k is the leave-one-out positive rate of the item's position stratum in its genome,
    (P_cell - y_k) / (n_cell - 1) (the genome's leave-one-out rate for an item alone in its stratum), or a
    user-supplied position-only probability (``q=``, e.g. a cross-fitted position model). Since
    y = q + (y - q), the excess splits exactly into E_pos + E_res; the share is 0 in expectation for a score that
    carries no position information and 1 for a score that depends on position only (up to position left inside
    the strata). It is reported when mean_g E_g > ``min_margin``. ``aggregate='pooled'`` weights genomes by
    P_g N_g. Intervals: cluster bootstrap (exact, per-genome parts) or block bootstrap (each replicate re-ranked
    with multiplicities, exact). ``info`` holds ``raw_excess`` (= AUROC - 1/2), ``position_excess``,
    ``residual_excess`` and ``flags``.

    ``labels='continuous'``: the same decomposition for a continuous label, with E_g = sum_k y_k c_k /
    sum_{k<l} |y_k - y_l| (a label-difference-weighted concordance minus 1/2; equal to AUROC - 1/2 for 0/1 labels)
    and q_k the leave-one-out mean label of the stratum; cluster bootstrap only."""
    continuous = labels == 'continuous'
    y, s, k, g, g_lab, rs, ok = _prepare(y, score, strata, groups, resample, n_boot, method, block_size, seed, order)
    if continuous:
        ok = np.isfinite(s) & np.isfinite(np.asarray(y, np.float64))
    G = len(g_lab)
    yy, ss, kk, gg = y[ok].astype(np.float64), s[ok], k[ok], g[ok]
    qq = None if q is None else np.asarray(q, np.float64)[ok]
    k_code, k_lab = factorize_sorted(kk)
    cell, cell_keys = pd.factorize(gg * max(len(k_lab), 1) + k_code)
    q_cell = (cell.astype(np.int64), len(cell_keys))
    E, Ep, W = _share_parts(yy, ss, q_cell, gg, G, q_fixed=qq, continuous=continuous)
    valid = np.isfinite(E)
    wts = np.where(valid, 1.0 if aggregate == 'mean' else W, 0.0)

    def ratio(Ev, Epv, wv):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.nansum(Epv * wv) / np.nansum(Ev * wv), np.nansum(Ev * wv) / np.sum(wv)
    sh, raw_ex = ratio(np.where(valid, E, 0), np.where(valid, Ep, 0), wts)
    flags = []
    est = sh
    if not raw_ex > min_margin:
        flags.append('raw_near_chance')
        est = np.nan
    rep, ci = None, (np.nan, np.nan)
    if rs is not None:
        if rs.method == 'cluster':
            Cf = rs.counts.astype(np.float64) * wts[None, :]
            num = Cf @ np.where(valid, Ep, 0)
            den = Cf @ np.where(valid, E, 0)
            raw_rep = den / Cf.sum(1)
            norm = Cf.sum(1)
        else:
            if qq is not None:
                raise NotImplementedError('block bootstrap with a fixed q is not implemented')
            unit = rs.unit_of_item[ok]
            num, den, raw_rep, norm = (np.empty(rs.n_boot) for _ in range(4))
            score_order = np.lexsort((ss, gg))              # fixed across replicates: O(n) per replicate
            for b in range(rs.n_boot):
                m = rs.counts[b, unit].astype(np.float64)
                Eb, Epb, Wb = _share_parts(yy, ss, q_cell, gg, G, w=m, order=score_order)
                vb = np.isfinite(Eb)
                wb = np.where(vb, 1.0 if aggregate == 'mean' else Wb, 0.0)
                num[b] = np.sum(np.where(vb, Epb, 0) * wb)
                den[b] = np.sum(np.where(vb, Eb, 0) * wb)
                norm[b] = wb.sum()
                raw_rep[b] = den[b] / norm[b]
        with np.errstate(invalid='ignore', divide='ignore'):
            rep = num / den
        if np.isfinite(est):
            if np.mean(~(raw_rep > 0)) > (1 - level) / 2:
                flags.append('ci_unbounded')
            else:
                rep = np.where(raw_rep > 0, rep, np.where(num >= 0, np.inf, -np.inf))
                ci = percentile_ci(rep, level)
    if np.isfinite(est):   # beyond [0, 1]: by the interval when there is one, else by the point estimate
        lo, hi = (ci if np.isfinite(ci[0]) else (est, est))
        if lo > 1:
            flags.append('share_above_1')
        if hi < 0:
            flags.append('share_below_0')
    per = pd.DataFrame(dict(group=g_lab, n=np.bincount(gg, minlength=G), n_pos=np.bincount(gg, weights=yy, minlength=G).astype(np.int64),
                            raw_excess=E, position_excess=Ep, share=Ep / E))
    info = dict(raw_excess=raw_ex, position_excess=raw_ex * sh if np.isfinite(sh) else np.nan,
                residual_excess=raw_ex * (1 - sh) if np.isfinite(sh) else np.nan, share_unreported=sh,
                flags=';'.join(flags), min_margin=min_margin, aggregate=aggregate,
                q='strata' if q is None else 'user', method=None if rs is None else rs.method)
    if rs is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            info['rep_position_excess'] = num / norm
            info['rep_raw_excess'] = den / norm
    return MetricResult(name, float(est), ci, rep, per, int(ok.sum()), int(yy.sum()), int(valid.sum()), info)


def equalized_score(score, strata, groups=None):
    """Within-(genome, stratum) percentile of the score, (midrank - 1/2) / n_cell: the score with its distribution
    made identical in every position stratum, keeping its ranking inside each stratum. Labels are not used."""
    s = np.asarray(score, np.float64)
    n = len(s)
    g = np.zeros(n, np.int64) if groups is None else factorize_sorted(groups)[0]
    k, kl = factorize_sorted(np.zeros(n, np.int64) if strata is None else strata)
    cell = pd.factorize(g * max(len(kl), 1) + k)[0].astype(np.int64)
    ok = np.isfinite(s)
    u = np.full(n, np.nan)
    c = cell[ok]
    R = _group_midranks(s[ok], c)
    nc = np.bincount(c)
    u[ok] = (R - 0.5) / nc[c]
    return u


def position_share_bound(y, score, strata=None, groups=None, *, aggregate='mean', fold=None, resample=None, n_boot=0,
                         method='cluster', block_size=50, seed=0, level=0.95, order=None, min_margin=0.02,
                         name='position_share'):
    """Position share (the headline): a lower bound (``estimate``) and an upper bound (``info['share_upper']``) on the
    share of a model's AUROC above chance that is due to gene position. The lower bound is the exact share when
    position is a consequence of the labels, the upper bound when position is a cause of them; with both mechanisms
    the share lies in between.

    Two "position-free" counterfactuals of the score, both computed from the scores and position strata alone:

    * held fixed: the AUROC within position strata, ``within``;
    * equalized: the AUROC of the score after it is replaced by its percentile inside its (genome, stratum) cell,
      ``auroc_eq`` (the score made identical in distribution across positions, ranks inside each stratum kept).

    With raw = the ordinary AUROC (per genome, mean over genomes),

        share_within    = 1 - (within - 1/2)   / (raw - 1/2)
        share_equalized = 1 - (auroc_eq - 1/2) / (raw - 1/2)
        share           = clip(min(share_within, share_equalized), 0, 1)      (the headline)
        share_upper     = clip(max(share_within, share_equalized), 0, 1)

    For a score made of a position part and a position-free part, the true share is the fraction of raw - 1/2 that
    would be lost without the position part. share_within equals it when position is a consequence of the label
    (the position-free part is independent of position given the label, as for operons, where co-transcription shapes
    gene spacing); share_equalized equals it when position is a cause of the label (the position-free part is
    independent of position, as for STRING links built from gene neighbourhood). In each world the other estimate errs
    on the far side, so the true share lies between ``share`` and ``share_upper`` in both. ``share`` is 0 for a score
    unrelated to position in either sense, about 1 for a function of position (minus the position signal left inside
    the strata, which row zero reports), monotone in the weight of the position part, and rank-based (invariant to
    monotone transforms of the score). Reported when raw - 1/2 > min_margin.

    ``fold``: for out-of-fold predictions whose folds split genomes, every AUROC is computed within (genome, fold)
    (see :func:`gain_over_position`). Intervals: cluster bootstrap from per-genome Mann-Whitney statistics (exact) or
    block bootstrap (quadratic forms; the percentiles are computed once). ``info``: raw, within, auroc_eq, share_within,
    share_equalized, share_upper
    (and its interval), unclipped values and flags (``raw_near_chance``, ``clipped_low``, ``clipped_high``,
    ``ci_unbounded``)."""
    y = np.asarray(y)
    s = np.asarray(score, np.float64)
    n = len(y)
    groups_ = np.zeros(n, np.int64) if groups is None else np.asarray(groups)
    strata_ = np.zeros(n, np.int64) if strata is None else np.asarray(strata)
    if resample is None and n_boot:
        resample = Resample(groups_, method=method, n_boot=n_boot, block_size=block_size, seed=seed, order=order)
    if fold is not None:
        fcode = factorize_sorted(fold)[0]
        strata_ = pd.factorize(pd.MultiIndex.from_arrays([strata_, fcode]))[0].astype(np.int64)
    else:
        fcode = None
    u = equalized_score(s, strata_, groups_)
    kw = dict(aggregate=aggregate, resample=resample, level=level)
    raw = within_auc(y, s, fcode, groups_, **kw)
    wit = within_auc(y, s, strata_, groups_, **kw)
    eq = within_auc(y, u, fcode, groups_, **kw)

    def shares(r, w, e):
        with np.errstate(invalid='ignore', divide='ignore'):
            d = np.asarray(r, np.float64) - 0.5
            sw = 1 - (np.asarray(w, np.float64) - 0.5) / d
            se = 1 - (np.asarray(e, np.float64) - 0.5) / d
        return sw, se
    sw, se = shares(raw.estimate, wit.estimate, eq.estimate)
    lo_raw, hi_raw = float(min(sw, se)), float(max(sw, se))
    flags = []
    est = float(np.clip(lo_raw, 0, 1))
    upper = float(np.clip(hi_raw, 0, 1))
    if not raw.estimate - 0.5 > min_margin:
        flags.append('raw_near_chance')
        est = upper = np.nan
    else:
        if lo_raw < 0:
            flags.append('clipped_low')
        if lo_raw > 1:
            flags.append('clipped_high')
    ci, ci_up, rep = (np.nan, np.nan), (np.nan, np.nan), None
    if resample is not None and np.isfinite(est):
        rsw, rse = shares(raw.replicates, wit.replicates, eq.replicates)
        bad = ~(raw.replicates - 0.5 > 0)
        if np.mean(bad) > (1 - level) / 2:
            flags.append('ci_unbounded')
        else:
            rep = np.where(bad, np.nan, np.clip(np.minimum(rsw, rse), 0, 1))
            rep_up = np.where(bad, np.nan, np.clip(np.maximum(rsw, rse), 0, 1))
            ci, ci_up = percentile_ci(rep, level), percentile_ci(rep_up, level)
    info = dict(raw=raw.estimate, within=wit.estimate, auroc_eq=eq.estimate, share_within=float(sw),
                share_equalized=float(se), share_upper=upper, share_upper_lo=ci_up[0], share_upper_hi=ci_up[1],
                lower_unclipped=lo_raw, upper_unclipped=hi_raw, flags=';'.join(flags), min_margin=min_margin,
                method=None if resample is None else resample.method)
    return MetricResult(name, est, ci, rep, None, raw.n_items, raw.n_pos, raw.n_groups, info)


def explained_share_ovr(y, scores, strata=None, groups=None, *, classes=None, resample=None, n_boot=0,
                        method='cluster', block_size=50, seed=0, level=0.95, min_margin=0.02, min_pos=1, **kw):
    """Position share for multi-class labels: one-vs-rest per class (column c of ``scores`` scores class
    ``classes[c]``), macro-averaged on the excess scale, share = sum_c E_pos,c / sum_c E_c (E_c = one-vs-rest AUROC
    excess of class c, mean over genomes). All classes share one Resample."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    classes = np.unique(y) if classes is None else np.asarray(classes)
    if resample is None and n_boot:
        resample = Resample(np.zeros(len(y), np.int64) if groups is None else groups, method=method, n_boot=n_boot,
                            block_size=block_size, seed=seed)
    ep, e, rnum, rden, used = 0.0, 0.0, 0.0, 0.0, []
    for c_i, c in enumerate(classes):
        yc = (y == c).astype(np.int8)
        if yc.sum() < min_pos:
            continue
        r = explained_share(yc, scores[:, c_i], strata, groups, resample=resample, min_margin=-np.inf, **kw)
        if not np.isfinite(r.info['raw_excess']):
            continue
        used.append(c)
        e += r.info['raw_excess']
        ep += r.info['raw_excess'] * r.info['share_unreported']
        if resample is not None:
            rnum = rnum + r.info['rep_position_excess']
            rden = rden + r.info['rep_raw_excess']
    k = max(len(used), 1)
    est = ep / e if e / k > min_margin else np.nan
    rep, ci = None, (np.nan, np.nan)
    if resample is not None and np.isfinite(est):
        with np.errstate(invalid='ignore', divide='ignore'):
            rep = np.where(rden > 0, rnum / rden, np.nan)
        ci = percentile_ci(rep, level)
    return MetricResult('position_share_ovr', float(est), ci, rep, None, len(y), 0,
                        len(np.unique(groups)) if groups is not None else 1,
                        dict(classes=len(used), raw_excess_macro=e / k, flags='' if np.isfinite(est) else 'raw_near_chance'))


# ------------------------------------------------------------------------------------------------ multi-class, ranks
def within_auc_ovr(y, scores, strata=None, groups=None, *, classes=None, resample=None, n_boot=0, method='cluster',
                   block_size=50, seed=0, level=0.95, min_pos=1, **kw):
    """Macro-averaged one-vs-rest AUROC with position held fixed, for multi-class labels.

    ``scores``: (n, K) array, column c scoring class ``classes[c]``. Every class uses the same Resample, so the
    macro average has a proper bootstrap interval. Classes with fewer than ``min_pos`` positives are skipped."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    classes = np.unique(y) if classes is None else np.asarray(classes)
    if resample is None and n_boot:
        resample = Resample(np.zeros(len(y), np.int64) if groups is None else groups, method=method, n_boot=n_boot,
                            block_size=block_size, seed=seed)
    ests, reps, used = [], [], []
    for c_i, c in enumerate(classes):
        yc = (y == c).astype(np.int8)
        if yc.sum() < min_pos:
            continue
        r = within_auc(yc, scores[:, c_i], strata, groups, resample=resample, **kw)
        ests.append(r.estimate)
        used.append(c)
        if r.replicates is not None:
            reps.append(r.replicates)
    est = float(np.nanmean(ests)) if ests else np.nan
    rep = np.nanmean(np.stack(reps), 0) if reps else None
    ci = percentile_ci(rep, level) if rep is not None else (np.nan, np.nan)
    per = pd.DataFrame(dict(cls=used, value=ests))
    return MetricResult('within_auroc_ovr' if strata is not None else 'auroc_ovr', est, ci, rep, per, len(y), 0,
                        len(np.unique(groups)) if groups is not None else 1, dict(classes=len(used)))


def concordance(y, score, strata=None, groups=None, *, resample=None, n_boot=0, seed=0, level=0.95):
    """Harrell's concordance index for a continuous label (chance 0.5), with position held fixed by ``strata``.

    C = 0.5 + 0.5 * sum(concordant - discordant) / sum(pairs with different labels), over pairs in the same
    (genome, stratum) cell; per genome, then averaged over genomes. Score ties count 1/2. Cluster bootstrap only.
    For binary labels it equals :func:`within_auc`."""
    from scipy.stats import kendalltau
    y = np.asarray(y, np.float64)
    s = np.asarray(score, np.float64)
    n = len(y)
    groups = np.zeros(n, np.int64) if groups is None else np.asarray(groups)
    strata = np.zeros(n, np.int64) if strata is None else np.asarray(strata)
    if resample is None and n_boot:
        resample = Resample(groups, method='cluster', n_boot=n_boot, seed=seed)
    if resample is not None and resample.method != 'cluster':
        raise NotImplementedError('concordance supports the cluster bootstrap only')
    g, g_lab = (resample.group_of_item, resample.group_labels) if resample is not None else factorize_sorted(groups)
    ok = np.isfinite(y) & np.isfinite(s)
    G = len(g_lab)
    k_code, k_lab = factorize_sorted(strata[ok])
    K = max(len(k_lab), 1)
    cell, keys = pd.factorize(g[ok] * K + k_code)
    order = np.argsort(cell, kind='stable')
    bounds = np.searchsorted(cell[order], np.arange(len(keys) + 1))
    yo, so = y[ok][order], s[ok][order]
    cd = np.zeros(len(keys))
    comp = np.zeros(len(keys))
    for c in range(len(keys)):
        yy, ss = yo[bounds[c]:bounds[c + 1]], so[bounds[c]:bounds[c + 1]]
        m = len(yy)
        if m < 2:
            continue
        n0 = m * (m - 1) / 2
        _, ty = np.unique(yy, return_counts=True)
        _, ts = np.unique(ss, return_counts=True)
        n2 = float((ty * (ty - 1) / 2).sum())
        n1 = float((ts * (ts - 1) / 2).sum())
        if n0 - n2 <= 0:
            continue
        if n0 - n1 <= 0:
            comp[c] = n0 - n2
            continue
        tau = kendalltau(ss, yy).statistic
        cd[c] = tau * np.sqrt((n0 - n1) * (n0 - n2))
        comp[c] = n0 - n2
    cg = (np.asarray(keys) // K).astype(np.int64)
    num = np.bincount(cg, weights=cd, minlength=G)
    den = np.bincount(cg, weights=comp, minlength=G)
    with np.errstate(invalid='ignore', divide='ignore'):
        Cg = np.where(den > 0, 0.5 + 0.5 * num / np.where(den > 0, den, 1), np.nan)
    valid = den > 0
    est = float(np.nanmean(Cg)) if valid.any() else np.nan
    rep, ci = None, (np.nan, np.nan)
    if resample is not None:
        Cf = resample.counts.astype(np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            rep = (Cf @ np.where(valid, Cg, 0.0)) / (Cf @ valid.astype(np.float64))
        ci = percentile_ci(rep, level)
    return MetricResult('within_concordance' if strata is not None else 'concordance', est, ci, rep,
                        pd.DataFrame(dict(group=g_lab, value=Cg, pairs=den)), int(ok.sum()), 0, int(valid.sum()), {})
