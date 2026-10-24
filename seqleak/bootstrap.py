"""Resampling plans shared by all statistics of one table.

Every statistic computed with the same :class:`Resample` sees the same bootstrap replicates, so intervals of
derived quantities (gain over position, position share) are paired.

Two plans:

* ``method='cluster'``: the units are the groups (genomes); each replicate draws G groups with replacement.
* ``method='block'``: for tasks with few genomes. The units are blocks of ``block_size`` consecutive items within
  each group (``numpy.array_split`` of the group's items into ``max(n_g // block_size, min(20, n_g))`` blocks);
  each group keeps its identity and its blocks are drawn with replacement.

Draw order, fixed for reproducibility: replicate by replicate; within a replicate, groups in sorted order
(``block``) or one draw of G group indices (``cluster``).
"""
import numpy as np
import pandas as pd


def factorize_sorted(x):
    """Codes 0..K-1 following the sorted order of the unique values (like ``np.unique(..., return_inverse=True)``)."""
    codes, uniques = pd.factorize(np.asarray(x), sort=True)
    if (codes < 0).any():
        raise ValueError('missing values in a grouping variable')
    return codes.astype(np.int64), np.asarray(uniques)


class Resample:
    """Bootstrap multiplicities of resampling units.

    Parameters
    ----------
    groups : array (n,) of group labels (genome ids), one per item.
    method : 'cluster' or 'block'.
    n_boot : number of replicates.
    block_size : items per block (``block`` only).
    seed, rng : random seed or an existing ``numpy.random.Generator`` (``rng`` wins; it is advanced in place, so
        several plans drawn from one generator reproduce scripts that share a generator).
    order : optional permutation (n,) giving the genome order of the items; blocks are cut along it. Default: the
        order of the items.
    max_blocks : cap on blocks per group (larger groups get proportionally larger blocks).
    min_blocks : floor on blocks per group (smaller groups get smaller blocks; groups with >= 50 x 20 items are
        unaffected, so the v1 plans are reproduced).

    Attributes
    ----------
    group_of_item : (n,) group code of each item (sorted order of labels).
    group_labels : (G,) labels.
    unit_of_item : (n,) unit code of each item.
    unit_group : (U,) group code of each unit.
    counts : (n_boot, U) int32 multiplicity of every unit in every replicate.
    """

    def __init__(self, groups, method='cluster', n_boot=2000, block_size=50, seed=0, rng=None, order=None,
                 max_blocks=1000, min_blocks=20):
        if method not in ('cluster', 'block'):
            raise ValueError("method must be 'cluster' or 'block'")
        self.method = method
        self.n_boot = int(n_boot)
        self.block_size = block_size
        self.group_of_item, self.group_labels = factorize_sorted(groups)
        n = len(self.group_of_item)
        G = len(self.group_labels)
        rng = np.random.default_rng(seed) if rng is None else rng
        B = self.n_boot
        if method == 'cluster':
            self.unit_of_item = self.group_of_item
            self.unit_group = np.arange(G)
            draws = rng.integers(0, G, size=(B, G))
            self.counts = np.bincount((draws + G * np.arange(B)[:, None]).ravel(), minlength=B * G).reshape(B, G).astype(np.int32)
            return
        order = np.arange(n) if order is None else np.asarray(order)
        g_ord = self.group_of_item[order]
        self.unit_of_item = np.empty(n, np.int64)
        nb, first = [], 0
        for g in range(G):
            idx = order[g_ord == g]
            k = max(1, len(idx) // block_size, min(min_blocks, len(idx)))
            k = min(k, max_blocks)
            for b, part in enumerate(np.array_split(idx, k)):
                self.unit_of_item[part] = first + b
            nb.append(k)
            first += k
        self.blocks_per_group = np.array(nb)
        self.unit_group = np.repeat(np.arange(G), nb)
        offs = np.concatenate([[0], np.cumsum(nb)])
        self.counts = np.zeros((B, first), np.int32)
        for b in range(B):
            for g in range(G):
                d = rng.integers(0, nb[g], nb[g])
                self.counts[b, offs[g]:offs[g + 1]] += np.bincount(d, minlength=nb[g]).astype(np.int32)

    @property
    def n_units(self):
        return len(self.unit_group)

    def __repr__(self):
        return f'Resample(method={self.method!r}, n_boot={self.n_boot}, groups={len(self.group_labels)}, units={self.n_units})'


def percentile_ci(rep, level=0.95):
    """Percentile interval of bootstrap replicates (``numpy.percentile``, linear interpolation), ignoring NaN.

    Infinite replicates are kept: a bound that falls on or next to an infinite replicate is infinite."""
    rep = np.asarray(rep, float)
    rep = rep[~np.isnan(rep)]
    if len(rep) == 0:
        return (np.nan, np.nan)
    a = (1 - level) / 2 * 100
    with np.errstate(invalid='ignore'):
        lo, hi = np.percentile(rep, [a, 100 - a])
    if np.isinf(rep).any():
        srt = np.sort(rep)
        m = len(srt) - 1
        for q, which in ((a, 0), (100 - a, 1)):
            pos = q / 100 * m
            pair = srt[int(np.floor(pos))], srt[int(np.ceil(pos))]
            val = pair[0] if not np.isfinite(pair[0]) else pair[1] if not np.isfinite(pair[1]) else None
            if val is not None:
                if which == 0:
                    lo = val
                else:
                    hi = val
    return (float(lo), float(hi))
