"""Position strata: items in one stratum share their gene position (up to binning).

Presets
-------
``'adjacent'`` (alias ``'operon_v1'``)
    Neighbouring gene pairs: strand state (same / opposite) x 17 intergenic-gap bins with edges
    -20, 0, 20, 40, 60, 80, 100, 130, 160, 200, 250, 300, 400, 500, 750, 1000 bp (34 strata).
``'pair'`` (alias ``'ppi_v1'``)
    Any gene pairs: 11 genes-apart bins (edges 1, 2, 3, 5, 10, 20, 50, 100, 300, 1000, right-closed) x strand switches
    (0, 1, 2, 3+) x 9 bins of the largest intervening gap (edges 50, 100, 150, 200, 300, 500, 1000, 2000) x strand
    state (792 strata).
``'gene'``
    Single genes: directon position (single / first / internal / last) x replication strand (if present) x five
    equal-width bins of distance to the origin (if present) x gene-length quintiles. Gene length is a gene-level
    covariate, not layout; for layout-only strata leave it out (a custom scheme without ``length``, or
    ``position_columns`` without it, which gives prognostic strata).
``'auto'``
    ``'pair'`` if ``genes_apart`` varies, ``'adjacent'`` for neighbouring pairs, else ``'gene'``.

Custom schemes are dicts ``{column: spec}`` where spec is a list of bin edges (``numpy.digitize``), an int q
(q quantile bins over all items), ``'cat'`` (one stratum per value) or ``('right', edges)`` (right-closed bins).
Codes are combined in mixed radix, so a stratum code identifies the bin of every column.
"""
import numpy as np
import pandas as pd

GAP_EDGES = (-np.inf, -20, 0, 20, 40, 60, 80, 100, 130, 160, 200, 250, 300, 400, 500, 750, 1000, np.inf)
K_EDGES = (1, 2, 3, 5, 10, 20, 50, 100, 300, 1000)
MAXGAP_EDGES = (50, 100, 150, 200, 300, 500, 1000, 2000)


def gap_strata(same_strand, gap, edges=GAP_EDGES):
    """Strand state x gap bin (the long-read operon strata of the v1 analysis): same * 100 + bin."""
    b = np.digitize(np.asarray(gap, np.float64), edges[1:-1])
    return np.asarray(same_strand).astype(np.int64) * 100 + b


def pair_strata(genes_apart, strand_switches, max_intervening_gap, same_strand):
    """Genes apart x strand switches x largest intervening gap x strand state (the PPI strata of the v1 analysis)."""
    kb = np.digitize(np.asarray(genes_apart), K_EDGES, right=True)
    mgb = np.digitize(np.asarray(max_intervening_gap), MAXGAP_EDGES)
    swb = np.minimum(np.asarray(strand_switches), 3)
    return (((kb * 4 + swb) * 9 + mgb) * 2 + np.asarray(same_strand).astype(np.int64)).astype(np.int64)


def _bin(values, spec):
    v = np.asarray(values)
    if isinstance(spec, str) and spec == 'cat':
        codes, uniq = pd.factorize(v, sort=True)
        return codes.astype(np.int64), max(len(uniq), 1)
    if isinstance(spec, (int, np.integer)):
        x = v.astype(np.float64)
        edges = np.unique(np.nanquantile(x, np.linspace(0, 1, int(spec) + 1)[1:-1]))
        return np.digitize(x, edges).astype(np.int64), len(edges) + 1
    if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == 'right':
        edges = np.asarray(spec[1], np.float64)
        return np.digitize(v.astype(np.float64), edges, right=True).astype(np.int64), len(edges) + 1
    edges = np.asarray(spec, np.float64)
    return np.digitize(v.astype(np.float64), edges).astype(np.int64), len(edges) + 1


def gene_scheme(features):
    """The default per-gene scheme restricted to the columns present."""
    s = {}
    if 'directon_pos' in features:
        s['directon_pos'] = 'cat'
    if 'replication_strand' in features and features['replication_strand'].notna().any():
        s['replication_strand'] = 'cat'
    if 'dist_origin' in features and features['dist_origin'].notna().any():
        s['dist_origin'] = [0.2, 0.4, 0.6, 0.8]
    if 'length' in features:
        s['length'] = 5
    if not s and 'strand' in features:
        s['strand'] = 'cat'
    return s


def resolve_scheme(features, scheme='auto'):
    if not isinstance(scheme, str) or scheme != 'auto':
        return scheme
    if 'genes_apart' in features and features['genes_apart'].nunique() > 1:
        return 'pair'
    if 'gap' in features and 'same_strand' in features:
        return 'adjacent'
    return 'gene'


def make_strata(features, scheme='auto'):
    """Integer stratum code of every item (see module docstring for the schemes)."""
    scheme = resolve_scheme(features, scheme)
    if isinstance(scheme, str):
        if scheme in ('adjacent', 'operon_v1'):
            return gap_strata(features['same_strand'].to_numpy(), features['gap'].to_numpy())
        if scheme in ('pair', 'ppi_v1'):
            return pair_strata(features['genes_apart'].to_numpy(), features['strand_switches'].to_numpy(),
                               features['max_intervening_gap'].to_numpy(), features['same_strand'].to_numpy())
        if scheme == 'gene':
            scheme = gene_scheme(features)
        else:
            raise ValueError(f'unknown scheme {scheme!r}')
    code = np.zeros(len(features), np.int64)
    for col, spec in scheme.items():
        b, nb = _bin(features[col].to_numpy(), spec)
        code = code * nb + b
    return code


def prognostic_strata(pos_pred, groups=None, n_bins=20):
    """Strata from quantile bins of an out-of-fold position-model prediction (a prognostic score), computed within
    each genome. Items in one stratum have about the same label probability given position."""
    p = np.asarray(pos_pred, np.float64)
    g = np.zeros(len(p), np.int64) if groups is None else pd.factorize(np.asarray(groups), sort=True)[0]
    out = np.zeros(len(p), np.int64)
    for k in np.unique(g):
        m = g == k
        r = pd.Series(p[m]).rank(method='average', pct=True).to_numpy()
        out[m] = k * n_bins + np.minimum((r * n_bins).astype(np.int64), n_bins - 1)
    return out


def describe_strata(features, strata, y=None):
    """Counts (and positives) per stratum, with the mean of every feature column."""
    df = features.copy()
    df['stratum'] = strata
    agg = {c: 'mean' for c in features.columns if np.issubdtype(features[c].dtype, np.number) or features[c].dtype == bool}
    out = df.groupby('stratum').agg(agg)
    out.insert(0, 'n', df.groupby('stratum').size())
    if y is not None:
        out.insert(1, 'n_pos', pd.Series(np.asarray(y)).groupby(strata).sum())
    return out
