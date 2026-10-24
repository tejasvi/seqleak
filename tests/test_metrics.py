import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from seqleak import metrics as M
from seqleak.bootstrap import Resample, percentile_ci
from seqleak.strata import gap_strata
from seqleak.synthetic import simulate_pairs


def ref_stratified_auc(y, s, strata):
    """The v1 reference (src/common.py stratified_auc)."""
    num = den = 0.0
    for k in np.unique(strata):
        m = strata == k
        p, n = y[m].sum(), (1 - y[m]).sum()
        if p == 0 or n == 0:
            continue
        num += roc_auc_score(y[m], s[m]) * p * n
        den += p * n
    return num / den if den > 0 else np.nan


def ref_within(y, s, strata, groups, how='mean'):
    vals, nums, dens = [], [], []
    for g in np.unique(groups):
        m = groups == g
        vals.append(ref_stratified_auc(y[m], s[m], strata[m]))
    return np.nanmean(vals)


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(1)
    n = 3000
    groups = rng.integers(0, 7, n)
    strata = rng.integers(0, 5, n)
    y = (rng.random(n) < 0.2 + 0.1 * strata).astype(np.int8)
    s = np.round(rng.normal(size=n) + 0.5 * y + 0.3 * strata, 1)   # rounding creates ties
    return y, s, strata, groups


def test_mann_whitney_cells_matches_sklearn(data):
    y, s, strata, groups = data
    cell = groups * 5 + strata
    P, N, U = M.mann_whitney_cells(y, s, cell, 35)
    for c in range(35):
        m = cell == c
        if y[m].min() == y[m].max():
            continue
        assert np.isclose(U[c] / (P[c] * N[c]), roc_auc_score(y[m], s[m]), atol=1e-12)


def test_within_auc_matches_reference(data):
    y, s, strata, groups = data
    r = M.within_auc(y, s, strata, groups)
    assert np.isclose(r.estimate, ref_within(y, s, strata, groups), atol=1e-12)
    raw = M.auroc(y, s, groups)
    assert np.isclose(raw.estimate, np.mean([roc_auc_score(y[groups == g], s[groups == g]) for g in range(7)]))
    pooled = M.within_auc(y, s, strata, groups, aggregate='pooled')
    num = den = 0.0
    for g in range(7):
        for k in range(5):
            m = (groups == g) & (strata == k)
            p, q = y[m].sum(), (1 - y[m]).sum()
            if p and q:
                num += roc_auc_score(y[m], s[m]) * p * q
                den += p * q
    assert np.isclose(pooled.estimate, num / den)


def _expand(idx_lists):
    return np.concatenate(idx_lists)


def test_cluster_bootstrap_is_exact(data):
    y, s, strata, groups = data
    rs = Resample(groups, 'cluster', n_boot=25, seed=3)
    r = M.within_auc(y, s, strata, groups, resample=rs)
    labels = rs.group_labels
    for b in range(25):
        # re-score the resampled data: copies of a genome are distinct genomes
        ys, ss, ks, gs = [], [], [], []
        copy = 0
        for g, c in enumerate(rs.counts[b]):
            for _ in range(c):
                m = groups == labels[g]
                ys.append(y[m]); ss.append(s[m]); ks.append(strata[m]); gs.append(np.full(m.sum(), copy))
                copy += 1
        ref = ref_within(np.concatenate(ys), np.concatenate(ss), np.concatenate(ks), np.concatenate(gs))
        assert np.isclose(r.replicates[b], ref, atol=1e-12)


def test_block_bootstrap_is_exact(data):
    y, s, strata, groups = data
    rs = Resample(groups, 'block', n_boot=15, block_size=40, seed=5)
    r = M.within_auc(y, s, strata, groups, resample=rs)
    blocks = [np.flatnonzero(rs.unit_of_item == u) for u in range(rs.n_units)]
    for b in range(15):
        vals = []
        for g in range(len(rs.group_labels)):
            units = np.flatnonzero(rs.unit_group == g)
            ix = np.concatenate([np.repeat(blocks[u][None, :], rs.counts[b, u], 0).ravel() for u in units if rs.counts[b, u]])
            vals.append(ref_stratified_auc(y[ix], s[ix], strata[ix]))
        assert np.isclose(r.replicates[b], np.nanmean(vals), atol=1e-12)
    # pooled across genomes, cluster resampling: quadratic form equals re-scoring the stacked data
    rc = Resample(groups, 'cluster', n_boot=10, seed=2)
    ra = M.within_auc(y, s, strata, groups, compare='across_groups', resample=rc)
    for b in range(10):
        ix = np.concatenate([np.flatnonzero(groups == rc.group_labels[g]) for g in range(7) for _ in range(rc.counts[b, g])])
        assert np.isclose(ra.replicates[b], ref_stratified_auc(y[ix], s[ix], strata[ix]), atol=1e-12)


def test_resample_streams_match_v1_scripts():
    groups = np.repeat(np.arange(53), 10)
    rng = np.random.default_rng(0)
    rs = Resample(groups, 'cluster', n_boot=50, rng=rng)
    rng2 = np.random.default_rng(0)
    for b in range(50):
        idx = rng2.integers(0, 53, 53)
        assert np.array_equal(rs.counts[b], np.bincount(idx, minlength=53))
    # block plan follows src/stats_lr.py: blocks = array_split(idx, len // 50), per replicate one draw per strain
    strain = np.repeat(np.arange(5), [430, 512, 77, 1000, 260])
    rng = np.random.default_rng(0)
    rs = Resample(strain, 'block', n_boot=20, block_size=50, rng=rng, min_blocks=1)   # v1 had no floor
    rng2 = np.random.default_rng(0)
    idx_by = {k: np.where(strain == k)[0] for k in range(5)}
    blocks = {k: np.array_split(idx_by[k], max(1, len(idx_by[k]) // 50)) for k in range(5)}
    for b in range(20):
        ix = {k: np.concatenate([blocks[k][i] for i in rng2.integers(0, len(blocks[k]), len(blocks[k]))]) for k in range(5)}
        mult = np.bincount(np.concatenate(list(ix.values())), minlength=len(strain))
        assert np.array_equal(mult, rs.counts[b, rs.unit_of_item])


def test_auprc_matches_sklearn(data):
    y, s, strata, groups = data
    r = M.auprc(y, s, groups)
    ref = [average_precision_score(y[groups == g], s[groups == g]) for g in range(7)]
    assert np.allclose(r.per_group['value'], ref, atol=1e-12)
    w = np.random.default_rng(0).integers(0, 4, len(y)).astype(float)
    order = np.lexsort((-s, groups))
    ap = M._ap_sorted(y[order].astype(float), w[order], groups[order], 7, s[order])
    ref = [average_precision_score(y[groups == g], s[groups == g], sample_weight=w[groups == g]) for g in range(7)]
    assert np.allclose(ap, ref, atol=1e-12)


def test_auprc_block_bootstrap_is_exact(data):
    y, s, strata, groups = data
    rs = Resample(groups, 'block', n_boot=5, block_size=60, seed=0)
    r = M.auprc(y, s, groups, resample=rs)
    for b in range(5):
        w = rs.counts[b, rs.unit_of_item].astype(float)
        ref = np.nanmean([average_precision_score(y[groups == g], s[groups == g], sample_weight=w[groups == g])
                          for g in range(7)])
        assert np.isclose(r.replicates[b], ref, atol=1e-12)


def test_gain_over_position(data):
    y, s, strata, groups = data
    rng = np.random.default_rng(4)
    pos = strata + 0.01 * rng.normal(size=len(y))
    pe = pos + 0.8 * s
    rs = Resample(groups, 'cluster', n_boot=200, seed=0)
    g = M.gain_over_position(y, pos, pe, groups, resample=rs)
    d = [roc_auc_score(y[groups == k], pe[groups == k]) - roc_auc_score(y[groups == k], pos[groups == k]) for k in range(7)]
    assert np.isclose(g.estimate, np.mean(d))
    assert np.isclose(g.info['p_wilcoxon'], wilcoxon(d).pvalue)
    assert g.ci[0] < g.estimate < g.ci[1]
    # block bootstrap: paired re-scoring
    rb = Resample(groups, 'block', n_boot=8, block_size=50, seed=1)
    gb = M.gain_over_position(y, pos, pe, groups, resample=rb)
    blocks = [np.flatnonzero(rb.unit_of_item == u) for u in range(rb.n_units)]
    for b in range(8):
        vals = []
        for k in range(7):
            ix = np.concatenate([np.repeat(blocks[u][None, :], rb.counts[b, u], 0).ravel() for u in np.flatnonzero(rb.unit_group == k) if rb.counts[b, u]])
            vals.append(roc_auc_score(y[ix], pe[ix]) - roc_auc_score(y[ix], pos[ix]))
        assert np.isclose(gb.replicates[b], np.mean(vals), atol=1e-12)


def test_position_share_definition_and_edges():
    assert np.isclose(M.position_share(0.8, 0.5), 1.0)
    assert np.isclose(M.position_share(0.8, 0.8), 0.0)
    assert np.isclose(M.position_share(0.7, 0.6), 0.5)
    assert np.isnan(M.position_share(0.515, 0.5))          # raw - chance <= 0.02
    assert np.isnan(M.position_share(0.45, 0.5))           # below chance
    assert M.position_share(0.7, 0.4) > 1                  # below chance at fixed position
    assert M.position_share(0.7, 0.75) < 0                 # better at fixed position
    arr = M.position_share(np.array([0.8, 0.51]), np.array([0.65, 0.5]))
    assert np.isclose(arr[0], 0.5) and np.isnan(arr[1])


def test_synthetic_known_answers():
    coords, P = simulate_pairs(n_genomes=30, n_genes=1500, seed=0)
    y = P['label'].to_numpy()
    g = P['genome_id'].to_numpy()
    k = gap_strata(P['same_strand'].to_numpy(), P['gap'].to_numpy())
    rs = Resample(g, 'cluster', n_boot=300, seed=0)
    out = {}
    for name in ('position_only', 'independent', 'mixed'):
        raw = M.auroc(y, P[name].to_numpy(), g, resample=rs)
        wit = M.within_auc(y, P[name].to_numpy(), k, g, resample=rs)
        out[name] = (raw, wit, M.position_share_result(raw, wit))
    raw, wit, sh = out['position_only']
    assert raw.estimate > 0.75
    assert abs(wit.estimate - 0.5) < 0.04          # residual gap signal inside 17 gap bins only
    assert sh.estimate > 0.85
    # a score that is a function of the stratum alone is exactly at chance within strata
    stratum_score = pd.Series(y).groupby(k).transform('mean').to_numpy()
    assert np.isclose(M.within_auc(y, stratum_score, k, g).estimate, 0.5)
    raw, wit, sh = out['independent']
    # non-collapsibility: a score unrelated to position ranks better within strata than overall when position
    # predicts the label, so the within-based share is negative
    assert wit.estimate > raw.estimate + 0.03 and sh.estimate < 0
    raw, wit, sh = out['mixed']
    assert 0.1 < sh.estimate < 0.95
    assert sh.ci[0] <= sh.estimate <= sh.ci[1]


def test_share_flags():
    rng = np.random.default_rng(0)
    n = 4000
    g = rng.integers(0, 20, n)
    y = rng.integers(0, 2, n)
    s = rng.normal(size=n)
    rs = Resample(g, 'cluster', n_boot=100, seed=0)
    raw = M.auroc(y, s, g, resample=rs)
    wit = M.within_auc(y, s, rng.integers(0, 3, n), g, resample=rs)
    sh = M.position_share_result(raw, wit)
    assert np.isnan(sh.estimate) and 'raw_near_chance' in sh.info['flags']


def test_ovr_and_concordance():
    rng = np.random.default_rng(0)
    n = 1500
    g = rng.integers(0, 5, n)
    k = rng.integers(0, 4, n)
    yc = rng.integers(0, 3, n)
    S = rng.normal(size=(n, 3)) + np.eye(3)[yc]
    r = M.within_auc_ovr(yc, S, k, g)
    ref = np.mean([M.within_auc((yc == c).astype(int), S[:, c], k, g).estimate for c in range(3)])
    assert np.isclose(r.estimate, ref)
    # concordance equals within_auc for binary labels, and brute force for continuous ones
    yb = rng.integers(0, 2, n)
    s = np.round(rng.normal(size=n) + yb, 1)
    assert np.isclose(M.concordance(yb, s, k, g).estimate, M.within_auc(yb, s, k, g).estimate)
    yv = np.round(rng.normal(size=300), 1)
    sv = np.round(yv + rng.normal(size=300), 1)
    num = den = 0.0
    for a in range(300):
        for b in range(a + 1, 300):
            if yv[a] == yv[b]:
                continue
            den += 1
            d = (yv[a] - yv[b]) * (sv[a] - sv[b])
            num += 1.0 if d > 0 else 0.5 if d == 0 else 0.0
    assert np.isclose(M.concordance(yv, sv).estimate, num / den)


def test_percentile_ci_with_infinite():
    lo, hi = percentile_ci(np.r_[np.linspace(0, 1, 99), np.inf])
    assert np.isfinite(lo) and np.isfinite(hi)
    lo, hi = percentile_ci(np.r_[np.linspace(0, 1, 90), [np.inf] * 10])
    assert np.isinf(hi)


def _expanded_share(y, s, strata, groups, mult):
    """Reference: re-score the data with every item repeated mult times (copies are distinct items)."""
    ix = np.repeat(np.arange(len(y)), mult)
    yy, ss, kk, gg = y[ix].astype(float), s[ix], strata[ix], groups[ix]
    num = den = 0.0
    for g in np.unique(gg):
        m = gg == g
        yg, sg, kg = yy[m], ss[m], kk[m]
        n = len(yg)
        R = pd.Series(sg).rank(method='average').to_numpy()
        c = R - (n + 1) / 2
        P = yg.sum()
        q = np.empty(n)
        for t in range(n):
            same = kg == kg[t]
            q[t] = (yg[same].sum() - yg[t]) / (same.sum() - 1) if same.sum() > 1 else (P - yg[t]) / (n - 1)
        num += (q * c).sum() / (P * (n - P))
        den += (yg * c).sum() / (P * (n - P))
    return num / den


def test_explained_share_identity_and_exactness(data):
    y, s, strata, groups = data
    r = M.explained_share(y, s, strata, groups)
    raw = M.auroc(y, s, groups)
    assert np.isclose(r.info['raw_excess'], raw.estimate - 0.5, atol=1e-12)        # E_g = AUROC_g - 1/2
    assert np.isclose(r.estimate, _expanded_share(y, s, strata, groups, np.ones(len(y), int)), atol=1e-12)
    rb = Resample(groups, 'block', n_boot=4, block_size=60, seed=0)
    r = M.explained_share(y, s, strata, groups, resample=rb)
    for b in range(4):
        mult = rb.counts[b, rb.unit_of_item]
        assert np.isclose(r.replicates[b], _expanded_share(y, s, strata, groups, mult), atol=1e-10)
    rc = Resample(groups, 'cluster', n_boot=30, seed=1)
    r = M.explained_share(y, s, strata, groups, resample=rc)
    for b in range(3):
        ix = np.concatenate([np.flatnonzero(groups == rc.group_labels[g]) for g in range(7) for _ in range(rc.counts[b, g])])
        gl = np.concatenate([np.full((groups == rc.group_labels[g]).sum(), 100 * g + c)
                             for g in range(7) for c in range(rc.counts[b, g])])     # copies are distinct genomes
        ref = _expanded_share(y[ix], s[ix], strata[ix], gl, np.ones(len(ix), int))
        assert np.isclose(r.replicates[b], ref, atol=1e-10)


def test_explained_share_known_answers():
    coords, P = simulate_pairs(n_genomes=30, n_genes=1500, seed=0)
    y = P['label'].to_numpy()
    g = P['genome_id'].to_numpy()
    k = gap_strata(P['same_strand'].to_numpy(), P['gap'].to_numpy())
    rs = Resample(g, 'cluster', n_boot=300, seed=0)
    sh = {n: M.explained_share(y, P[n].to_numpy(), k, g, resample=rs) for n in ('position_only', 'independent', 'mixed')}
    assert 0.97 < sh['position_only'].estimate < 1.03
    assert abs(sh['independent'].estimate) < 0.03
    assert 0.3 < sh['mixed'].estimate < 0.95 and sh['mixed'].ci[0] < sh['mixed'].estimate < sh['mixed'].ci[1]
    # a score that is a function of the stratum alone: 1 (exactly, apart from items alone in their stratum)
    stratum_score = pd.Series(np.random.default_rng(0).normal(size=k.max() + 1)).iloc[k].to_numpy()
    assert abs(M.explained_share(y, stratum_score, k, g).estimate - 1.0) < 1e-3
    big = np.isin(k, np.flatnonzero(np.bincount(k) > 2000))                # no singleton cells
    assert np.isclose(M.explained_share(y[big], stratum_score[big], k[big], g[big]).estimate, 1.0, atol=1e-12)
    # pure noise: raw AUROC at chance -> not reported
    noise = M.explained_share(y, np.random.default_rng(12345).normal(size=len(y)), k, g)
    assert np.isnan(noise.estimate) and 'raw_near_chance' in noise.info['flags']


def test_share_bound_criteria():
    """The headline on both causal worlds: 0 for scores without position information, true share inside the bound."""
    from seqleak.synthetic import simulate_pairs_consequence

    def std(x):
        return (x - x.mean()) / x.std()
    # position is a consequence of the label
    A = simulate_pairs_consequence(n_genomes=30, n_pairs=1500, seed=0)
    y, g = A.label.to_numpy(), A.genome_id.to_numpy()
    k = gap_strata(A.same_strand.to_numpy(), A.gap.to_numpy())
    assert M.position_share_bound(y, A.z.to_numpy(), k, g).estimate < 0.03
    assert M.position_share_bound(y, A.position_only.to_numpy(), k, g).estimate > 0.85
    a_free = M.auroc(y, A.z.to_numpy(), g).estimate
    prev = -1
    for lam in (0.25, 0.5, 0.75):
        s = lam * std(A.position_only.to_numpy()) + (1 - lam) * std(A.z.to_numpy())
        b = M.position_share_bound(y, s, k, g)
        true = 1 - (a_free - 0.5) / (b.info['raw'] - 0.5)
        assert b.estimate - 0.02 <= true <= b.info['share_upper'] + 0.02
        assert abs(b.info['share_within'] - true) < 0.02          # exact in this world
        assert b.estimate > prev
        prev = b.estimate
    # position is a cause of the label
    coords, P = simulate_pairs(n_genomes=30, n_genes=1500, seed=0)
    y, g = P.label.to_numpy(), P.genome_id.to_numpy()
    k = gap_strata(P.same_strand.to_numpy(), P.gap.to_numpy())
    b0 = M.position_share_bound(y, P.independent.to_numpy(), k, g)
    assert b0.estimate == 0.0 and 'clipped_low' in b0.info['flags'] and b0.info['share_upper'] < 0.05
    pos_part = np.where(P.same_strand, 2.0 - P.gap / 60.0, -3.0)
    a_free = M.auroc(y, P.independent.to_numpy(), g).estimate
    s = 0.5 * std(pos_part) + 0.5 * std(P.independent.to_numpy())
    rs = Resample(g, 'cluster', n_boot=200, seed=0)
    b = M.position_share_bound(y, s, k, g, resample=rs)
    true = 1 - (a_free - 0.5) / (b.info['raw'] - 0.5)
    assert b.estimate - 0.02 <= true <= b.info['share_upper'] + 0.02
    assert abs(b.info['share_equalized'] - true) < 0.02           # exact in this world
    assert b.ci[0] <= b.estimate <= b.ci[1]
    # rank-based: invariant to a monotone transform
    b2 = M.position_share_bound(y, np.exp(2 * s) + 3, k, g, resample=rs)
    assert b2.estimate == b.estimate and np.allclose(b2.replicates, b.replicates, equal_nan=True)
    # replicates equal the statistic recomputed on the resampled genomes (copies are separate genomes)
    u = M.equalized_score(s, k, g)
    for r in range(3):
        ix = np.concatenate([np.flatnonzero(g == rs.group_labels[j]) for j in range(len(rs.group_labels)) for _ in range(rs.counts[r, j])])
        gl = np.concatenate([np.full((g == rs.group_labels[j]).sum(), 1000 * j + c) for j in range(len(rs.group_labels))
                             for c in range(rs.counts[r, j])])
        raw = M.auroc(y[ix], s[ix], gl).estimate
        wit = M.within_auc(y[ix], s[ix], k[ix], gl).estimate
        eq = M.auroc(y[ix], u[ix], gl).estimate
        ref = np.clip(min(1 - (wit - .5) / (raw - .5), 1 - (eq - .5) / (raw - .5)), 0, 1)
        assert np.isclose(b.replicates[r], ref, atol=1e-10)
    # near chance: not reported
    weak = M.position_share_bound(y, np.random.default_rng(7).normal(size=len(y)), k, g)
    assert np.isnan(weak.estimate) and 'raw_near_chance' in weak.info['flags']


def test_explained_share_continuous_and_ovr(data):
    y, s, strata, groups = data
    b = M.explained_share(y, s, strata, groups)
    c = M.explained_share(y.astype(float), s, strata, groups, labels='continuous')
    assert np.isclose(b.estimate, c.estimate) and np.isclose(b.info['raw_excess'], c.info['raw_excess'])
    # continuous: raw excess = sum_{k<l} (y_k - y_l) sign(s_k - s_l) / 2 / sum_{k<l} |y_k - y_l|, per genome
    rng = np.random.default_rng(3)
    yc = np.round(s + rng.normal(size=len(s)), 1)
    r = M.explained_share(yc, s, strata, groups, labels='continuous')
    vals = []
    for g_ in range(7):
        m = groups == g_
        a, z = yc[m], s[m]
        dy = a[:, None] - a[None, :]
        vals.append((dy * np.sign(z[:, None] - z[None, :])).sum() / 4 / (np.abs(dy).sum() / 2))
    assert np.isclose(r.info['raw_excess'], np.mean(vals))
    # one-vs-rest: macro on the excess scale
    yk = rng.integers(0, 3, len(y))
    S = rng.normal(size=(len(y), 3)) + np.eye(3)[yk] + 0.3 * strata[:, None]
    rs = Resample(groups, 'cluster', n_boot=50, seed=0)
    o = M.explained_share_ovr(yk, S, strata, groups, resample=rs)
    parts = [M.explained_share((yk == c).astype(int), S[:, c], strata, groups) for c in range(3)]
    e = sum(p.info['raw_excess'] for p in parts)
    ep = sum(p.info['raw_excess'] * p.info['share_unreported'] for p in parts)
    assert np.isclose(o.estimate, ep / e) and o.ci[0] < o.estimate < o.ci[1]
