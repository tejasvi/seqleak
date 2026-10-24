import os

import numpy as np
import pandas as pd

import seqleak
from seqleak import features as F
from seqleak.baseline import make_folds, position_model
from seqleak.metrics import auroc
from seqleak.strata import make_strata, pair_strata
from seqleak.synthetic import simulate_genomes, simulate_pairs


def brute_pair(coords, i, j):
    st, en = coords.start.to_numpy(), coords.end.to_numpy()
    sd = np.where(coords.strand.to_numpy() > 0, 1, -1)
    a, b = min(i, j), max(i, j)
    gaps = [max(st[t + 1] - en[t], 0) for t in range(a, b)]
    return dict(genes_apart=b - a, same_strand=sd[a] == sd[b], gap=st[b] - en[a],
                max_intervening_gap=max(gaps) if gaps else 0,
                strand_switches=sum(sd[t + 1] != sd[t] for t in range(a, b)))


def test_pair_features_brute_force():
    coords = simulate_genomes(n_genomes=3, n_genes=300, seed=2)
    rng = np.random.default_rng(0)
    gid = rng.choice(coords.genome_id.unique(), 400)
    i = rng.integers(0, 300, 400)
    j = np.clip(i + rng.integers(-150, 150, 400), 0, 299)
    out = F.pair_features(coords, i, j, genome_id=gid, contig=np.zeros(400, int))
    for t in range(400):
        ref = brute_pair(coords[coords.genome_id == gid[t]].reset_index(drop=True), i[t], j[t])
        for k, v in ref.items():
            assert out[k].iloc[t] == v, (k, t)


def test_adjacent_pairs_and_strata():
    coords = simulate_genomes(n_genomes=2, n_genes=100, seed=1)
    P = F.adjacent_pairs(coords)
    assert len(P) == 2 * 99
    assert (P.genes_apart == 1).all() and (P.strand_switches == (~P.same_strand).astype(int)).all()
    s = make_strata(P)                      # auto -> adjacent scheme
    assert set(np.unique(s // 100)) <= {0, 1}
    q = pair_strata(np.array([1, 2, 1000]), np.array([0, 5, 1]), np.array([0, 3000, 120]), np.array([True, False, True]))
    assert len(np.unique(q)) == 3


def test_gene_features():
    coords = pd.DataFrame(dict(start=[100, 400, 900, 2000, 2600, 3500], end=[350, 850, 1700, 2500, 3300, 3900],
                               strand=['+', '+', '+', '-', '-', '+']))
    gf = F.gene_features(coords, origin=0, chrom_length=8000)
    assert list(gf.directon_size) == [3, 3, 3, 2, 2, 1]
    assert list(gf.directon_pos) == [1, 2, 3, 3, 1, 0]       # minus-strand directon starts at its right end
    assert gf.gap_up.isna().iloc[0] and gf.gap_up.iloc[1] == 400 - 350
    assert gf.gap_up.iloc[3] == 2600 - 2500                  # minus strand: upstream neighbour is the next gene
    assert (gf.replication_strand == np.array([1, 1, 1, 0, 0, 1])).all()   # all genes in the right replichore
    assert np.all((gf.dist_origin >= 0) & (gf.dist_origin <= 1))
    assert np.isclose(gf.rule_down.iloc[0], np.exp(-50 / 500))


def test_origin_from_gc_skew():
    rng = np.random.default_rng(0)
    L = 20000
    # leading strand G-rich between origin (5000) and terminus (15000)
    seq = np.where(rng.random(L) < 0.5, 'A', 'T').astype('<U1')
    gc = rng.random(L) < 0.5
    right = (np.arange(L) >= 5000) & (np.arange(L) < 15000)
    seq[gc] = np.where(right[gc] ^ (rng.random(gc.sum()) < 0.3), 'G', 'C')
    ori, ter = F.origin_from_gc_skew(''.join(seq))
    assert abs(ori - 5000) < 300 and abs(ter - 15000) < 300


def test_folds_are_grouped():
    g = np.repeat(np.arange(23), 10)
    f = make_folds(g, 5, seed=0)
    for k in np.unique(g):
        assert len(np.unique(f[g == k])) == 1
    rng = np.random.default_rng(0)
    G = np.unique(g)
    fold_of = dict(zip(G, rng.permutation(len(G)) % 5))       # the v1 recipe (src/analysis_ppi2.py)
    assert np.array_equal(f, np.array([fold_of[x] for x in g]))


def test_position_model_learns_position_not_latent():
    coords, P = simulate_pairs(n_genomes=12, n_genes=800, seed=3)
    y, g = P.label.to_numpy(), P.genome_id.to_numpy()
    oracle = auroc(y, P.position_only.to_numpy(), g).estimate      # the true position effect
    pos = position_model(P, y, g, folds=4)
    assert auroc(y, pos, g).estimate > oracle - 0.01
    pe = position_model(P, y, g, folds=4, extra=P.independent.to_numpy())
    assert auroc(y, pe, g).estimate > auroc(y, pos, g).estimate + 0.02
    gbm = position_model(P, y, g, folds=4, model='gbm')
    assert auroc(y, gbm, g).estimate > oracle - 0.02


def test_single_genome_oof_auroc_is_fold_stratified():
    """Pooling out-of-fold predictions of different fold models biases the AUROC down for a coarse feature;
    comparing items only within their fold recovers the AUROC of a model fitted on independent data."""
    from seqleak.metrics import within_auc

    def genome(n, seed):
        rng = np.random.default_rng(seed)
        d = rng.integers(0, 6, n).astype(float)                    # coarse feature, weak effect: many near-ties
        y = (rng.random(n) < np.where(d < 3, 0.6, 0.4)).astype(int)
        return pd.DataFrame(dict(d=d)), y
    F, y = genome(6000, 1)
    g = np.zeros(len(y), int)
    order = np.arange(len(y))
    pooled, strat = [], []
    for k in (5, 20):
        fold = make_folds(g, f'contig{k}', order=order)
        pred = position_model(F, y, g, spec='generic', folds=f'contig{k}', order=order)
        pooled.append(auroc(y, pred, g).estimate)
        strat.append(within_auc(y, pred, fold, g).estimate)
    Ftr, ytr = genome(60000, 2)                                  # independent training data, same process
    both = pd.concat([Ftr, F], ignore_index=True)
    yb = np.r_[ytr, y]
    pred_tr = position_model(both, yb, np.r_[np.zeros(len(ytr)), np.ones(len(y))], spec='generic', folds=None,
                             train_mask=np.r_[np.ones(len(ytr), bool), np.zeros(len(y), bool)])[len(ytr):]
    ref = auroc(y, pred_tr, g).estimate
    assert pooled[1] < strat[1] - 0.005                          # pooling across fold models biases the AUROC down
    for p_, s_ in zip(pooled, strat):                            # the fold-stratified AUROC is closer to the reference
        assert abs(s_ - ref) < abs(p_ - ref)


def test_row_zero_end_to_end(tmp_path):
    coords, P = simulate_pairs(n_genomes=12, n_genes=600, seed=4)
    items = P[['genome_id', 'contig', 'i', 'j', 'label', 'genes_apart', 'same_strand', 'gap', 'max_intervening_gap',
               'strand_switches']]
    seqleak.tasks.write_task(str(tmp_path / 'toy'), items,
                              scores={m: P[m].to_numpy() for m in ('position_only', 'independent', 'mixed')},
                              task_md='toy\n', scores_readme='synthetic\n')
    df = seqleak.row_zero(str(tmp_path / 'toy'), n_boot=200, cache_dir=str(tmp_path / 'cache'))
    assert list(df.scorer[:2]) == [seqleak.report.ROW_ZERO, seqleak.report.RULE]
    d = df.set_index('scorer')
    assert d.loc['position_only', 'share'] > 0.85                # up to the position signal left inside strata
    assert d.loc['independent', 'share'] < 0.05 and d.loc['independent', 'share_upper'] < 0.1
    assert d.loc['mixed', 'share'] <= d.loc['mixed', 'share_upper']
    assert d.loc[seqleak.report.ROW_ZERO, 'share'] > 0.85       # strata capture most of the position model's signal
    assert d.loc['independent', 'gain'] > 0.01 and d.loc['independent', 'gain_p'] < 0.01
    assert abs(d.loc['position_only', 'gain']) < 0.01
    txt = seqleak.format_table(df)
    assert 'row zero' in txt
    # cached second run gives the same table
    df2 = seqleak.row_zero(str(tmp_path / 'toy'), n_boot=200, cache_dir=str(tmp_path / 'cache'))
    assert np.allclose(df.select_dtypes('number').to_numpy(), df2.select_dtypes('number').to_numpy(), equal_nan=True)
    assert len(os.listdir(tmp_path / 'cache' / 'toy')) == 4
    # few genomes -> block bootstrap and leave-one-genome-out
    small = items[items.genome_id.isin(['g000', 'g001', 'g002', 'g003'])]
    seqleak.tasks.write_task(str(tmp_path / 'small'), small.reset_index(drop=True),
                              scores={'mixed': P.loc[small.index, 'mixed'].to_numpy()})
    ds = seqleak.row_zero(str(tmp_path / 'small'), n_boot=100)
    assert ds.attrs['method'] == 'block' and ds.attrs['folds'] == 'logo'
    assert np.isfinite(ds.set_index('scorer').loc['mixed', 'within_lo'])


def test_gene_preset_counts_length_layout_columns_do_not():
    coords = simulate_genomes(n_genomes=12, n_genes=400, seed=5)
    feats = pd.concat([F.gene_features(c[['start', 'end', 'strand']].reset_index(drop=True), origin=0,
                                       chrom_length=int(c.end.max()) + 1000)
                       for _, c in coords.groupby('genome_id', sort=True)], ignore_index=True)
    genome = coords.sort_values('genome_id', kind='stable').genome_id.to_numpy()
    lz = np.log(feats.length.to_numpy(float))
    y = (np.random.default_rng(0).random(len(feats)) < 1 / (1 + np.exp(-3 * (lz - lz.mean())))).astype(int)
    score = {'length': feats.length.to_numpy(float)}            # labels and score driven by gene length alone
    layout = ['strand', 'replication_strand', 'dist_origin', 'directon_pos', 'directon_size', 'operon_rule', 'density']
    a = seqleak.report.evaluate(y, genome, feats, score, n_boot=0)
    b = seqleak.report.evaluate(y, genome, feats, score, position_columns=layout, n_boot=0)
    assert a.attrs['scheme'] == 'gene' and b.attrs['scheme'] == 'prognostic'
    ga, gb = a.set_index('scorer').loc['length'], b.set_index('scorer').loc['length']
    assert ga.share > 0.5 and abs(ga.gain) < 0.01      # the 'gene' preset counts gene length as position
    assert gb.share < 0.2 and gb.gain > 0.1            # the layout-only columns do not
