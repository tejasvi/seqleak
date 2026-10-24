"""Unit tests of the rules v1.1 additions: low-information inputs, constructed negatives inside splits, unmatched sites,
the motif guard, the published-relative criterion, pooled-library structure and test-only tasks."""
import numpy as np
import pandas as pd

from seqleak.audit import load_task, run_audit
from seqleak.audit.synthetic import dinuc_shuffle, markov_seqs

RNG = np.random.default_rng(42)


def audit_df(tmp_path, df, name='t', **kw):
    p = tmp_path / f'{name}.parquet'
    df.to_parquet(p)
    task = load_task(data=[str(p)])
    rep, items = run_audit(task, n_boot=50, quiet=True, homology='off', **kw)
    return rep, items


def base(n_tr=800, n_te=300, L=300, seed=0, pos_share=0.5):
    rng = np.random.default_rng(seed)
    n = n_tr + n_te
    seqs = markov_seqs(n, L, rng)
    y = (rng.random(n) < pos_share).astype(int)
    split = np.array(['train'] * n_tr + ['test'] * n_te)
    return seqs, y, split, rng


def test_lowinfo(tmp_path):
    seqs, y, split, rng = base(seed=1)
    te = np.flatnonzero(split == 'test')[:30]
    tr = np.flatnonzero(split == 'train')[:30]
    for i in np.r_[te, tr]:
        seqs[i] = 'N' * 300
        y[i] = 0
    rep, items = audit_df(tmp_path, pd.DataFrame(dict(seq=seqs, label=y, split=split)))
    assert rep['checks']['shortcut.lowinfo']['status'] == 'FAIL'
    assert rep['checks']['overlap.exact']['status'] == 'PASS'     # all-N windows are not counted as copies
    assert items['lowinfo'].sum() == 60


def test_synthetic_negatives_within_split(tmp_path):
    seqs, y, split, rng = base(seed=2)
    n = len(seqs)
    for i in range(0, n - 1, 2):      # every second item is a dinucleotide shuffle of the previous one, same split
        y[i], y[i + 1] = 1, 0
        seqs[i + 1] = dinuc_shuffle(seqs[i], rng)
        split[i + 1] = split[i]
    rep, _ = audit_df(tmp_path, pd.DataFrame(dict(seq=seqs, label=y, split=split)))
    assert rep['checks']['construction.shuffle']['value'] == 0.0
    assert rep['checks']['construction.synthetic_negatives']['status'] == 'FAIL'


def _site_task(seed, neg_gt_share, n_tr=1600, n_te=600, pos_share=0.5, flank=False, pflank=0.6, nflank=3):
    """Every positive has GT at 150 (a site); negatives have it with probability neg_gt_share; with ``flank``, positives
    also carry a consensus at the first ``nflank`` flanking positions with probability pflank each."""
    seqs, y, split, rng = base(n_tr, n_te, seed=seed, pos_share=pos_share)
    flanks = [(147, 'A'), (148, 'A'), (149, 'G'), (152, 'A'), (153, 'G'), (154, 'T')][:nflank]
    out = []
    for s, lab in zip(seqs, y):
        s = list(s)
        if lab == 1 or rng.random() < neg_gt_share:
            s[150:152] = 'GT'
        elif ''.join(s[150:152]) == 'GT':
            s[150:152] = 'CA'
        if flank and lab == 1:
            for p, b in flanks:
                if rng.random() < pflank:
                    s[p] = b
        out.append(''.join(s))
    return pd.DataFrame(dict(seq=out, label=y, split=split))


def test_unmatched_site(tmp_path):
    rep, _ = audit_df(tmp_path, _site_task(3, neg_gt_share=0.3))
    r = rep['checks']['construction.unmatched_site']
    assert r['status'] == 'FAIL' and r['evidence']['kmer'] in ('GT', 'G', 'T')
    rep, _ = audit_df(tmp_path, _site_task(4, neg_gt_share=1.0), name='decoy')
    assert rep['checks']['construction.unmatched_site']['status'] == 'PASS'


def test_motif_guard_under_imbalance(tmp_path):
    # 1:20 imbalance, every item has the GT (decoy negatives), positives carry a flanking consensus
    df = _site_task(5, neg_gt_share=1.0, n_tr=4000, n_te=1500, pos_share=1 / 21, flank=True)
    rep, _ = audit_df(tmp_path, df)
    m = rep['checks']['shortcut.position_motif']
    assert m['evidence']['test_auroc'] > 0.85
    assert m['status'] == 'WARN' and 'guard' in m['evidence']
    # same, but 40% of negatives lack the GT: the unmatched site lets the imbalance criterion FAIL
    df = _site_task(6, neg_gt_share=0.6, n_tr=4000, n_te=1500, pos_share=1 / 21, flank=True)
    rep, _ = audit_df(tmp_path, df, name='unmatched')
    assert rep['checks']['construction.unmatched_site']['status'] == 'FAIL'
    assert rep['checks']['shortcut.position_motif']['status'] == 'FAIL'


def test_published_relative(tmp_path):
    df = _site_task(7, neg_gt_share=1.0, flank=True)      # flanking consensus only: motif rule WARN on its own
    rep, _ = audit_df(tmp_path, df)
    m = rep['checks']['shortcut.position_motif']
    assert m['status'] == 'WARN', m
    mcc = m['value']
    rep, _ = audit_df(tmp_path, df, published_best=min(0.99, mcc / 0.8), name='rel_hi')
    assert rep['checks']['shortcut.position_motif']['status'] == 'FAIL'
    rep, _ = audit_df(tmp_path, df, published_best=min(0.99, mcc / 0.5), name='rel_lo')
    assert rep['checks']['shortcut.position_motif']['status'] == 'WARN'


def _library_task(seed, n_lib=3, single=False):
    rng = np.random.default_rng(seed)
    n_tr, n_te = 1500, 600
    n = n_tr + n_te
    backbones = markov_seqs(n_lib, 40, rng)
    var = markov_seqs(n, 120, rng)
    lib = rng.integers(0, n_lib, n) if not single else np.zeros(n, int)
    p_pos = np.array([0.9, 0.1, 0.5])[:n_lib]
    y = (rng.random(n) < p_pos[lib]).astype(int)
    seqs = [backbones[l] + v for l, v in zip(lib, var)]
    return pd.DataFrame(dict(seq=seqs, label=y, split=['train'] * n_tr + ['test'] * n_te))


def test_library(tmp_path):
    rep, _ = audit_df(tmp_path, _library_task(8))
    r = rep['checks']['construction.library']
    assert r['status'] == 'FAIL' and r['evidence']['n_backbones'] >= 2, r
    rep, _ = audit_df(tmp_path, _library_task(9, single=True), name='single')
    assert rep['checks']['construction.library']['status'] == 'PASS'


def test_test_only_task(tmp_path):
    seqs, y, split, rng = base(n_tr=0, n_te=900, seed=10)
    pos = np.flatnonzero(y == 1)
    for i in pos[rng.random(len(pos)) < 0.9]:
        seqs[i] = seqs[i][:240]                          # length shortcut
    df = pd.DataFrame(dict(seq=seqs, label=y, split='test'))
    rep, _ = audit_df(tmp_path, df)
    assert rep['task']['mode'] == 'test_only'
    assert rep['checks']['overlap.exact']['status'] == 'SKIP'
    assert rep['checks']['shortcut.length']['status'] == 'FAIL'
    assert rep['checks']['baseline.lookup']['value'] is None
    assert rep['task']['n'] == dict(train=0, dev=0, test=900)


def test_unmatched_site_multiclass_warns(tmp_path):
    """Three classes, one defined by a site the others lack (donor vs acceptor vs neither): WARN, not FAIL."""
    seqs, _, split, rng = base(seed=11)
    y = np.arange(len(seqs)) % 3
    for i in range(len(seqs)):
        s_ = list(seqs[i])
        if y[i] == 1:
            s_[150:152] = 'GT'
        elif ''.join(s_[150:152]) == 'GT':
            s_[150:152] = 'CA'
        seqs[i] = ''.join(s_)
    rep, _ = audit_df(tmp_path, pd.DataFrame(dict(seq=seqs, label=y, split=split)))
    c = rep['checks']['construction.unmatched_site']
    assert c['value'] >= 0.25 and c['status'] == 'WARN' and 'more than two classes' in c['reason']


def test_robustness_rna_classes_backbones():
    """RNA input (U) is not low-information; the majority label per sequence matches a full (sequence x class) table;
    many classes and many shared segments stay bounded."""
    from seqleak.audit.construct import acgt_fraction, library_groups, opposite_twin_items, unmatched_site
    rng = np.random.default_rng(3)
    assert np.allclose(acgt_fraction(['ACGU' * 25, 'uuuuacgt', 'NNNNACGT', '']), [1.0, 1.0, 0.5, 0.0])
    uid = rng.integers(0, 50, 400)
    yi = rng.integers(0, 4, 400)
    pairs = rng.integers(0, 50, (60, 2))
    hist = np.zeros((50, 4), np.int64)
    np.add.at(hist, (uid, yi), 1)
    lab = hist.argmax(1)
    mark, npairs, _ = opposite_twin_items(pairs, uid, yi, 4)
    want = np.zeros(50, bool)
    opp = lab[pairs[:, 0]] != lab[pairs[:, 1]]
    want[pairs[opp, 0]] = True
    want[pairs[opp, 1]] = True
    assert npairs == 60 and (mark == want[uid]).all()
    seqs = markov_seqs(600, 60, rng)
    y = np.arange(600) % 30                                    # 30 classes: at most 20 are tried
    r = unmatched_site(seqs, y, 30, np.arange(400), np.arange(400, 600))
    assert r is not None and 'statistic' in r
    # 70 distinct 30-bp segments, each in its own 1/70 of the items plus a shared one in all: at most 62 backbones
    segs = markov_seqs(70, 30, rng)
    lib = np.arange(7000) % 70
    body = markov_seqs(7000, 40, rng)
    items = [segs[g] + b for g, b in zip(lib, body)]
    sig, bbs = library_groups(items, min_df=0.01, max_items=7000)
    assert len(bbs) <= 62 and sig.dtype == np.int64
