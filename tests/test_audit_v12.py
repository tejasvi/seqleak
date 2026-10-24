"""Unit tests of the rules v1.2 additions: genome membership (k-mer containment against reference FASTA files),
synthetic_negatives scored on the test split, and motif tables anchored at the ends of variable-length items."""
import gzip
import shutil

import numpy as np
import pandas as pd
import pytest

from seqleak.audit import load_task, run_audit
from seqleak.audit import genome as G
from seqleak.audit.synthetic import dinuc_shuffle, markov_seqs

COMP = str.maketrans('ACGT', 'TGCA')


def write_ref(path, seqs, width=60, gz=False):
    op = gzip.open if gz else open
    with op(path, 'wt') as f:
        for i, s in enumerate(seqs):
            f.write(f'>chr{i}\n')
            for j in range(0, len(s), width):
                f.write(s[j:j + width] + '\n')


def windows(ref, n, L, rng):
    out = []
    for _ in range(n):
        s = ref[int(rng.integers(0, len(ref) - L)):][:L]
        out.append(s if rng.random() < 0.5 else s.translate(COMP)[::-1])
    return out


def audit(tmp_path, df, name='t', **kw):
    p = tmp_path / f'{name}.parquet'
    df.to_parquet(p)
    task = load_task(data=[str(p)], anchor_col=kw.pop('anchor_col', None))
    return run_audit(task, n_boot=50, quiet=True, homology='off', **kw)


def genome_task(rng, ref, synthetic=True, n=600, L=200):
    pos = windows(ref, n, L, rng)
    neg = [dinuc_shuffle(s, rng) for s in windows(ref, n, L, rng)] if synthetic else windows(ref, n, L, rng)
    seqs = pos + neg
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    split = np.where(rng.random(2 * n) < 0.75, 'train', 'test')
    return pd.DataFrame(dict(seq=seqs, label=y, split=split))


def test_genome_membership(tmp_path):
    rng = np.random.default_rng(0)
    ref = markov_seqs(1, 400_000, rng)[0]
    write_ref(tmp_path / 'ref.fa.gz', [ref[:250_000], ref[250_000:]], width=61, gz=True)
    rep, items = audit(tmp_path, genome_task(rng, ref, True), genome=[str(tmp_path / 'ref.fa.gz')])
    c = rep['checks']['construction.genome_membership']
    assert c['status'] == 'FAIL' and c['value'] > 0.95
    assert c['evidence']['shares']['test']['1']['members'] > 0.95 and c['evidence']['shares']['test']['0']['members'] < 0.05
    assert 'genome_membership' in items.columns
    rep, _ = audit(tmp_path, genome_task(rng, ref, False), name='c', genome=[str(tmp_path / 'ref.fa.gz')])
    assert rep['checks']['construction.genome_membership']['status'] == 'PASS'


def test_genome_membership_skips_and_registry(tmp_path):
    rng = np.random.default_rng(1)
    ref = markov_seqs(1, 100_000, rng)[0]
    write_ref(tmp_path / 'r.fa', [ref])
    df = genome_task(rng, ref, True, n=200)
    rep, _ = audit(tmp_path, df)
    assert rep['checks']['construction.genome_membership']['status'] == 'SKIP'
    rep, _ = audit(tmp_path, df, genome=[str(tmp_path / 'missing.fa')])
    assert 'not found' in rep['checks']['construction.genome_membership']['reason']
    short = genome_task(rng, ref, True, n=200, L=25)
    rep, _ = audit(tmp_path, short, name='s', genome=[str(tmp_path / 'r.fa')])
    c = rep['checks']['construction.genome_membership']
    assert c['status'] == 'SKIP' and 'too short' in c['reason']
    (tmp_path / 'reg.tsv').write_text(f'toy\t{tmp_path / "r.fa"}\n')
    rep, _ = audit(tmp_path, df, name='g', genome=['toy'], genome_registry=str(tmp_path / 'reg.tsv'))
    assert rep['checks']['construction.genome_membership']['status'] == 'FAIL'


def test_genome_membership_two_references(tmp_path):
    """Classes drawn from two organisms: with one reference membership predicts the label; with both it does not."""
    rng = np.random.default_rng(2)
    a, b = markov_seqs(2, 150_000, rng)
    write_ref(tmp_path / 'a.fa', [a])
    write_ref(tmp_path / 'b.fa', [b])
    seqs = windows(a, 400, 200, rng) + windows(b, 400, 200, rng)
    df = pd.DataFrame(dict(seq=seqs, label=np.r_[np.ones(400, int), np.zeros(400, int)],
                           split=np.where(rng.random(800) < 0.75, 'train', 'test')))
    rep, _ = audit(tmp_path, df, genome=[str(tmp_path / 'a.fa')])
    assert rep['checks']['construction.genome_membership']['status'] == 'FAIL'
    rep, _ = audit(tmp_path, df, name='ab', genome=[str(tmp_path / 'a.fa'), str(tmp_path / 'b.fa')])
    assert rep['checks']['construction.genome_membership']['status'] == 'PASS'


def test_kmer_membership_matches_brute_force(tmp_path):
    rng = np.random.default_rng(3)
    recs = [''.join(rng.choice(list('ACGT'), 30_000)), ''.join(rng.choice(list('ACGT'), 20_000))]
    recs[1] = recs[1][:500] + 'N' * 30 + recs[1][530:].lower()
    write_ref(tmp_path / 'x.fa', recs, width=1_000_000)          # single-line records
    ks = set()
    for r in recs:
        R = r.upper()
        for i in range(len(R) - 20):
            w = R[i:i + 21]
            if 'N' not in w:
                ks.add(min(w, w.translate(COMP)[::-1]))
    items = windows(recs[0], 50, 120, rng) + [dinuc_shuffle(s, rng) for s in windows(recs[0], 50, 120, rng)] + ['ACGTN' * 3]
    memb, info = G.kmer_membership(items, [str(tmp_path / 'x.fa')], k=21, per_item=32)

    def frac(s):
        n = len(s) - 20
        m = min(32, n)
        if m <= 0:
            return np.nan
        ws = [s[i:i + 21] for i in ((j * (len(s) - 21)) // max(m - 1, 1) for j in range(m)) if 'N' not in s[i:i + 21]]
        return np.mean([min(w, w.translate(COMP)[::-1]) in ks for w in ws]) if ws else np.nan
    assert np.allclose(memb, [frac(s) for s in items], equal_nan=True)
    assert info['unplaced_items'] == 1


@pytest.mark.skipif(shutil.which('minimap2') is None, reason='minimap2 not installed')
def test_genome_membership_minimap2(tmp_path):
    rng = np.random.default_rng(4)
    ref = markov_seqs(1, 200_000, rng)[0]
    write_ref(tmp_path / 'm.fa', [ref])
    rep, _ = audit(tmp_path, genome_task(rng, ref, True, n=300), genome=[str(tmp_path / 'm.fa')], genome_method='minimap2')
    assert rep['checks']['construction.genome_membership']['status'] == 'FAIL'


def test_synthetic_negatives_on_test_split(tmp_path):
    """Shuffled negatives only inside a small test split: a small share of all items, but every test item."""
    rng = np.random.default_rng(5)
    n_tr, n_pairs = 3000, 200
    tr = markov_seqs(n_tr, 200, rng)
    pos = markov_seqs(n_pairs, 200, rng)
    seqs = tr + pos + [dinuc_shuffle(s, rng) for s in pos]
    y = np.r_[(rng.random(n_tr) < 0.5).astype(int), np.ones(n_pairs, int), np.zeros(n_pairs, int)]
    split = np.r_[np.array(['train'] * n_tr), np.array(['test'] * (2 * n_pairs))]
    rep, _ = audit(tmp_path, pd.DataFrame(dict(seq=seqs, label=y, split=split)))
    c = rep['checks']['construction.synthetic_negatives']
    assert c['status'] == 'FAIL' and c['value'] > 0.9 and c['evidence']['share_all_items'] < 0.15


def test_anchored_motif(tmp_path):
    rng = np.random.default_rng(6)
    n = 800
    lens = rng.integers(150, 400, n)
    seqs = [markov_seqs(1, int(L), rng)[0] for L in lens]
    y = (rng.random(n) < 0.5).astype(int)
    for i in np.flatnonzero(y == 1):
        s = seqs[i]
        seqs[i] = s[:len(s) - 30] + 'TATA' + s[len(s) - 26:]          # motif 30 bases from the end
    split = np.where(rng.random(n) < 0.75, 'train', 'test')
    rep, _ = audit(tmp_path, pd.DataFrame(dict(seq=seqs, label=y, split=split)))
    c = rep['checks']['shortcut.anchored_motif']
    assert c['status'] == 'FAIL' and 'end-' in c['reason']
    assert rep['checks']['shortcut.position_motif']['status'] == 'SKIP'
    # control: same lengths, no motif
    ctrl = [markov_seqs(1, int(L), rng)[0] for L in lens]
    rep, _ = audit(tmp_path, pd.DataFrame(dict(seq=ctrl, label=y, split=split)), name='c')
    assert rep['checks']['shortcut.anchored_motif']['status'] == 'PASS'
    # a site column: motif 5 bases after a per-item site position
    site = rng.integers(20, 120, n)
    seqs2 = [markov_seqs(1, int(L), rng)[0] for L in lens]
    for i in np.flatnonzero(y == 1):
        s = seqs2[i]
        seqs2[i] = s[:site[i] + 5] + 'GATC' + s[site[i] + 9:]
    rep, _ = audit(tmp_path, pd.DataFrame(dict(seq=seqs2, label=y, split=split, site=site)), name='a', anchor_col='site')
    c = rep['checks']['shortcut.anchored_motif']
    assert c['status'] == 'FAIL' and 'from the site' in c['reason']


def test_anchored_motif_skips_fixed_length(tmp_path):
    rng = np.random.default_rng(7)
    seqs = markov_seqs(400, 120, rng)
    df = pd.DataFrame(dict(seq=seqs, label=(rng.random(400) < 0.5).astype(int),
                           split=np.where(rng.random(400) < 0.75, 'train', 'test')))
    rep, _ = audit(tmp_path, df)
    assert rep['checks']['shortcut.anchored_motif']['status'] == 'SKIP'


def test_single_class_training_split(tmp_path):
    """Training split with one class (positives only), test split with genomic positives and shuffled negatives: the
    rules are cross-validated within the test items, so membership and the twin statistic still find the construction."""
    rng = np.random.default_rng(8)
    ref = markov_seqs(1, 300_000, rng)[0]
    write_ref(tmp_path / 's.fa', [ref])
    tr = windows(ref, 600, 200, rng)
    pos = windows(ref, 200, 200, rng)
    seqs = tr + pos + [dinuc_shuffle(x, rng) for x in pos]
    y = np.r_[np.ones(600, int), np.ones(200, int), np.zeros(200, int)]
    split = np.r_[np.array(['train'] * 600), np.array(['test'] * 400)]
    rep, _ = audit(tmp_path, pd.DataFrame(dict(seq=seqs, label=y, split=split)), genome=[str(tmp_path / 's.fa')])
    assert rep['task']['single_class_train'] is True
    c = rep['checks']['construction.genome_membership']
    assert c['status'] == 'FAIL' and 'training split has one class' in c['reason']
    assert rep['checks']['construction.synthetic_negatives']['status'] == 'FAIL'
