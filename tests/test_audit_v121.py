"""Regression tests of rules v1.2.1: in multi-class tasks the one-feature rule (stats.fit_bins) gives a point mass at
the largest value its own bin. Up to v1.2 that value shared the last bin with the values below it, so a feature with
two values gave a single bin, a constant prediction and MCC 0 (a seven-class task whose negatives were all absent
from the genome passed the genome-membership check)."""
import numpy as np
import pandas as pd

from seqleak.audit import load_task, run_audit
from seqleak.audit import stats as S
from seqleak.audit.shortcuts import rule_eval
from seqleak.audit.synthetic import dinuc_shuffle, markov_seqs

COMP = str.maketrans('ACGT', 'TGCA')
# class sizes in the proportions of the seven-class task of the spike-in study (1008, 753, 103, 65, 58, 17, 9)
SIZES = (250, 188, 26, 16, 15, 5, 4)


def fit_bins_v12(x, yi, K, nb=64):
    """The rule as frozen in v1.0-v1.2, for comparison."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    b = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, max(len(edges) - 2, 0))
    glob = int(np.argmax(np.bincount(yi, minlength=K)))
    cnt = np.zeros((max(len(edges) - 1, 1), K), np.int64)
    np.add.at(cnt, (b, yi), 1)
    return edges, np.where(cnt.sum(1) > 0, cnt.argmax(1), glob)


def two_valued(sizes=SIZES):
    yi = np.repeat(np.arange(len(sizes)), sizes)
    return (yi != 0).astype(np.float64), yi


def test_two_valued_feature_gets_two_bins():
    x, yi = two_valued()
    edges, lab = S.fit_bins(x, yi, len(SIZES))
    pred = S.apply_bins(x, edges, lab)
    assert (pred[x == 0] == 0).all() and (pred[x == 1] == 1).all()
    old = S.apply_bins(x, *fit_bins_v12(x, yi, len(SIZES)))
    assert len(np.unique(old)) == 1                      # the v1.2 defect on the same input
    for n in (70, 700):                                  # every split of n items into two values
        for k in range(1, n):
            x = np.r_[np.zeros(k), np.ones(n - k)]
            yi = (x > 0).astype(np.int64)
            pred = S.apply_bins(x, *S.fit_bins(x, yi, 3))
            if np.quantile(x, 63 / 64) == 1:             # the larger value holds at least one quantile step
                assert (pred == yi).all(), (n, k)
            else:                                        # a rarer largest value shares the last bin, as before
                assert np.array_equal(pred, S.apply_bins(x, *fit_bins_v12(x, yi, 3))), (n, k)


def test_unchanged_without_point_mass_at_maximum():
    rng = np.random.default_rng(0)
    for n, K in ((40, 3), (500, 4), (3000, 7)):
        yi = rng.integers(0, K, n)
        for x in (rng.normal(size=n), rng.integers(0, 50, n).astype(float) + np.r_[np.zeros(n - 1), 100.0],
                  np.round(rng.random(n), 2) * (rng.random(n) < 0.9)):
            if np.quantile(x, 63 / 64) == x.max():
                continue
            xt = np.r_[x, rng.normal(size=200) * 3]
            e1, l1 = S.fit_bins(x, yi, K)
            e0, l0 = fit_bins_v12(x, yi, K)
            assert np.array_equal(e1, e0) and np.array_equal(l1, l0)
            assert np.array_equal(S.apply_bins(xt, e1, l1), S.apply_bins(xt, e0, l0))


def test_rule_eval_two_valued_multiclass():
    x, yi = two_valued()
    n = len(x)
    folds = np.arange(n) % 5
    allm = np.ones(n, bool)
    r = rule_eval(x, yi, len(SIZES), allm, allm, folds, n_boot=20, cv_as_test=True)
    assert r['test'] > 0.7 and r['cv'] > 0.7


def write_ref(path, seqs, width=60):
    with open(path, 'w') as f:
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


def seven_class_task(rng, ref, synthetic, test_only, L=200):
    seqs, labels = [], []
    for c, m in enumerate(SIZES):
        w = windows(ref, m, L, rng)
        seqs += [dinuc_shuffle(s, rng) for s in w] if (synthetic and c == 0) else w
        labels += [f'Sigma{c:02d}'] * m
    split = np.full(len(seqs), 'test') if test_only else np.where(rng.random(len(seqs)) < 0.75, 'train', 'test')
    return pd.DataFrame(dict(seq=seqs, label=labels, split=split))


def membership(tmp_path, df, name):
    p = tmp_path / f'{name}.parquet'
    df.to_parquet(p)
    rep, _ = run_audit(load_task(data=[str(p)]), n_boot=50, quiet=True, homology='off',
                       genome=[str(tmp_path / 'ref.fa')], checks=['construction.genome_membership'])
    return rep['checks']['construction.genome_membership']


def test_genome_membership_seven_classes(tmp_path):
    rng = np.random.default_rng(3)
    ref = markov_seqs(1, 300_000, rng)[0]
    write_ref(tmp_path / 'ref.fa', [ref])
    for test_only in (True, False):
        c = membership(tmp_path, seven_class_task(rng, ref, True, test_only), f's{int(test_only)}')
        sh = c['evidence']['shares']['test']
        assert sh['Sigma00']['mean_membership'] < 0.05
        assert min(v['mean_membership'] for k, v in sh.items() if k != 'Sigma00' and v['n']) > 0.95
        assert c['status'] == 'FAIL' and c['value'] > 0.7, (test_only, c['status'], c['value'])
        c = membership(tmp_path, seven_class_task(rng, ref, False, test_only), f'g{int(test_only)}')
        assert c['status'] == 'PASS', (test_only, c['status'], c['reason'])
