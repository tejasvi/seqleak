"""Unit tests of seqleak audit / clean: every check on small planted examples, plus the CLI."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from seqleak.audit import load_task, run_audit
from seqleak.audit.kmers import best_per_query, build_index, query
from seqleak.audit.stats import best_threshold, mcc_from_confusion, wilson
from seqleak.audit.synthetic import dinuc_shuffle, make_task, markov_seqs, segment_substitute
from seqleak.audit.twins import composition_twins, positional_twins
from seqleak.cli import main

SMALL = dict(n_train=800, n_test=300)


def audit_df(tmp_path, df, name='t', **kw):
    p = tmp_path / f'{name}.parquet'
    df.to_parquet(p)
    dist = kw.pop('distance_cols', None)
    seq_col = kw.pop('seq_col', None)
    task = load_task(data=[str(p)], distance_cols=dist, seq_col=seq_col)
    rep, items = run_audit(task, n_boot=50, quiet=True, homology='off', **kw)
    return rep, items


def status(rep, cid):
    return rep['checks'][cid]['status']


# ---------------------------------------------------------------- building blocks
def test_kmer_engine_overlap_and_rc():
    rng = np.random.default_rng(0)
    base = ''.join(rng.choice(list('ACGT'), 3000))
    ref = [base[:500], base[1000:1500], ''.join(rng.choice(list('ACGT'), 500))]
    rc = base[1000:1500][::-1].translate(str.maketrans('ACGT', 'TGCA'))
    q = [base[250:750], rc, ''.join(rng.choice(list('ACGT'), 500))]
    h = query(build_index(ref, k=21, scale=1), q)
    b = best_per_query(h, 3)
    c = b['shared'] / np.maximum(h.n_inf, 1)
    assert b['r'][0] == 0 and 0.45 < c[0] < 0.52 and b['max_coll_shift'][0] == -250
    assert b['r'][1] == 1 and c[1] == 1.0 and not b['max_coll_same'][1]
    assert b['r'][2] == -1


def test_twin_detectors():
    rng = np.random.default_rng(1)
    pos = markov_seqs(30, 300, rng)
    shuf = [dinuc_shuffle(s, rng) for s in pos[:10]]
    seg = [segment_substitute(s, rng) for s in pos[10:20]]
    seqs = pos + shuf + seg + markov_seqs(30, 300, rng)
    cp, ci = composition_twins(seqs)
    gp, gi = positional_twins(seqs)
    assert {tuple(p) for p in cp} == {(i, 30 + i) for i in range(10)}
    assert {tuple(p) for p in gp} >= {(10 + i, 40 + i) for i in range(10)}
    assert np.all(ci < 0.6) and np.all((gi >= 0.4) & (gi < 1.0))


def test_stats():
    t, d, m = best_threshold(np.array([1, 2, 3, 4, 5, 6.]), np.array([0, 0, 0, 1, 1, 1]))
    assert d == 1 and 3 <= t < 4 and m == pytest.approx(1.0)
    t, d, m = best_threshold(np.array([1, 2, 3, 4, 5, 6.]), np.array([1, 1, 1, 0, 0, 0]))
    assert d == -1 and m == pytest.approx(1.0)
    C = np.array([[50, 0], [0, 50]])
    assert mcc_from_confusion(C[None])[0] == pytest.approx(1.0)
    lo, hi = wilson(5, 100)
    assert lo < 0.05 < hi


# ---------------------------------------------------------------- checks
@pytest.mark.parametrize('plant,cid', [('exact', 'overlap.exact'), ('near', 'overlap.near'),
                                       ('shifted', 'overlap.windows'), ('shuffle', 'construction.shuffle'),
                                       ('segment', 'construction.segment')])
def test_planted_overlap_and_twins(tmp_path, plant, cid):
    df, _ = make_task(plant, 0.3, seed=3, **SMALL)
    rep, _ = audit_df(tmp_path, df)
    assert status(rep, cid) == 'FAIL', rep['checks'][cid]


def test_twin_direction(tmp_path):
    # same-label positional twins (near copies) are relatedness: WARN at most in construction.segment
    df, _ = make_task('near', 0.3, seed=16, **SMALL)
    rep, _ = audit_df(tmp_path, df)
    r = rep['checks']['construction.segment']
    assert r['status'] == 'WARN' and r['evidence']['label_relation_on_test'] == 'copy'
    assert status(rep, 'overlap.near') == 'FAIL'


def test_clean_control_passes(tmp_path):
    df, _ = make_task('none', 0.0, seed=4, **SMALL)
    rep, items = audit_df(tmp_path, df)
    assert rep['verdict']['status'] == 'PASS', rep['verdict']
    assert set(items['cv_fold'][items['split'] == 'train']) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize('plant,cid', [('length', 'shortcut.length'), ('npad', 'shortcut.ambiguous'),
                                       ('motif', 'shortcut.position_motif')])
def test_planted_shortcuts(tmp_path, plant, cid):
    df, info = make_task(plant, 0.9, seed=5, **SMALL)
    rep, _ = audit_df(tmp_path, df)
    assert info['oracle_mcc'] > 0.8
    assert status(rep, cid) == 'FAIL', rep['checks'][cid]


def test_imbalanced_motif_auroc(tmp_path):
    # 1:20 imbalance; every positive has GT at 150, 20% of negatives too: MCC about 0.4 (WARN), AUROC about 0.9 (FAIL)
    rng = np.random.default_rng(17)
    n = 4200
    y = (rng.random(n) < 1 / 21).astype(int)
    seqs = markov_seqs(n, 300, rng)
    gt = (y == 1) | (rng.random(n) < 0.2)
    seqs = [s[:150] + 'GT' + s[152:] if g else s[:150] + ('CA' if s[150:152] == 'GT' else s[150:152]) + s[152:]
            for s, g in zip(seqs, gt)]
    split = np.where(np.arange(n) < 3200, 'train', 'test')
    df = pd.DataFrame(dict(seq=seqs, label=y, split=split))
    rep, _ = audit_df(tmp_path, df)
    r = rep['checks']['shortcut.position_motif']
    assert r['status'] == 'FAIL' and r['evidence']['status_from'] == 'auroc' and r['evidence']['test_auroc'] > 0.85


def test_imbalanced_twins_chance_corrected(tmp_path):
    # near copies with labels drawn independently under 1:20 imbalance: copy agreement is about chance, so no WARN
    rng = np.random.default_rng(18)
    n_tr, n_te = 2000, 600
    seqs = markov_seqs(n_tr + n_te, 300, rng)
    y = (rng.random(n_tr + n_te) < 1 / 21).astype(int)
    for t in range(n_tr, n_tr + 300):
        seqs[t] = mutate_seq(seqs[t - n_tr], rng)
    df = pd.DataFrame(dict(seq=seqs, label=y, split=['train'] * n_tr + ['test'] * n_te))
    rep, _ = audit_df(tmp_path, df)
    r = rep['checks']['construction.segment']
    assert r['value'] >= 0.4 and r['status'] == 'PASS', r


def mutate_seq(s, rng):
    from seqleak.audit.synthetic import mutate
    return mutate(s, 0.05, rng)


def test_case_shortcut(tmp_path):
    df, _ = make_task('none', 0.0, seed=6, **SMALL)
    m = df['label'] == 1
    df.loc[m, 'seq'] = df.loc[m, 'seq'].str[:100].str.lower() + df.loc[m, 'seq'].str[100:]
    rep, _ = audit_df(tmp_path, df)
    assert status(rep, 'shortcut.case') == 'FAIL'


def test_composition_warns_never_fails(tmp_path):
    df, _ = make_task('control_gc', 0.0, seed=7, **SMALL)
    rep, _ = audit_df(tmp_path, df)
    assert status(rep, 'shortcut.composition') == 'WARN'
    assert status(rep, 'shortcut.kmer') == 'WARN'
    leak = [c for c in rep['verdict']['fail']]
    assert leak == []


def test_position_distance(tmp_path):
    df, info = make_task('position', 0.9, seed=8, **SMALL)
    rep, _ = audit_df(tmp_path, df, distance_cols=['dist'])
    assert info['oracle_auroc'] > 0.85
    assert status(rep, 'position.distance') == 'FAIL'
    df, _ = make_task('none', 0.0, seed=8, **SMALL)
    df['dist'] = np.random.default_rng(0).normal(size=len(df))
    rep, _ = audit_df(tmp_path, df, distance_cols=['dist'])
    assert status(rep, 'position.distance') == 'PASS'


def test_row_zero(tmp_path):
    from seqleak.synthetic import simulate_pairs
    from seqleak.tasks import write_task
    _, items = simulate_pairs(n_genomes=12, n_genes=400, seed=0)
    write_task(str(tmp_path / 'gp'), items.drop(columns=['z', 'position_only', 'independent', 'mixed']))
    df, _ = make_task('none', 0.0, seed=9, **SMALL)
    rep, _ = audit_df(tmp_path, df, task_dir=str(tmp_path / 'gp'), checks=['position'])
    r = rep['checks']['position.row_zero']
    assert r['status'] in ('WARN', 'FAIL') and r['value'] > 0.6


def test_windows_with_coordinates(tmp_path):
    df, _ = make_task('none', 0.0, seed=10, **SMALL)
    n = len(df)
    df['chrom'] = 'chr1'
    df['start'] = np.arange(n) * 1000
    df['end'] = df['start'] + 300
    te = np.flatnonzero(df['split'] == 'test')[:90]
    df.loc[te, 'start'] = df['start'].iloc[0] + 100     # 90 test windows overlap the first training window
    df.loc[te, 'end'] = df.loc[te, 'start'] + 300
    rep, _ = audit_df(tmp_path, df)
    r = rep['checks']['overlap.windows']
    assert r['status'] == 'FAIL' and r['evidence']['coordinates']['overlap'] == pytest.approx(90 / 300)


def test_pair_items(tmp_path):
    rng = np.random.default_rng(11)
    n_tr, n_te = 600, 200
    enh = markov_seqs(n_tr + n_te, 400, rng)
    pro = markov_seqs(n_tr + n_te, 300, rng)
    y = rng.integers(0, 2, n_tr + n_te)
    split = np.array(['train'] * n_tr + ['test'] * n_te)
    for t in range(n_tr, n_tr + 60):      # 30% of test pairs: shifted copies of both elements of a training pair
        j = t - n_tr
        enh[t] = enh[j][100:] + markov_seqs(1, 100, rng)[0]
        pro[t] = pro[j]
        y[t] = y[j]
    df = pd.DataFrame(dict(enhancer=enh, promoter=pro, label=y, split=split))
    rep, _ = audit_df(tmp_path, df, seq_col='enhancer,promoter')
    assert status(rep, 'overlap.pair') == 'FAIL'
    assert rep['checks']['overlap.pair']['value'] == pytest.approx(0.3)
    assert rep['task']['kind'] == 'multi_element'


def test_reference_copies(tmp_path):
    df, _ = make_task('none', 0.0, seed=12, **SMALL)
    ref = df.copy()
    ref['split'] = np.where(np.arange(len(ref)) % 2, 'train', 'test')
    p = tmp_path / 'ref.parquet'
    ref.to_parquet(p)
    from seqleak.audit import load_reference
    R = load_reference('ref', [str(p)])
    q = tmp_path / 'task.parquet'
    df.to_parquet(q)
    task = load_task(data=[str(q)])
    rep, _ = run_audit(task, [R], n_boot=50, quiet=True, homology='off')
    assert status(rep, 'reference.copies') == 'FAIL'


def test_lookup_baseline(tmp_path):
    df, _ = make_task('exact', 0.3, seed=13, **SMALL)
    rep, _ = audit_df(tmp_path, df)
    lb = rep['lookup_baseline']
    assert lb['value'] > 0.2 and lb['n_with_partner'] >= 90
    df, _ = make_task('none', 0.0, seed=13, **SMALL)
    rep, _ = audit_df(tmp_path, df)
    assert abs(rep['lookup_baseline']['value']) < 0.15


# ---------------------------------------------------------------- CLI
def test_cli_fasta_and_exit_codes(tmp_path):
    df, _ = make_task('exact', 0.2, seed=14, **SMALL)
    for sp in ('train', 'test'):
        for lab in (0, 1):
            with open(tmp_path / f'{sp}_{lab}.fa', 'w') as f:
                for _, r in df[(df['split'] == sp) & (df['label'] == lab)].iterrows():
                    f.write(f">{r['id']}\n{r['seq']}\n")
    out = tmp_path / 'rep.json'
    code = main(['audit', '--train', str(tmp_path / 'train_0.fa@0'), str(tmp_path / 'train_1.fa@1'),
                 '--test', str(tmp_path / 'test_0.fa@0'), str(tmp_path / 'test_1.fa@1'), '--out', str(out),
                 '--homology', 'off', '--n-boot', '20', '--fail-on', 'fail', '--quiet'])
    rep = json.load(open(out))
    assert code == 1 and rep['verdict']['status'] == 'FAIL' and rep['schema'] == 'seqleak.audit/1'
    assert os.path.exists(tmp_path / 'rep.md') and os.path.exists(tmp_path / 'rep.items.parquet')
    assert main(['audit', '--train', str(tmp_path / 'missing.fa'), '--test', str(tmp_path / 'x.fa'), '--out',
                 str(out), '--quiet']) == 2


@pytest.mark.parametrize('mode', ['filter', 'resplit'])
def test_cli_clean(tmp_path, mode):
    df, _ = make_task('shuffle', 0.3, seed=15, **SMALL)
    p = tmp_path / 'task.csv'
    df.to_csv(p, index=False)
    od = tmp_path / f'clean_{mode}'
    assert main(['clean', '--data', str(p), '--out-dir', str(od), '--mode', mode, '--homology', 'off',
                 '--n-boot', '20', '--quiet']) == 0
    rep = json.load(open(od / 'clean.json'))
    man = pd.read_csv(od / 'manifest.tsv', sep='\t')
    assert len(man) == len(df)
    after = rep['audit_after']['statuses']
    assert after['construction.shuffle'][0] != 'FAIL'
    assert after['overlap.exact'][0] == 'PASS'
    if mode == 'filter':
        assert (man['new_split'] == 'dropped').sum() >= 80
        assert (man.loc[man['orig_split'] == 'train', 'new_split'] == 'train').all()
    tr = pd.read_csv(od / 'train.csv')
    assert list(tr.columns) == list(df.columns)
