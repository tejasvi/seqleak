"""Synthetic tasks with planted leaks for testing the audit (the planted-leak validation suite).

Base task: sequences from an order-3 Markov chain fitted to a fixed GRCh38 background, length 300, labels balanced and
independent of sequence. ``make_task(plant, dose, seed)`` returns (DataFrame with seq, label, split[, dist]), info)
where info holds the planted items and, for shortcut and position plants, the oracle score of the planted rule."""
import os
import random

import numpy as np
import pandas as pd

from .seqs import DATA

PLANTS = ['none', 'exact', 'near', 'shifted', 'shuffle', 'segment', 'length', 'npad', 'motif', 'position',
          'control_gc', 'control_motif', 'shuffle_within', 'allN', 'site', 'library', 'control_library',
          'genome', 'control_genome', 'anchored', 'control_anchored', 'shuffle_test']
# rules v1.1 plants: shuffle_within (d of items are positive / dinucleotide-shuffled negative pairs inside one split),
# allN (d of items are windows of N with label 0), site (every positive has GT at 150, a share d of negatives lack it),
# library (three libraries with 40-bp constant segments; P(positive | library) = 0.5 + d/2, 0.5 - d/2, 0.5);
# control_library (one constant segment in every item, labels independent of it).
# rules v1.2 plants: genome (items are distinct windows of a synthetic reference genome written to a FASTA file; a
# share d of negatives are dinucleotide shuffles of other windows, so they are absent from the reference),
# control_genome (every item a window of the reference), anchored (variable-length items, 150-400 bases; a share d of
# positives carry TATAAA 30 bases before the end), control_anchored (90% of positives carry TATAAA at a random
# position), shuffle_test (training items without construction; a share d of test items are positive /
# dinucleotide-shuffled negative pairs).
ALPH = np.array(list('ACGT'))


def _chain():
    """Order-3 transitions (from background 4-mer counts) and the initial 3-mer distribution."""
    z = np.load(os.path.join(DATA, 'markov_bg_grch38.npz'))
    c = z['k3'].reshape(64, 4).astype(float) + 1.0
    p3 = z['k2'].astype(float) + 1.0
    return c / c.sum(1, keepdims=True), p3 / p3.sum()


def markov_seqs(n, L, rng, gc_shift=0.0):
    """n sequences of length L from the background order-3 chain (optionally GC-shifted)."""
    P, p3 = _chain()
    if gc_shift:
        P = P * np.array([1 - gc_shift, 1 + gc_shift, 1 + gc_shift, 1 - gc_shift])[None, :]
        P = P / P.sum(1, keepdims=True)
    X = np.zeros((n, L), np.int64)
    first = rng.choice(64, size=n, p=p3)
    X[:, 0] = first // 16
    X[:, 1] = (first // 4) % 4
    X[:, 2] = first % 4
    cum = np.cumsum(P, 1)
    for j in range(3, L):
        ctx = X[:, j - 3] * 16 + X[:, j - 2] * 4 + X[:, j - 1]
        u = rng.random(n)
        X[:, j] = (u[:, None] > cum[ctx]).sum(1)
    X = np.minimum(X, 3)
    return [''.join(r) for r in ALPH[X]]


def dinuc_shuffle(s, rng):
    """Dinucleotide shuffle (Altschul-Erickson: Eulerian walk with a random last-edge arborescence); keeps the
    first and last base and every dinucleotide count."""
    r = random.Random(int(rng.integers(1 << 31)))
    edges = {}
    for a, b in zip(s[:-1], s[1:]):
        edges.setdefault(a, []).append(b)
    last = s[-1]
    while True:
        lastedge = {v: r.choice(o) for v, o in edges.items() if v != last}
        ok = True
        for v in lastedge:
            seen = set()
            while v != last:
                if v in seen or v not in lastedge:
                    ok = False
                    break
                seen.add(v)
                v = lastedge[v]
            if not ok:
                break
        if ok:
            break
    lists = {}
    for v, o in edges.items():
        o = list(o)
        if v in lastedge:
            o.remove(lastedge[v])
            r.shuffle(o)
            o.append(lastedge[v])
        else:
            r.shuffle(o)
        lists[v] = o
    ptr = dict.fromkeys(lists, 0)
    out = [s[0]]
    v = s[0]
    for _ in range(len(s) - 1):
        nxt = lists[v][ptr[v]]
        ptr[v] += 1
        out.append(nxt)
        v = nxt
    return ''.join(out)


def segment_substitute(s, rng, n_seg=20, keep=8):
    L = len(s)
    sl = L // n_seg
    kept = set(rng.choice(n_seg, keep, replace=False).tolist())
    new = markov_seqs(1, L, rng)[0]
    return ''.join(s[i * sl:(i + 1) * sl] if i in kept else new[i * sl:(i + 1) * sl] for i in range(n_seg)) + s[n_seg * sl:]


def mutate(s, rate, rng):
    a = np.array(list(s))
    m = rng.random(len(a)) < rate
    a[m] = [rng.choice([c for c in 'ACGT' if c != x]) for x in a[m]]
    return ''.join(a)


def revcomp(s):
    return s.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def _mcc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    tp = np.sum((y == 1) & (p == 1))
    tn = np.sum((y == 0) & (p == 0))
    fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0))
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def make_task(plant='none', dose=0.1, seed=0, n_train=4000, n_test=1000, L=300, workdir=None):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    split = np.array(['train'] * n_train + ['test'] * n_test)
    y = np.zeros(n, np.int64)
    y[rng.permutation(n)[:n // 2]] = 1
    gc = 0.15 if plant == 'control_gc' else 0.0
    seqs = markov_seqs(n, L, rng)
    if gc:
        pos = np.flatnonzero(y == 1)
        shifted = markov_seqs(len(pos), L, rng, gc_shift=gc)
        for i, s in zip(pos, shifted):
            seqs[i] = s
    info = dict(plant=plant, dose=dose, seed=seed)
    te = np.flatnonzero(split == 'test')
    tr = np.flatnonzero(split == 'train')
    k = int(round(dose * n_test))
    dist = None
    if plant in ('exact', 'near', 'shifted'):
        sel = rng.choice(te, k, replace=False)
        src = rng.choice(tr, k, replace=False)
        for i, j in zip(sel, src):
            y[i] = y[j]
            if plant == 'exact':
                seqs[i] = seqs[j] if rng.random() < 0.5 else revcomp(seqs[j])
            elif plant == 'near':
                seqs[i] = mutate(seqs[j], 0.02, rng)
            else:
                seqs[i] = seqs[j][L // 2:] + markov_seqs(1, L - L // 2, rng)[0]
        info['planted'] = sel.tolist()
    elif plant in ('shuffle', 'segment'):
        sel = rng.choice(te, k, replace=False)
        src = rng.choice(tr[y[tr] == 1], k, replace=False)
        for i, j in zip(sel, src):
            y[i] = 0
            seqs[i] = dinuc_shuffle(seqs[j], rng) if plant == 'shuffle' else segment_substitute(seqs[j], rng)
        info['planted'] = sel.tolist()
    elif plant in ('length', 'npad', 'motif'):
        pos = np.flatnonzero(y == 1)
        sel = pos[rng.random(len(pos)) < dose]
        motif = 'TATAAA'
        for i in sel:
            if plant == 'length':
                seqs[i] = seqs[i][:L - 40]
            elif plant == 'npad':
                seqs[i] = seqs[i][:L - 30] + 'N' * 30
            else:
                seqs[i] = seqs[i][:150] + motif + seqs[i][156:]
        mark = np.zeros(n, bool)
        mark[sel] = True
        info['oracle_mcc'] = _mcc(y[te], mark[te].astype(int))
    elif plant == 'control_motif':
        pos = np.flatnonzero(y == 1)
        for i in pos[rng.random(len(pos)) < 0.9]:
            p = int(rng.integers(0, L - 6))
            seqs[i] = seqs[i][:p] + 'TATAAA' + seqs[i][p + 6:]
    if plant == 'shuffle_within':
        k2 = int(round(dose * n / 2))
        for sp in ('train', 'test'):
            idx = np.flatnonzero(split == sp)
            m = int(round(k2 * len(idx) / n))
            pick = rng.choice(idx, 2 * m, replace=False)
            for a, b in zip(pick[:m], pick[m:]):
                y[a], y[b] = 1, 0
                seqs[b] = dinuc_shuffle(seqs[a], rng)
    elif plant == 'allN':
        for sp in ('train', 'test'):
            idx = np.flatnonzero(split == sp)
            for i in rng.choice(idx, int(round(dose * len(idx))), replace=False):
                seqs[i] = 'N' * L
                y[i] = 0
    elif plant == 'site':
        for i in range(n):
            s_ = list(seqs[i])
            if y[i] == 1 or rng.random() >= dose:
                s_[150:152] = 'GT'
            elif ''.join(s_[150:152]) == 'GT':
                s_[150:152] = 'CA'
            seqs[i] = ''.join(s_)
    elif plant in ('library', 'control_library'):
        n_lib = 3 if plant == 'library' else 1
        bb = markov_seqs(n_lib, 40, rng)
        lib = rng.integers(0, n_lib, n)
        p_pos = np.array([0.5 + dose / 2, 0.5 - dose / 2, 0.5])[:n_lib] if plant == 'library' else np.array([0.5])
        y = (rng.random(n) < p_pos[lib]).astype(np.int64)
        seqs = [bb[l] + s_[40:] for l, s_ in zip(lib, seqs)]
        if plant == 'library':
            # oracle: majority label of each library on train, scored on test
            maj = np.array([int(np.mean(y[tr][lib[tr] == g]) >= 0.5) if np.any(lib[tr] == g) else 0 for g in range(n_lib)])
            info['oracle_mcc'] = _mcc(y[te], maj[lib[te]])
    if plant in ('genome', 'control_genome'):
        neg = np.flatnonzero(y == 0)
        syn = neg[rng.random(len(neg)) < dose] if plant == 'genome' else np.zeros(0, np.int64)
        tiles = (n + len(syn)) * 3 // 2 + 10                     # distinct, non-overlapping windows
        gen = ''.join(markov_seqs(tiles * L // 5000 + 1, 5000, rng))
        pick = rng.choice(len(gen) // L, n + len(syn), replace=False)
        win = [gen[t * L:(t + 1) * L] for t in pick]
        seqs = [w if rng.random() < 0.5 else revcomp(w) for w in win[:n]]
        for i, w in zip(syn, win[n:]):
            seqs[i] = dinuc_shuffle(w, rng)
        member = np.ones(n, bool)
        member[syn] = False
        if plant == 'genome':
            info['oracle_mcc'] = _mcc(y[te], member[te].astype(int))
        import tempfile
        path = os.path.join(workdir or tempfile.gettempdir(), f'genome_{plant}_{dose}_{seed}_{os.getpid()}.fa')
        with open(path, 'w') as f:
            f.write('>synthetic\n')
            for j in range(0, len(gen), 80):
                f.write(gen[j:j + 80] + '\n')
        info['genome'] = path
    elif plant in ('anchored', 'control_anchored'):
        lens = rng.integers(150, 401, n)
        seqs = [s_[:int(m)] for s_, m in zip(markov_seqs(n, 400, rng), lens)]
        pos = np.flatnonzero(y == 1)
        if plant == 'anchored':
            sel = pos[rng.random(len(pos)) < dose]
            for i in sel:
                m = len(seqs[i])
                seqs[i] = seqs[i][:m - 30] + 'TATAAA' + seqs[i][m - 24:]
            mark = np.zeros(n, bool)
            mark[sel] = True
            info['oracle_mcc'] = _mcc(y[te], mark[te].astype(int))
        else:
            for i in pos[rng.random(len(pos)) < 0.9]:
                p_ = int(rng.integers(0, len(seqs[i]) - 6))
                seqs[i] = seqs[i][:p_] + 'TATAAA' + seqs[i][p_ + 6:]
    elif plant == 'shuffle_test':
        m = int(round(dose * n_test / 2))
        pick = rng.choice(te, 2 * m, replace=False)
        for a, b in zip(pick[:m], pick[m:]):
            y[a], y[b] = 1, 0
            seqs[b] = dinuc_shuffle(seqs[a], rng)
    if plant == 'position':
        dist = rng.normal(size=n) + 2 * dose * y
        from scipy.stats import norm
        info['oracle_auroc'] = float(norm.cdf(2 * dose / np.sqrt(2)))
    df = pd.DataFrame(dict(id=[f's{i}' for i in range(n)], seq=seqs, label=y, split=split))
    if dist is not None:
        df['dist'] = dist
    return df, info
