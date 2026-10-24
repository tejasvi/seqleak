"""Constructed-negative twins between different sequences of a task.

Composition twins: same length, dinucleotide counts and first/last base (what a dinucleotide shuffle keeps) and
Hamming identity < ``max_identity`` (default 0.6; shuffles of each other are about as similar as unrelated sequences;
near-identical natural sequences that happen to share a signature are excluded).
Positional twins: equal-length sequences sharing at least ``min_shared`` (3) 12-mers at the same offsets (every
offset; 12-mer groups of 2-``max_group`` sequences) and Hamming identity >= ``min_identity`` (default
0.4): what segment substitution leaves (DeePromoter-style negatives keep 8 of 20 segments in place, identity about
0.55; unrelated sequences of the same length are about 0.25-0.3 identical).
Both work on unique (upper-case) sequences and return index pairs (a < b) with their identity."""
import numpy as np

from .seqs import chunk_ranges, concat, kmer_counts


def hamming_identity(codes, off, a, b, chunk=8_000_000):
    """Identity of equal-length sequence pairs (a[i], b[i]) (share of positions with the same base)."""
    out = np.zeros(len(a))
    if len(a) == 0:
        return out
    lens = (off[a + 1] - off[a])
    for L in np.unique(lens):
        sel = np.flatnonzero(lens == L)
        step = max(1, int(chunk // max(L, 1)))
        for s in range(0, len(sel), step):
            ii = sel[s:s + step]
            ar = np.arange(L)
            A = codes[off[a[ii]][:, None] + ar]
            B = codes[off[b[ii]][:, None] + ar]
            out[ii] = (A == B).mean(1) if L else 0.0
    return out


def _co(seqs):
    c, _, o = concat(seqs)
    return c, o


def composition_twins(seqs, max_identity=0.6, max_group=50):
    codes, _, off = concat(seqs)
    n = len(seqs)
    if n < 2:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    lens = np.diff(off)
    c2 = np.concatenate([kmer_counts(*_co(seqs[a:b]), 2) for a, b in chunk_ranges(seqs)])
    first = np.where(lens > 0, codes[np.minimum(off[:-1], max(len(codes) - 1, 0))], 9) if len(codes) else np.zeros(n)
    last = np.where(lens > 0, codes[np.maximum(off[1:] - 1, 0)], 9) if len(codes) else np.zeros(n)
    sig = np.column_stack([lens, c2, first, last]).astype(np.int64)
    _, inv, cnt = np.unique(sig, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    ok = (cnt[inv] >= 2) & (cnt[inv] <= max_group) & (lens >= 20)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    o = idx[np.argsort(inv[idx], kind='stable')]
    g = inv[o]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(o)]
    pa, pb = [], []
    for s, e in zip(starts, ends):
        m = o[s:e]
        ia, ib = np.triu_indices(len(m), 1)
        pa.append(m[ia])
        pb.append(m[ib])
    a = np.concatenate(pa)
    b = np.concatenate(pb)
    ident = hamming_identity(codes, off, a, b)
    keep = ident < max_identity
    a, b, ident = a[keep], b[keep], ident[keep]
    sw = a > b
    a[sw], b[sw] = b[sw], a[sw]
    return np.column_stack([a, b]).astype(np.int64), ident


def positional_twins(seqs, min_identity=0.4, min_shared=3, stride=1, K=12, max_group=20, max_len=2000,
                     block_keys=4_000_000):
    codes, _, off = concat(seqs)
    lens = np.diff(off)
    pairs, idents = [], []
    for L in np.unique(lens):
        if L < K + 1 or L > max_len:
            continue
        members = np.flatnonzero(lens == L)
        nm = len(members)
        if nm < 2:
            continue
        starts_all = np.arange(0, L - K + 1, stride)
        per_block = max(1, int(block_keys // nm))
        pkeys = []
        for b0 in range(0, len(starts_all), per_block):
            starts = starts_all[b0:b0 + per_block]
            code = np.zeros((nm, len(starts)), np.int64)
            bad = np.zeros((nm, len(starts)), bool)
            for j in range(K):
                c = codes[off[members][:, None] + (starts + j)[None, :]]
                bad |= c > 3
                code = code * 4 + np.minimum(c, 3)
            key = starts[None, :].astype(np.int64) * (4 ** K) + code
            mem = np.broadcast_to(np.arange(nm)[:, None], key.shape)
            key = key[~bad]
            mem = mem[~bad]
            o = np.argsort(key, kind='stable')
            key, mem = key[o], mem[o]
            st = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
            en = np.r_[st[1:], len(key)]
            sz = en - st
            sel = (sz >= 2) & (sz <= max_group)
            for size in np.unique(sz[sel]):
                gs = st[sel & (sz == size)]
                ia, ib = np.triu_indices(size, 1)
                M = mem[gs[:, None] + np.arange(size)]
                a = M[:, ia].ravel()
                b = M[:, ib].ravel()
                pkeys.append(np.minimum(a, b) * nm + np.maximum(a, b))
        if not pkeys:
            continue
        u, c = np.unique(np.concatenate(pkeys), return_counts=True)
        u = u[c >= min_shared]
        if len(u) == 0:
            continue
        a = members[u // nm]
        b = members[u % nm]
        ident = hamming_identity(codes, off, a, b)
        keep = (ident >= min_identity) & (ident < 1.0)
        pairs.append(np.column_stack([a[keep], b[keep]]))
        idents.append(ident[keep])
    if not pairs:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    return np.concatenate(pairs).astype(np.int64), np.concatenate(idents)
