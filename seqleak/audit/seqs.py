"""Sequence encoding, canonical forms and k-mer arrays, vectorised over the concatenation of many sequences."""
import hashlib
import os

import numpy as np

LUT = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate(b'ACGT'):
    LUT[_c] = _i
    LUT[_c + 32] = _i  # lowercase
COMP = str.maketrans('ACGTNacgtnRYKMBDHVrykmbdhv', 'TGCANtgcanYRMKVHDByrmkvhdb')
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def revcomp(s):
    return s.translate(COMP)[::-1]


def canonical(s):
    """Upper-case sequence or its reverse complement, whichever sorts first."""
    s = s.upper()
    r = revcomp(s)
    return s if s <= r else r


def concat(seqs):
    """Concatenate sequences; return codes (0-3 ACGT, 4 other), raw bytes and offsets (n + 1)."""
    bs = [s.encode('ascii', 'replace') for s in seqs]
    lens = np.fromiter((len(b) for b in bs), dtype=np.int64, count=len(bs))
    off = np.zeros(len(bs) + 1, dtype=np.int64)
    np.cumsum(lens, out=off[1:])
    raw = np.frombuffer(b''.join(bs), dtype=np.uint8) if bs else np.zeros(0, np.uint8)
    return LUT[raw], raw, off


def chunk_ranges(seqs, max_bases=2_000_000):
    """(start, stop) index ranges of consecutive sequences with about max_bases bases each."""
    lens = np.fromiter((len(x) for x in seqs), np.int64, count=len(seqs))
    out = []
    a = 0
    n = len(seqs)
    while a < n:
        b = a
        tot = 0
        while b < n and (b == a or tot + lens[b] <= max_bases):
            tot += lens[b]
            b += 1
        out.append((a, b))
        a = b
    return out


def seq_ids(off):
    return np.repeat(np.arange(len(off) - 1, dtype=np.int64), np.diff(off))


def kmer_counts(codes, off, k):
    """Counts of k-mers (ACGT only) per sequence: array (n, 4**k), int32."""
    n = len(off) - 1
    L = len(codes)
    if L < k:
        return np.zeros((n, 4 ** k), dtype=np.int32)
    m = L - k + 1
    valid = np.ones(m, dtype=bool)
    idx = np.zeros(m, dtype=np.int64)
    for j in range(k):
        c = codes[j:m + j]
        valid &= c < 4
        idx = idx * 4 + np.minimum(c, 3)
    sid = seq_ids(off)
    valid &= sid[:m] == sid[k - 1:]
    flat = sid[:m][valid] * (4 ** k) + idx[valid]
    return np.bincount(flat, minlength=n * 4 ** k).reshape(n, 4 ** k).astype(np.int32)


def kmer_positions(codes, off, k):
    """Every k-mer start of the concatenation: forward code, reverse-complement code (int64), validity (only ACGT and
    within one sequence), sequence id and position within the sequence."""
    L = len(codes)
    m = L - k + 1
    if m <= 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, bool), z, z
    fwd = np.zeros(m, np.int64)
    rev = np.zeros(m, np.int64)
    valid = np.ones(m, bool)
    for j in range(k):
        c = codes[j:m + j]
        valid &= c < 4
        cc = np.minimum(c, 3).astype(np.int64)
        fwd <<= 2
        fwd |= cc
        rev |= (3 - cc) << (2 * j)
    sid = seq_ids(off)
    valid &= sid[:m] == sid[k - 1:]
    s = sid[:m]
    pos = np.arange(m, dtype=np.int64) - off[s]
    return fwd, rev, valid, s, pos


def mix64(x):
    """splitmix64 finaliser (uint64 in, uint64 out): a fixed, seedless hash used for k-mer sampling."""
    x = np.asarray(x).astype(np.uint64)
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return x


def hash64(strings):
    """Stable 64-bit digests of strings (blake2b), as int64."""
    out = np.empty(len(strings), np.int64)
    for i, s in enumerate(strings):
        out[i] = int.from_bytes(hashlib.blake2b(s.encode('ascii', 'replace'), digest_size=8).digest(), 'little',
                                signed=True)
    return out


def runs_max(mask, off):
    """Longest run of True per sequence (mask over the concatenation)."""
    n = len(off) - 1
    out = np.zeros(n, np.int64)
    if len(mask) == 0 or not mask.any():
        return out
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    # split runs at sequence boundaries
    b = off[1:-1]
    cuts = b[(b > 0) & (b < len(mask))]
    cuts = cuts[mask[cuts] & mask[cuts - 1]]
    if len(cuts):
        starts = np.sort(np.concatenate([starts, cuts]))
        ends = np.sort(np.concatenate([ends, cuts]))
    sid = np.searchsorted(off, starts, side='right') - 1
    np.maximum.at(out, sid, ends - starts)
    return out


class Markov:
    """Order-k Markov chain on ACGT with add-one smoothing; log2 probabilities indexed by (k+1)-mer code."""

    def __init__(self, k):
        self.k = k
        self.logp = None

    def fit_counts(self, cnt):
        c = np.asarray(cnt, float).reshape(-1, 4) + 1.0
        self.logp = np.log2(c / c.sum(1, keepdims=True)).reshape(-1)
        return self

    def score(self, codes, off, mean=True):
        """Mean (or total) log2 P(x_i | previous k) over valid positions, per sequence."""
        k = self.k + 1
        n = len(off) - 1
        L = len(codes)
        if L < k:
            return np.zeros(n)
        m = L - k + 1
        valid = np.ones(m, bool)
        idx = np.zeros(m, np.int64)
        for j in range(k):
            c = codes[j:m + j]
            valid &= c < 4
            idx = idx * 4 + np.minimum(c, 3)
        sid = seq_ids(off)
        valid &= sid[:m] == sid[k - 1:]
        lp = np.where(valid, self.logp[idx], 0.0)
        s = np.bincount(sid[:m], weights=lp, minlength=n)
        if not mean:
            return s
        cnt = np.bincount(sid[:m], weights=valid, minlength=n)
        return s / np.maximum(cnt, 1)


_BG = {}


def background():
    """Order-5 and order-1 Markov models of a fixed GRCh38 background (300 random 100-kb chunks of chr1-22)."""
    if not _BG:
        z = np.load(os.path.join(DATA, 'markov_bg_grch38.npz'))
        _BG['m5'] = Markov(5).fit_counts(z['k5'])
        _BG['m1'] = Markov(1).fit_counts(z['k1'])
    return _BG['m5'], _BG['m1']
