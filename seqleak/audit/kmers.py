"""Alignment-free relatedness: canonical k-mer containment between query and reference sequences.

Sampling: FracMinHash (a k-mer is kept when mix64(code) % scale == 0, the same k-mers in every sequence). Each
sequence contributes each sampled canonical k-mer once (first position). K-mers present in more than ``max_df``
reference sequences are uninformative (repeats, low complexity, or a highly redundant set): they are excluded from both
the shared count and the query's denominator. For every query and reference pair sharing informative k-mers the
engine returns the shared count and the largest number of shared k-mers on one diagonal (same offset on the same or
opposite strand): a collinear, ungapped run, i.e. an overlapping or shifted window.
"""
from dataclasses import dataclass

import numpy as np

from .seqs import concat, kmer_positions, mix64


def auto_scale(lengths):
    med = float(np.median(lengths)) if len(lengths) else 0.0
    return max(1, int(round(med / 128.0)))


def sample_kmers(seqs, k, scale, chunk_bases=1_000_000):
    """Sampled canonical k-mers of every sequence, one entry per (sequence, k-mer) (first position).
    Returns kmer (int64), item (int64), pos (int64), ori (bool: forward strand is the canonical one), lengths."""
    parts = []
    n = len(seqs)
    lens = np.fromiter((len(s) for s in seqs), np.int64, count=n)
    start = 0
    while start < n:
        stop = start
        tot = 0
        while stop < n and (tot == 0 or tot + lens[stop] <= chunk_bases):
            tot += lens[stop]
            stop += 1
        codes, _, off = concat(seqs[start:stop])
        fwd, rev, valid, sid, pos = kmer_positions(codes, off, k)
        can = np.minimum(fwd, rev)
        keep = valid
        if scale > 1:
            keep = keep & ((mix64(can) % np.uint64(scale)) == 0)
        kk = can[keep]
        it = sid[keep] + start
        ps = pos[keep]
        ori = (fwd <= rev)[keep]
        # one entry per (item, kmer): first position
        o = np.lexsort((ps, kk, it))
        kk, it, ps, ori = kk[o], it[o], ps[o], ori[o]
        first = np.ones(len(kk), bool)
        first[1:] = (kk[1:] != kk[:-1]) | (it[1:] != it[:-1])
        parts.append((kk[first], it[first], ps[first], ori[first]))
        start = stop
    if not parts:
        z = np.zeros(0, np.int64)
        return z, z, z, np.zeros(0, bool), lens
    kk = np.concatenate([p[0] for p in parts])
    it = np.concatenate([p[1] for p in parts])
    ps = np.concatenate([p[2] for p in parts])
    ori = np.concatenate([p[3] for p in parts])
    return kk, it, ps, ori, lens


@dataclass
class Index:
    kmer: np.ndarray
    item: np.ndarray
    pos: np.ndarray
    ori: np.ndarray
    n_items: int
    n_kmers: np.ndarray
    max_len: int
    k: int
    scale: int


def build_index(seqs, k=21, scale=1):
    kk, it, ps, ori, lens = sample_kmers(seqs, k, scale)
    o = np.argsort(kk, kind='stable')
    return Index(kk[o], it[o].astype(np.int64), ps[o].astype(np.int64), ori[o], len(seqs),
                 np.bincount(it, minlength=len(seqs)).astype(np.int64), int(lens.max()) if len(lens) else 0, k, scale)


@dataclass
class Hits:
    """Sparse query x reference table (kept pairs) and per-query k-mer counts."""
    q: np.ndarray            # query index
    r: np.ndarray            # reference index
    shared: np.ndarray       # shared informative sampled k-mers
    coll: np.ndarray         # largest count on one diagonal
    shift: np.ndarray        # offset of that diagonal (query pos - reference pos; same strand), or a large value
    same_strand: np.ndarray  # whether that diagonal is on the same strand
    n_all: np.ndarray        # sampled k-mers per query
    n_inf: np.ndarray        # informative sampled k-mers per query (reference df <= max_df)

    def containment(self):
        return self.shared / np.maximum(self.n_inf[self.q], 1)

    def collinear(self):
        return self.coll / np.maximum(self.n_inf[self.q], 1)


def _expand(lo, df):
    tot = int(df.sum())
    starts = np.repeat(lo, df)
    rel = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(df) - df, df)
    return starts + rel


def query(index, qseqs, max_df=64, keep_top=8, keep_c=0.2, exclude=None, chunk_hits=20_000_000):
    """Containment of each query in the reference sequences of ``index``.

    Keeps, per query, the ``keep_top`` partners with the most shared k-mers plus every partner with containment >=
    ``keep_c``. ``exclude``: optional array (len(qseqs)) of a reference index to ignore per query (self matches when the
    query set is the reference set), -1 for none."""
    kk, it, ps, ori, qlens = sample_kmers(qseqs, index.k, index.scale)
    nq = len(qseqs)
    lo = np.searchsorted(index.kmer, kk, 'left')
    hi = np.searchsorted(index.kmer, kk, 'right')
    df = hi - lo
    inf = df <= max_df
    n_all = np.bincount(it, minlength=nq).astype(np.int64)
    n_inf = np.bincount(it[inf], minlength=nq).astype(np.int64)
    use = inf & (df > 0)
    it, ps, ori, lo, df = it[use], ps[use], ori[use], lo[use], df[use]
    Lm = int(max(index.max_len, int(qlens.max()) if nq else 0)) + 1
    W = 5 * Lm + 2
    out = []
    # chunks of whole queries with about chunk_hits hits each (entries are sorted by query)
    bounds = [0]
    if len(it):
        qh = np.bincount(it, weights=df, minlength=nq)
        cq = np.cumsum(qh)
        qcuts = [0]
        while qcuts[-1] < nq:
            base = cq[qcuts[-1] - 1] if qcuts[-1] > 0 else 0.0
            j = int(np.searchsorted(cq, base + chunk_hits, 'right'))
            qcuts.append(min(max(j, qcuts[-1] + 1), nq))
        bounds = [int(x) for x in np.searchsorted(it, qcuts, 'left')]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        if True:
            ridx = _expand(lo[a:b], df[a:b])
            rep = df[a:b]
            qi = np.repeat(it[a:b], rep)
            qp = np.repeat(ps[a:b], rep)
            qo = np.repeat(ori[a:b], rep)
            ri = index.item[ridx]
            rp = index.pos[ridx]
            ro = index.ori[ridx]
            if exclude is not None:
                ok = exclude[qi] != ri
                qi, qp, qo, ri, rp, ro = qi[ok], qp[ok], qo[ok], ri[ok], rp[ok], ro[ok]
            if len(qi) == 0:
                continue
            same = qo == ro
            d = np.where(same, qp - rp + Lm, 3 * Lm + qp + rp)
            pair = qi * index.n_items + ri
            if index.n_items and nq and float(nq) * index.n_items * W < 9.0e18:
                key = pair * W + d
                u, cnt = np.unique(key, return_counts=True)
                up = u // W
                ud = u % W
            else:
                o = np.lexsort((d, pair))
                pair_s, d_s = pair[o], d[o]
                new = np.ones(len(o), bool)
                new[1:] = (pair_s[1:] != pair_s[:-1]) | (d_s[1:] != d_s[:-1])
                st = np.flatnonzero(new)
                cnt = np.diff(np.append(st, len(o)))
                up, ud = pair_s[st], d_s[st]
            # per pair: shared = sum of diagonal counts, coll = max, with its diagonal
            newp = np.ones(len(up), bool)
            newp[1:] = up[1:] != up[:-1]
            ps_ = np.flatnonzero(newp)
            shared = np.add.reduceat(cnt, ps_)
            o2 = np.lexsort((-cnt, up))
            firsts = o2[np.flatnonzero(np.r_[True, up[o2][1:] != up[o2][:-1]])]
            coll = cnt[firsts]
            dbest = ud[firsts]
            pp = up[ps_]
            qq = pp // index.n_items
            rr = pp % index.n_items
            # keep top partners per query plus all with containment >= keep_c
            cont = shared / np.maximum(n_inf[qq], 1)
            o3 = np.lexsort((-coll, -shared, qq))
            qs = qq[o3]
            rank = np.arange(len(qs)) - np.searchsorted(qs, qs, 'left')
            keep = (rank < keep_top) | (cont[o3] >= keep_c)
            sel = o3[keep]
            same_s = dbest[sel] < 3 * Lm
            shift = np.where(same_s, dbest[sel] - Lm, dbest[sel] - 3 * Lm)
            out.append((qq[sel], rr[sel], shared[sel], coll[sel], shift, same_s))
    if out:
        cols = [np.concatenate([o[i] for o in out]) for i in range(6)]
    else:
        z = np.zeros(0, np.int64)
        cols = [z, z, z, z, z, np.zeros(0, bool)]
    return Hits(cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], n_all, n_inf)


def best_per_query(h, nq):
    """Best partner per query (most shared; ties: longest collinear run, then lowest index): arrays of length nq with
    -1 / 0 where a query has no partner. Also the largest collinear fraction over kept partners."""
    best_r = np.full(nq, -1, np.int64)
    best_shared = np.zeros(nq, np.int64)
    best_coll = np.zeros(nq, np.int64)
    max_coll = np.zeros(nq, np.int64)
    max_coll_r = np.full(nq, -1, np.int64)
    max_coll_shift = np.zeros(nq, np.int64)
    max_coll_same = np.zeros(nq, bool)
    if len(h.q):
        o = np.lexsort((h.r, -h.coll, -h.shared, h.q))
        f = o[np.flatnonzero(np.r_[True, h.q[o][1:] != h.q[o][:-1]])]
        best_r[h.q[f]] = h.r[f]
        best_shared[h.q[f]] = h.shared[f]
        best_coll[h.q[f]] = h.coll[f]
        o = np.lexsort((h.r, -h.coll, h.q))
        f = o[np.flatnonzero(np.r_[True, h.q[o][1:] != h.q[o][:-1]])]
        max_coll[h.q[f]] = h.coll[f]
        max_coll_r[h.q[f]] = h.r[f]
        max_coll_shift[h.q[f]] = h.shift[f]
        max_coll_same[h.q[f]] = h.same_strand[f]
    return dict(r=best_r, shared=best_shared, coll=best_coll, max_coll=max_coll, max_coll_r=max_coll_r,
                max_coll_shift=max_coll_shift, max_coll_same=max_coll_same)
