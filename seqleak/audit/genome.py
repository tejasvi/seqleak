"""Membership of items in reference genomes (construction.genome_membership).

Shuffled or generated negatives are not part of any genome; sequences drawn from the organism's genome are. Built-in
method: canonical k-mer containment. Each item contributes up to ``per_item`` evenly spaced canonical k-mers, the
reference FASTA files (plain or gzip) are streamed once, and an item's membership is the share of its sampled k-mers
found in a reference. Optional method: full-length alignment with minimap2 (one alignment covering >= 95% of the item
at >= 95% identity)."""
import gzip
import os
import shutil
import subprocess
import tempfile

import numpy as np

from .seqs import LUT, chunk_ranges, concat


def resolve(refs, registry=None):
    """Reference names or paths -> list of existing FASTA paths. Names are looked up in a TSV registry
    (name<TAB>path; ``registry`` or the environment variable SEQLEAK_GENOME_REGISTRY)."""
    reg = {}
    reg_path = registry or (os.environ.get('SEQLEAK_GENOME_REGISTRY') or os.environ.get('POSBENCH_GENOME_REGISTRY'))
    if reg_path:
        for line in open(reg_path):
            if line.strip() and not line.startswith('#'):
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 2:
                    reg[parts[0]] = parts[1]
    out = []
    for r in refs:
        p = reg.get(r, r)
        if not os.path.isfile(p):
            raise FileNotFoundError(f'reference not found: {r}')
        out.append(p)
    return out


def _open(path):
    with open(path, 'rb') as f:
        gz = f.read(2) == b'\x1f\x8b'
    return gzip.open(path, 'rb') if gz else open(path, 'rb')


def fasta_blocks(path, k, block=1 << 24, read_size=1 << 24):
    """Sequence blocks (bytes) of every FASTA record; consecutive blocks of one record overlap by k - 1 bases."""
    buf = bytearray()

    def take(seg):
        nonlocal buf
        buf += seg.translate(None, b'\r\n')
        while len(buf) >= block + k - 1:
            yield bytes(buf[:block + k - 1])
            del buf[:block]

    def end_record():
        nonlocal buf
        if len(buf) >= k:
            yield bytes(buf)
        buf = bytearray()

    pending = b''
    with _open(path) as f:
        while True:
            chunk = f.read(read_size)
            last = not chunk
            data = pending + chunk
            if not last:
                cut = data.rfind(b'\n')
                if cut < 0:
                    if data.startswith(b'>'):      # an unfinished header line
                        pending = data
                    else:                          # part of a long sequence line
                        yield from take(data)
                        pending = b''
                    continue
                pending, data = data[cut + 1:], data[:cut + 1]
            pos = 0
            while True:
                h = data.find(b'>', pos)
                if h < 0:
                    yield from take(data[pos:])
                    break
                yield from take(data[pos:h])
                yield from end_record()
                e = data.find(b'\n', h)
                pos = len(data) if e < 0 else e + 1
            if last:
                break
    yield from end_record()


def _fwd_codes(c, k):
    """Forward codes (uint64) of every k-mer of the 2-bit array c (uint64), k <= 31, by doubling."""
    n = len(c) - k + 1
    parts = {1: c}
    w = 1
    while 2 * w <= k:
        p = parts[w]
        parts[2 * w] = (p[:len(p) - w] << np.uint64(2 * w)) | p[w:]
        w *= 2
    out, done = None, 0
    for w in sorted(parts, reverse=True):
        if done + w <= k:
            seg = parts[w][done:done + n]
            out = seg.copy() if out is None else (out << np.uint64(2 * w)) | seg
            done += w
    return out


def canonical_codes(raw_codes, k):
    """Canonical k-mer codes (uint64) and validity (only A/C/G/T) at every start of one sequence (LUT codes)."""
    n = len(raw_codes) - k + 1
    if n <= 0:
        return np.zeros(0, np.uint64), np.zeros(0, bool)
    bad = np.concatenate([[0], np.cumsum(raw_codes > 3, dtype=np.int64)])
    ok = (bad[k:] - bad[:n]) == 0
    c = np.minimum(raw_codes, 3).astype(np.uint64)
    f = _fwd_codes(c, k)
    r = _fwd_codes((np.uint64(3) - c)[::-1], k)[::-1]
    return np.minimum(f, r), ok


def sample_item_kmers(seqs, k, per_item):
    """Up to ``per_item`` evenly spaced valid canonical k-mers per item: (codes uint64, item index int64)."""
    codes_all, items_all = [], []
    for a, b in chunk_ranges(seqs, 4_000_000):
        codes, _, off = concat(seqs[a:b])
        lens = np.diff(off)
        m = np.clip(lens - k + 1, 0, per_item)
        rows = np.repeat(np.arange(b - a), m)
        if len(rows) == 0:
            continue
        # evenly spaced start positions within each item
        j = np.arange(len(rows)) - np.repeat(np.cumsum(m) - m, m)
        span = np.repeat(np.maximum(lens - k, 0), m)
        denom = np.repeat(np.maximum(m - 1, 1), m)
        start = off[rows] + (j * span) // denom
        win = codes[start[:, None] + np.arange(k)[None, :]]
        good = (win < 4).all(1)
        w = np.minimum(win, 3).astype(np.uint64)
        f = np.zeros(len(rows), np.uint64)
        r = np.zeros(len(rows), np.uint64)
        for t in range(k):
            f = (f << np.uint64(2)) | w[:, t]
            r = r | ((np.uint64(3) - w[:, t]) << np.uint64(2 * t))
        codes_all.append(np.minimum(f, r)[good])
        items_all.append(rows[good] + a)
    if not codes_all:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64)
    return np.concatenate(codes_all), np.concatenate(items_all)


def reference_hits(paths, query, k, block=1 << 24, prefix_bits=28):
    """Which of the sorted unique canonical codes ``query`` occur in any reference (bool array); bases streamed."""
    found = np.zeros(len(query), bool)
    if len(query) == 0:
        return found, 0
    shift = np.uint64(max(2 * k - prefix_bits, 0))
    bm = np.zeros(1 << max(prefix_bits - 3, 0), np.uint8)
    pref = np.unique((query >> shift).astype(np.int64))
    np.bitwise_or.at(bm, pref >> 3, (1 << (pref & 7)).astype(np.uint8))
    n_bases = 0
    for path in paths:
        for blk in fasta_blocks(path, k, block):
            raw = LUT[np.frombuffer(blk, np.uint8)]
            n_bases += len(raw)
            can, ok = canonical_codes(raw, k)
            can = can[ok]
            p = (can >> shift).astype(np.int64)
            cand = can[((bm[p >> 3] >> (p & 7).astype(np.uint8)) & 1).astype(bool)]
            if len(cand) == 0:
                continue
            idx = np.searchsorted(query, cand)
            idx[idx >= len(query)] = len(query) - 1
            hit = query[idx] == cand
            found[idx[hit]] = True
    return found, n_bases


def kmer_membership(seqs, paths, k=21, per_item=32):
    """Membership per item = share of its sampled canonical k-mers found in the references (NaN when an item has no
    valid k-mer). Returns (membership array, info dict)."""
    codes, items = sample_item_kmers(seqs, k, per_item)
    uq, inv = np.unique(codes, return_inverse=True)
    found, n_bases = reference_hits(paths, uq, k)
    n = len(seqs)
    tot = np.bincount(items, minlength=n).astype(np.float64)
    hit = np.bincount(items, weights=found[inv].astype(np.float64), minlength=n)
    memb = np.full(n, np.nan)
    has = tot > 0
    memb[has] = hit[has] / tot[has]
    return memb, dict(method='kmer', k=k, per_item=per_item, query_kmers=int(len(uq)), reference_bases=int(n_bases),
                      unplaced_items=int((~has).sum()))


def minimap2_membership(seqs, paths, minimap2=None, threads=1, workdir=None, cov=0.95, ident=0.95):
    """Membership 1/0 per item: one minimap2 alignment covering >= cov of the item at >= ident identity (preset sr for
    median length <= 1 kb, asm5 above). Returns (membership array, info dict); raises when the binary is missing."""
    mm = minimap2 or shutil.which('minimap2')
    if not mm:
        raise FileNotFoundError('minimap2 binary not found')
    lens = np.fromiter((len(s) for s in seqs), np.int64, count=len(seqs))
    preset = 'sr' if np.median(lens) <= 1000 else 'asm5'
    base = workdir or os.environ.get('TMPDIR') or os.getcwd()
    os.makedirs(base, exist_ok=True)
    W = tempfile.mkdtemp(prefix='seqleak_genome_', dir=base)
    memb = np.zeros(len(seqs))
    try:
        q = os.path.join(W, 'q.fa')
        with open(q, 'w') as f:
            for i, s in enumerate(seqs):
                f.write(f'>{i}\n{s}\n')
        for ref in paths:
            out = os.path.join(W, 'hits.paf')
            with open(out, 'w') as fo:
                subprocess.run([mm, '-t', str(threads), '-x', preset, '-c', '--secondary=no', ref, q], check=True,
                               stdout=fo, stderr=subprocess.DEVNULL, env=dict(os.environ, TMPDIR=W))
            with open(out) as fi:
                for line in fi:
                    p = line.split('\t')
                    qlen, qs, qe, nmatch, alen = int(p[1]), int(p[2]), int(p[3]), int(p[9]), int(p[10])
                    if (qe - qs) >= cov * qlen and nmatch >= ident * max(alen, 1):
                        memb[int(p[0])] = 1.0
    finally:
        shutil.rmtree(W, ignore_errors=True)
    return memb, dict(method='minimap2', preset=preset, coverage=cov, identity=ident)
