"""Construction-defect statistics added in rules v1.1: low-information inputs (windows of N), constructed negatives
anywhere in the data (twin pairs with opposite labels), negatives that lack the site defining the positives (an
unmatched fixed-position k-mer), and pooled-library structure (shared constant segments whose groups predict the
label)."""
import numpy as np

from .seqs import chunk_ranges, concat, kmer_positions


def acgt_fraction(seqs):
    """Share of A/C/G/T (either case; U counts as T for RNA input) per sequence (chunked: memory does not grow with
    the total length)."""
    out = np.zeros(len(seqs))
    for a, b in chunk_ranges(seqs, 2_000_000):
        codes, raw, off = concat(seqs[a:b])
        if len(codes) == 0:
            continue
        sid = np.repeat(np.arange(b - a), np.diff(off))
        good = np.bincount(sid, weights=(codes < 4) | (raw == 85) | (raw == 117), minlength=b - a)
        out[a:b] = good / np.maximum(np.diff(off), 1)
    return out


def opposite_twin_items(pairs_u, uid, yi, K):
    """Items whose unique sequence is in a twin pair with different (majority) labels, and the pairwise flip kappa
    against the agreement of two random labels (label frequencies of all items)."""
    nu = int(uid.max()) + 1 if len(uid) else 0
    if len(pairs_u) == 0 or nu == 0:
        return np.zeros(len(uid), bool), 0, None
    # majority label per unique sequence (ties: the smaller label index), without an (nu x K) table
    uk, cnt = np.unique(uid.astype(np.int64) * K + yi, return_counts=True)
    u_of, y_of = uk // K, uk % K
    o = np.lexsort((y_of, -cnt, u_of))
    first = np.ones(len(o), bool)
    first[1:] = u_of[o][1:] != u_of[o][:-1]
    lab = np.zeros(nu, np.int64)
    lab[u_of[o][first]] = y_of[o][first]
    a, b = pairs_u[:, 0], pairs_u[:, 1]
    opp = lab[a] != lab[b]
    mark = np.zeros(nu, bool)
    mark[a[opp]] = True
    mark[b[opp]] = True
    pi = np.bincount(yi, minlength=K) / max(len(yi), 1)
    p_same = float((pi * pi).sum())
    flip = float(opp.mean())
    kappa = (flip - (1 - p_same)) / p_same if p_same > 0 else 0.0
    return mark[uid], int(len(a)), float(kappa)


def site_table(X, rows, yi, cls, k):
    """Per-position frequency of every k-mer among rows of class cls and of the other classes: arrays (npos, 4**k)."""
    n, L = X.shape
    npos = L - k + 1
    code = np.zeros((len(rows), npos), np.int64)
    bad = np.zeros((len(rows), npos), bool)
    for j in range(k):
        c = X[rows, j:j + npos]
        bad |= c > 3
        code = code * 4 + np.minimum(c, 3)
    A = 4 ** k + 1
    code = np.where(bad, 4 ** k, code)
    flat = (np.arange(npos)[None, :] * A + code)
    yr = yi[rows]
    m = yr == cls
    fc = np.bincount(flat[m].ravel(), minlength=npos * A).reshape(npos, A)[:, :4 ** k] / max(m.sum(), 1)
    fo = np.bincount(flat[~m].ravel(), minlength=npos * A).reshape(npos, A)[:, :4 ** k] / max((~m).sum(), 1)
    return fc, fo


def unmatched_site(seqs, yi, K, fit_rows, eval_rows, site_min=0.95, max_rows=40000, seed=0, max_len=2000,
                   max_classes=20):
    """Best site-defining fixed-position k-mer (k = 1..3): present in >= site_min of one class on the fit rows, the
    largest drop in the other classes. Returns a dict with the class, k-mer, position, its rates on fit and evaluation
    rows, and the statistic = share of other-class evaluation items lacking it (0 when the k-mer is not present in
    >= 0.9 of the class on the evaluation rows). None when not applicable (variable or long sequences). With more than
    two classes, the max_classes most frequent classes on the fit rows are tried."""
    lens = np.fromiter((len(s) for s in seqs), np.int64, count=len(seqs))
    if len(lens) == 0 or len(np.unique(lens)) != 1 or lens[0] > max_len or lens[0] < 4:
        return None
    L = int(lens[0])
    codes, _, _ = concat(seqs)
    X = codes.reshape(len(seqs), L)
    rng = np.random.default_rng(seed)
    fr = np.asarray(fit_rows)
    if len(fr) > max_rows:
        fr = np.sort(rng.choice(fr, max_rows, replace=False))
    best = None
    if K > 2:
        classes = np.argsort(-np.bincount(yi[fr], minlength=K), kind='stable')[:max_classes].tolist()
    else:
        classes = [0, 1]
    for c in classes:
        if (yi[fr] == c).sum() < 20 or (yi[fr] != c).sum() < 20:
            continue
        for k in (1, 2, 3):
            fc, fo = site_table(X, fr, yi, c, k)
            gap = np.where(fc >= site_min, fc - fo, -1.0)
            j = np.unravel_index(int(np.argmax(gap)), gap.shape)
            if gap[j] <= 0:
                continue
            cand = (float(gap[j]), c, k, int(j[0]), int(j[1]), float(fc[j]), float(fo[j]))
            if best is None or cand[0] > best[0] + 1e-12:
                best = cand
    if best is None:
        return dict(found=False, statistic=0.0)
    gap, c, k, pos, km, fc, fo = best
    er = np.asarray(eval_rows)
    code = np.zeros(len(er), np.int64)
    bad = np.zeros(len(er), bool)
    for j in range(k):
        v = X[er, pos + j]
        bad |= v > 3
        code = code * 4 + np.minimum(v, 3)
    has = (code == km) & ~bad
    ye = yi[er]
    pc = float(has[ye == c].mean()) if (ye == c).any() else None
    po = float(has[ye != c].mean()) if (ye != c).any() else None
    kmer = ''.join('ACGT'[(km >> (2 * (k - 1 - t))) & 3] for t in range(k))
    stat = (1 - po) if (pc is not None and po is not None and pc >= 0.9) else 0.0
    return dict(found=True, cls=int(c), kmer=kmer, pos=pos, k=k, fit_rate_class=fc, fit_rate_other=fo,
                eval_rate_class=pc, eval_rate_other=po, statistic=float(stat))


def _lowcx(code, k):
    s = ''.join('ACGT'[(code >> (2 * (k - 1 - t))) & 3] for t in range(k))
    return len(set(s)) <= 2 or len({s[i:i + 2] for i in range(0, k - 1, 2)}) <= 2


def library_groups(seqs, k=21, min_df=0.05, min_kmers=5, pos_consistency=0.8, max_items=40000, seed=0,
                   max_frequent=5000, max_bases=20_000_000, max_backbones=62):
    """Constant segments shared by a large share of items (pooled-library backbones): canonical k-mers present in
    >= min_df of (a sample of) the items, not low-complexity, clustered by identical presence patterns; a backbone is
    a cluster of >= min_kmers k-mers whose representative k-mer sits at one offset in >= pos_consistency of the items
    that carry it (at most max_backbones, the most widely shared). Returns (signature per item for all items, list of
    backbones with sizes)."""
    n = len(seqs)
    rng = np.random.default_rng(seed)
    med = float(np.median([len(x) for x in seqs])) if n else 0.0
    cap = int(min(max_items, max(200, max_bases // max(med, 1))))      # sample bounded in items and in bases
    samp = np.sort(rng.choice(n, cap, replace=False)) if n > cap else np.arange(n)
    sub = [seqs[i] for i in samp]
    codes, _, off = concat(sub)
    fwd, rev, valid, sid, pos = kmer_positions(codes, off, k)
    can = np.minimum(fwd, rev)[valid]
    sid = sid[valid]
    pos = pos[valid]
    if len(can) == 0:
        return np.zeros(n, np.int64), []
    o = np.lexsort((pos, can, sid))
    can, sid, pos = can[o], sid[o], pos[o]
    first = np.ones(len(can), bool)
    first[1:] = (can[1:] != can[:-1]) | (sid[1:] != sid[:-1])
    can, sid, pos = can[first], sid[first], pos[first]
    u, inv, cnt = np.unique(can, return_inverse=True, return_counts=True)
    freq = np.flatnonzero(cnt >= min_df * len(sub))
    if len(freq) == 0:
        return np.zeros(n, np.int64), []
    if len(freq) > max_frequent:
        freq = freq[np.argsort(-cnt[freq], kind='stable')[:max_frequent]]
    freq = np.array([f for f in freq if not _lowcx(int(u[f]), k)], np.int64)
    if len(freq) == 0:
        return np.zeros(n, np.int64), []
    col = np.full(len(u), -1, np.int64)
    col[freq] = np.arange(len(freq))
    m = col[inv] >= 0
    P = np.zeros((len(sub), len(freq)), bool)
    P[sid[m], col[inv][m]] = True
    posmat = np.full((len(sub), len(freq)), -1, np.int64)
    posmat[sid[m], col[inv][m]] = pos[m]
    groups = {}
    for j in range(P.shape[1]):
        groups.setdefault(P[:, j].tobytes(), []).append(j)
    backbones = []
    for cols in groups.values():
        if len(cols) < min_kmers:
            continue
        rep = cols[0]
        carriers = P[:, rep]
        if carriers.sum() < min_df * len(sub):
            continue
        ps = posmat[carriers, rep]
        _, pc = np.unique(ps, return_counts=True)
        if pc.max() / max(len(ps), 1) < pos_consistency:
            continue
        backbones.append(dict(kmer_code=int(u[freq[rep]]), n_kmers=len(cols), share=float(carriers.mean()),
                              offset=int(np.unique(ps)[np.argmax(pc)])))
    if not backbones:
        return np.zeros(n, np.int64), []
    backbones = sorted(backbones, key=lambda b: -b['share'])[:max_backbones]     # bit signatures fit in int64
    # signature of every item: which backbones it carries (the representative k-mer on either strand)
    from .seqs import revcomp
    reps_ = []
    for b in backbones:
        kmer = ''.join('ACGT'[(b['kmer_code'] >> (2 * (k - 1 - t))) & 3] for t in range(k))
        reps_.append((kmer, revcomp(kmer)))
    sig = np.zeros(n, np.int64)
    for i, x in enumerate(seqs):
        x = x.upper()
        v = 0
        for bi, (f_, r_) in enumerate(reps_):
            if f_ in x or r_ in x:
                v |= 1 << bi
        sig[i] = v
    return sig, backbones
