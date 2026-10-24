"""Gene-position features of gene pairs and of single genes.

Coordinates table (``coords``): one row per gene with columns ``start`` and ``end`` (bp, start <= end) and
``strand`` (+1/-1, '+'/'-', or 1/0), optionally ``genome_id`` and ``contig``. Within each (genome_id, contig) the
rows must be in genome order; the gene index used by the task tables (``i``, ``j``) is the 0-based row number of
the gene within its contig. Contigs are treated as linear (a circular chromosome's first and last genes are not
neighbours).
"""
import numpy as np
import pandas as pd

PAIR_FEATURES = ('genes_apart', 'same_strand', 'gap', 'max_intervening_gap', 'strand_switches')


def strand_sign(x):
    """+1 / -1 from '+'/'-' strings or numbers (> 0 means plus)."""
    x = np.asarray(x)
    if x.dtype.kind in 'OUS':
        xs = np.char.strip(x.astype(str))
        return np.where(np.isin(xs, ['+', '1', '+1', 'plus', 'forward', 'fwd']), 1, -1).astype(np.int8)
    return np.where(x.astype(np.float64) > 0, 1, -1).astype(np.int8)


def _layout(coords):
    """Stable-sort genes by (genome_id, contig) keeping row order within contigs; offsets of every contig."""
    n = len(coords)
    gid = coords['genome_id'].to_numpy() if 'genome_id' in coords else np.zeros(n, np.int64)
    ctg = coords['contig'].to_numpy() if 'contig' in coords else np.zeros(n, np.int64)
    keys = pd.MultiIndex.from_arrays([gid, ctg])
    code, uniq = pd.factorize(keys)
    order = np.argsort(code, kind='stable')
    sc = code[order]
    offsets = np.searchsorted(sc, np.arange(len(uniq) + 1))
    return order, uniq, offsets


def pair_features(coords, i, j, genome_id=None, contig=None):
    """Position features of gene pairs (i, j) within a contig.

    Parameters
    ----------
    coords : DataFrame of genes (see module docstring).
    i, j : int arrays of gene indices (0-based within the contig).
    genome_id, contig : arrays giving each pair's genome and contig (required when ``coords`` has several).

    Returns a DataFrame aligned with the pairs:

    * ``genes_apart``: |j - i|
    * ``same_strand``: strand[i] == strand[j]
    * ``gap``: start[b] - end[a] with a = min(i, j), b = max(i, j) (bp between the two genes; negative = overlap)
    * ``max_intervening_gap``: max over consecutive genes t = a..b-1 of max(start[t+1] - end[t], 0)
    * ``strand_switches``: number of strand changes between consecutive genes from a to b
    """
    i = np.asarray(i, np.int64)
    j = np.asarray(j, np.int64)
    m = len(i)
    order, uniq, offsets = _layout(coords)
    st = coords['start'].to_numpy(np.int64)[order]
    en = coords['end'].to_numpy(np.int64)[order]
    sd = strand_sign(coords['strand'].to_numpy())[order]
    if len(uniq) == 1 and genome_id is None and contig is None:
        cidx = np.zeros(m, np.int64)
    else:
        gid = np.zeros(m, np.int64) if genome_id is None else np.asarray(genome_id)
        ctg = np.zeros(m, np.int64) if contig is None else np.asarray(contig)
        cidx = uniq.get_indexer(pd.MultiIndex.from_arrays([gid, ctg]))
        if (cidx < 0).any():
            raise KeyError(f'{int((cidx < 0).sum())} pairs refer to (genome_id, contig) not in coords')
    a = np.minimum(i, j)
    b = np.maximum(i, j)
    n_in = offsets[cidx + 1] - offsets[cidx]
    if (a < 0).any() or (b >= n_in).any():
        raise IndexError('gene index outside its contig')
    ga = offsets[cidx] + a
    gb = offsets[cidx] + b
    sw = np.concatenate([[0], np.cumsum(sd[1:] != sd[:-1])])
    gaps = np.maximum(st[1:] - en[:-1], 0) if len(st) > 1 else np.zeros(0, np.int64)
    maxgap = _range_max(gaps, ga, gb - ga)
    return pd.DataFrame({
        'genes_apart': (b - a).astype(np.int32),
        'same_strand': sd[ga] == sd[gb],
        'gap': (st[gb] - en[ga]).astype(np.int64),
        'max_intervening_gap': maxgap.astype(np.int64),
        'strand_switches': (sw[gb] - sw[ga]).astype(np.int32),
    })


def _range_max(arr, start, length):
    """max(arr[start : start + length]) for every query (0 where length == 0); sparse table built level by level,
    O(n) memory."""
    out = np.zeros(len(start), dtype=arr.dtype if len(arr) else np.int64)
    q = length > 0
    if not q.any():
        return out
    lev = np.floor(np.log2(np.maximum(length, 1))).astype(np.int64)
    top = int(lev[q].max())
    T = arr.copy()
    w = 1
    for L in range(top + 1):
        sel = np.flatnonzero(q & (lev == L))
        if len(sel):
            out[sel] = np.maximum(T[start[sel]], T[start[sel] + length[sel] - w])
        if L < top:
            T = np.maximum(T[:-w], T[w:])
            w *= 2
    return out


def adjacent_pairs(coords):
    """All pairs of neighbouring genes (i, i + 1) of every contig, with their position features."""
    order, uniq, offsets = _layout(coords)
    rows = []
    for c in range(len(uniq)):
        n = offsets[c + 1] - offsets[c]
        if n < 2:
            continue
        g, k = uniq[c]
        i = np.arange(n - 1)
        rows.append(pd.DataFrame(dict(genome_id=g, contig=k, i=i, j=i + 1)))
    P = pd.concat(rows, ignore_index=True)
    F = pair_features(coords, P.i.values, P.j.values, P.genome_id.values, P.contig.values)
    return pd.concat([P, F], axis=1)


def rule_score(same_strand, gap, scale=500.0, cutoff=500.0):
    """The parameter-free operon rule: 1[same strand] * exp(-gap / scale), 0 for gaps above ``cutoff``.

    Overlapping genes (negative gap) score above 1 (more overlap, higher score, as in the v1 analysis); overlaps
    longer than ``cutoff`` are treated as an overlap of ``cutoff``, so the score is at most exp(cutoff / scale) and
    annotation artefacts (gene records spanning a contig) cannot overflow."""
    same_strand = np.asarray(same_strand, bool)
    gap = np.asarray(gap, np.float64)
    keep = same_strand & (gap <= cutoff)
    out = np.zeros(gap.shape)
    out[keep] = np.exp(-np.maximum(gap[keep], -cutoff) / scale)
    return out


def gene_features(coords, origin=None, terminus=None, chrom_length=None, chromosome=None, window=10_000,
                  rule_scale=500.0, rule_cutoff=500.0):
    """Per-gene layout features, aligned with the rows of ``coords``.

    Parameters
    ----------
    coords : DataFrame of genes (see module docstring).
    origin : replication origin (bp on the chromosome contig): a number for a single genome or a dict
        genome_id -> bp. Without it, ``dist_origin`` and ``replication_strand`` are NaN.
    terminus : same form; default origin + chrom_length / 2.
    chrom_length : same form; default the largest gene end on the chromosome contig.
    chromosome : same form, the contig holding the origin; default the contig with the most genes.
    window : bp window for the local gene density.

    Returns a DataFrame with ``strand`` (+1/-1), ``length`` (end - start + 1), ``gap_up`` / ``gap_down`` (bp to the
    previous / next gene of the contig, signed; NaN at contig ends), ``same_strand_up`` / ``same_strand_down``
    (1/0, NaN at ends), ``rule_up`` / ``rule_down`` (operon rule score with that neighbour, 0 at ends),
    ``operon_rule`` (max of the two), ``directon_size`` (genes in the run of consecutive same-strand genes),
    ``directon_pos`` (0 single gene, 1 first in transcription order, 2 internal, 3 last), ``density`` (genes per kb
    whose midpoint is within window/2 of the gene's midpoint), ``dist_origin`` (0 at the origin, 1 at the terminus)
    and ``replication_strand`` (1 leading = transcribed in the direction of fork movement, 0 lagging).
    """
    n = len(coords)
    order, uniq, offsets = _layout(coords)
    st = coords['start'].to_numpy(np.int64)[order]
    en = coords['end'].to_numpy(np.int64)[order]
    sd = strand_sign(coords['strand'].to_numpy())[order]
    contig_of = np.repeat(np.arange(len(uniq)), np.diff(offsets))
    first = np.zeros(n, bool)
    last = np.zeros(n, bool)
    first[offsets[:-1][np.diff(offsets) > 0]] = True
    last[offsets[1:][np.diff(offsets) > 0] - 1] = True
    gap_prev = np.full(n, np.nan)
    gap_prev[1:] = st[1:] - en[:-1]
    gap_prev[first] = np.nan
    gap_next = np.full(n, np.nan)
    gap_next[:-1] = st[1:] - en[:-1]
    gap_next[last] = np.nan
    same_prev = np.full(n, np.nan)
    same_prev[1:] = (sd[1:] == sd[:-1]).astype(float)
    same_prev[first] = np.nan
    same_next = np.full(n, np.nan)
    same_next[:-1] = (sd[1:] == sd[:-1]).astype(float)
    same_next[last] = np.nan
    r_prev = np.where(np.isfinite(gap_prev), rule_score(same_prev == 1, np.nan_to_num(gap_prev, nan=1e9), rule_scale, rule_cutoff), 0.0)
    r_next = np.where(np.isfinite(gap_next), rule_score(same_next == 1, np.nan_to_num(gap_next, nan=1e9), rule_scale, rule_cutoff), 0.0)
    plus = sd > 0
    up = np.where(plus, gap_prev, gap_next)       # upstream / downstream in transcription order
    down = np.where(plus, gap_next, gap_prev)
    s_up = np.where(plus, same_prev, same_next)
    s_down = np.where(plus, same_next, same_prev)
    rule_up = np.where(plus, r_prev, r_next)
    rule_down = np.where(plus, r_next, r_prev)
    # directons: runs of consecutive same-strand genes within a contig
    new_run = first | np.r_[True, sd[1:] != sd[:-1]]
    run = np.cumsum(new_run) - 1
    size = np.bincount(run)[run]
    lo_end = new_run
    hi_end = np.r_[new_run[1:], True]
    head = np.where(plus, lo_end, hi_end)
    tail = np.where(plus, hi_end, lo_end)
    dpos = np.where(size == 1, 0, np.where(head, 1, np.where(tail, 3, 2)))
    # local density
    mid = (st + en) / 2.0
    big = float(max(mid.max() if n else 0, 1)) * 4 + 1e7
    key = contig_of * big + mid
    ks = np.sort(key)
    cnt = np.searchsorted(ks, key + window / 2, 'right') - np.searchsorted(ks, key - window / 2, 'left')
    density = cnt / (window / 1000.0)
    # replication
    dist = np.full(n, np.nan)
    lead = np.full(n, np.nan)
    if origin is not None:
        gids = [u[0] for u in uniq]

        def per_genome(v, g):
            return v.get(g) if isinstance(v, dict) else v
        for g in dict.fromkeys(gids):
            ori = per_genome(origin, g)
            if ori is None:
                continue
            cands = [c for c in range(len(uniq)) if uniq[c][0] == g]
            chrom = per_genome(chromosome, g)
            if chrom is None:
                c = max(cands, key=lambda c: offsets[c + 1] - offsets[c])
            else:
                c = [c for c in cands if uniq[c][1] == chrom][0]
            sl = slice(offsets[c], offsets[c + 1])
            L = per_genome(chrom_length, g) or float(en[sl].max())
            ter = per_genome(terminus, g)
            ter = (ori + L / 2.0) % L if ter is None else ter
            x = mid[sl]
            d = np.abs(x - ori)
            d = np.minimum(d, L - d)
            dter = np.abs(ter - ori)
            dter = min(dter, L - dter)
            dist[sl] = np.clip(d / max(dter, 1.0), 0, 1)
            right = ((x - ori) % L) < ((ter - ori) % L)    # replichore replicated in increasing coordinate
            lead[sl] = np.where(right, sd[sl] > 0, sd[sl] < 0).astype(float)
    out = pd.DataFrame({
        'strand': sd, 'length': en - st + 1, 'gap_up': up, 'gap_down': down, 'same_strand_up': s_up,
        'same_strand_down': s_down, 'rule_up': rule_up, 'rule_down': rule_down,
        'operon_rule': np.maximum(rule_up, rule_down), 'directon_size': size, 'directon_pos': dpos,
        'density': density, 'dist_origin': dist, 'replication_strand': lead})
    res = out.iloc[np.argsort(order)].reset_index(drop=True)
    res.index = coords.index
    return res


def origin_from_gc_skew(sequence):
    """(origin, terminus) bp positions from the cumulative GC skew: minimum of cumsum(G - C) = origin,
    maximum = terminus (Lobry 1996; Grigoriev 1998). For a chromosome sequence given 5'->3' on the plus strand."""
    s = np.frombuffer(sequence.upper().encode('ascii'), dtype=np.uint8)
    cum = np.cumsum((s == ord('G')).astype(np.int64) - (s == ord('C')).astype(np.int64))
    return int(np.argmin(cum)), int(np.argmax(cum))
