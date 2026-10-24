"""seqleak clean: a reduced-leakage split of a benchmark task, with a manifest of every change.

Items are joined into groups by the relations the audit measures: exact or reverse-complement copies, near-duplicates
(k-mer containment >= --near-threshold, any direction), overlapping or shifted windows (collinear k-mer run >=
--overlap-threshold), composition and positional twins, and, when coordinates exist, the locus (intervals within
--locus-gap on one chromosome) or the chromosome. Multi-element items are joined through every element.
- ``--mode filter`` keeps train as it is and drops every dev or test item whose group holds an item of an earlier
  split (train before dev before test); the test set keeps its identity minus the leaked items.
- ``--mode resplit`` deals whole groups to train/dev/test (seeded; groups stratified by majority label) to the target
  fractions (default: the input's proportions).
Outputs: DIR/{train,dev,test}.<fmt> (input columns, original rows), DIR/manifest.tsv, DIR/clean.json and, unless
--no-reaudit, DIR/audit_after.json / .md (the audit of the new split with the same settings)."""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from .audit import io as AIO
from .audit.kmers import auto_scale, build_index, query
from .audit.run import Ctx, components
from .audit.twins import composition_twins, positional_twins

ORDER = {'train': 0, 'dev': 1, 'test': 2}
LINKS = ['exact', 'near', 'overlap', 'twins', 'coords', 'homology', 'reference']


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _log(msg, quiet):
    if not quiet:
        print(f'[seqleak clean] {msg}', file=sys.stderr, flush=True)


def all_vs_all(seqs, k, scale, max_df, near_thr, overlap_thr):
    """Pairs (a, b) of different unique sequences with containment >= near_thr (near) or a collinear run >=
    overlap_thr (overlap), either direction."""
    idx = build_index(seqs, k=k, scale=scale)
    h = query(idx, seqs, max_df=max_df, keep_top=16, keep_c=min(near_thr, overlap_thr),
              exclude=np.arange(len(seqs)))
    c = h.containment()
    cl = h.collinear()
    near = c >= near_thr
    ov = cl >= overlap_thr
    return (h.q[near], h.r[near]), (h.q[ov], h.r[ov])


def loci(chrom, start, end, gap):
    """Locus id per interval: intervals on one chromosome closer than ``gap`` bp are merged."""
    df = pd.DataFrame(dict(c=chrom, s=start, e=end, i=np.arange(len(chrom))))
    out = np.full(len(df), -1, np.int64)
    nxt = 0
    for c, g in df.dropna().sort_values(['c', 's']).groupby('c', sort=True):
        s = g['s'].to_numpy(np.float64)
        e = np.maximum.accumulate(g['e'].to_numpy(np.float64))
        new = np.r_[True, s[1:] > e[:-1] + gap]
        lid = np.cumsum(new) - 1 + nxt
        nxt = int(lid.max()) + 1
        out[g['i'].to_numpy()] = lid
    miss = out < 0
    out[miss] = nxt + np.arange(miss.sum())
    return out


def _write(df, path, fmt, task):
    if fmt == 'parquet':
        df.to_parquet(path, index=False)
    elif fmt == 'csv':
        df.to_csv(path, index=False)
    elif fmt == 'tsv':
        df.to_csv(path, sep='\t', index=False)
    elif fmt == 'fasta':
        seqc = task.columns['seq'][0]
        idc = task.columns.get('id') or 'id'
        with open(path, 'w') as f:
            for i, (_, r) in enumerate(df.iterrows()):
                rid = r[idc] if idc in df.columns else f'item{i}'
                extra = ' '.join(f'{c}={r[c]}' for c in df.columns if c not in (seqc, idc) and not pd.isna(r[c]))
                f.write(f'>{rid} {extra}\n{r[seqc]}\n')
    else:
        raise ValueError(fmt)


def run_clean(task, refs, args, argv=None, version='', audit_kw=None):
    t0 = time.time()
    quiet = getattr(args, 'quiet', False)
    links = LINKS if args.links in (None, 'all') else [x.strip() for x in args.links.split(',')]
    for x in links:
        if x not in LINKS:
            raise AIO.InputError(f'unknown link type {x}')
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    k = args.k
    max_df = args.max_df
    ctx = Ctx(task, k, args.sketch_scale, max_df, args.seed, 0, args.folds, 0, True)
    n = ctx.n
    split = ctx.split
    nodes = n
    ea, eb, etype = [], [], []
    offs = []

    def add(a, b, t):
        ea.append(np.asarray(a, np.int64))
        eb.append(np.asarray(b, np.int64))
        etype.append((t, len(ea[-1])))

    # unique-sequence nodes per element; item -- its sequence (exact copies share a node)
    for e in range(ctx.E):
        offs.append(nodes)
        if 'exact' in links:
            # canonical id: reverse complements share a node
            add(np.arange(n), nodes + ctx.cid[e], 'exact')
            nodes += int(ctx.cid[e].max()) + 1
        else:
            add(np.arange(n), nodes + ctx.uid[e], 'self')
            nodes += int(ctx.uid[e].max()) + 1
    item_uid_node = []
    rel_counts = {}
    for e in range(ctx.E):
        seqs = ctx.useq[e]
        scale = auto_scale([len(s) for s in seqs]) if args.sketch_scale == 'auto' else int(args.sketch_scale)
        # map unique sequence -> its node (canonical or plain)
        node_of_u = np.zeros(len(seqs), np.int64)
        base_ids = ctx.cid[e] if 'exact' in links else ctx.uid[e]
        node_of_u[ctx.uid[e]] = offs[e] + base_ids
        item_uid_node.append(node_of_u)
        if 'near' in links or 'overlap' in links:
            # doubled max_df: k-mers whose document frequency exceeds max_df only once dev/test items are counted
            # would otherwise hide cross-split overlaps that the audit (fit-split frequencies) finds
            (na, nb), (oa, ob) = all_vs_all(seqs, k, scale, 2 * max_df, args.near_threshold, args.overlap_threshold)
            if 'near' in links:
                add(node_of_u[na], node_of_u[nb], 'near')
                rel_counts[f'near_pairs_e{e}'] = int(len(na))
            if 'overlap' in links:
                add(node_of_u[oa], node_of_u[ob], 'overlap')
                rel_counts[f'overlap_pairs_e{e}'] = int(len(oa))
            _log(f'element {e}: {len(seqs)} unique sequences, {len(na)} near and {len(oa)} overlap pairs', quiet)
        if 'twins' in links and ctx.E == 1:
            cp, _ = composition_twins(seqs)
            gp, _ = positional_twins(seqs)
            if len(cp):
                add(node_of_u[cp[:, 0]], node_of_u[cp[:, 1]], 'twins')
            if len(gp):
                add(node_of_u[gp[:, 0]], node_of_u[gp[:, 1]], 'twins')
            rel_counts['composition_twin_pairs'] = int(len(cp))
            rel_counts['positional_twin_pairs'] = int(len(gp))
    group_by = args.group_by
    if group_by == 'auto':
        group_by = 'locus' if task.has_coords else 'cluster'
    if group_by in ('locus', 'chromosome') and not task.has_coords:
        raise AIO.InputError(f'--group-by {group_by} needs coordinates')
    if task.has_coords and 'coords' in links and group_by in ('locus', 'chromosome'):
        it = task.items
        if group_by == 'locus':
            lid = loci(it['chrom'].to_numpy(), it['start'].to_numpy(np.float64), it['end'].to_numpy(np.float64),
                       args.locus_gap)
        else:
            lid = pd.factorize(it['chrom'].astype(str))[0].astype(np.int64)
        add(np.arange(n), nodes + lid, 'coords')
        nodes += int(lid.max()) + 1
        rel_counts['n_' + group_by] = int(len(np.unique(lid)))
    a = np.concatenate(ea) if ea else np.zeros(0, np.int64)
    b = np.concatenate(eb) if eb else np.zeros(0, np.int64)
    comp = components(nodes, a, b)
    group = pd.factorize(comp[:n])[0].astype(np.int64)
    # reference copies (dropped with --drop-reference)
    ref_drop = np.zeros(n, bool)
    if refs and args.drop_reference:
        from .audit.seqs import canonical
        ours = [canonical(s) for s in task.items['seq0']]
        for ref in refs:
            R = ref['items']
            rt = R[R['split'].isin(['test', 'all'])]
            rset = {canonical(s) for s in rt['seq0'].astype(str)}
            ref_drop |= np.array([s in rset for s in ours], bool)
            if len(rt):
                ru = sorted(set(rt['seq0'].astype(str).str.upper()))
                idx = build_index(ru, k=k, scale=auto_scale([len(s) for s in ru]))
                h = query(idx, [s.upper() for s in task.items['seq0']], max_df=max_df, keep_top=1, keep_c=1.1)
                from .audit.kmers import best_per_query
                bb = best_per_query(h, n)
                ref_drop |= bb['shared'] / np.maximum(h.n_inf, 1) >= args.near_threshold
    # relation flags per item (for the manifest reasons): which relation ties the item to another split
    G = pd.DataFrame(dict(group=group, split=split))
    rank = np.array([ORDER[s] for s in split])
    gmin = G.assign(r=rank).groupby('group')['r'].transform('min').to_numpy()
    new = split.astype(object).copy()
    reason = np.array(['kept'] * n, dtype=object)
    linked = np.array([''] * n, dtype=object)
    # one representative item of the earliest split per group
    first_rep = pd.Series(np.arange(n)).groupby([group, rank]).first()
    if args.mode == 'filter':
        drop = rank > gmin
        new[drop] = 'dropped'
        for i in np.flatnonzero(drop):
            rep = first_rep.get((group[i], gmin[i]))
            linked[i] = str(ctx.ids[rep]) if rep is not None else ''
        reason[drop] = ['linked_to_' + ('train' if g == 0 else 'dev') for g in gmin[drop]]
    else:
        fr = args.fractions
        if fr:
            f = np.array([float(x) for x in fr.split(',')], float)
            if len(f) != 3:
                raise AIO.InputError('--fractions needs TRAIN,DEV,TEST')
        else:
            f = np.array([(split == s).mean() for s in ('train', 'dev', 'test')])
        f = f / f.sum()
        rng = np.random.default_rng(args.seed)
        gs = pd.DataFrame(dict(group=group, y=ctx.yi if ctx.classification else 0))
        gsize = gs.groupby('group').size()
        gmaj = gs.groupby('group')['y'].agg(lambda v: np.bincount(v).argmax() if len(v) else 0)
        assign = {}
        names = np.array(['train', 'dev', 'test'])
        for lab in sorted(gmaj.unique()):
            gl = gmaj.index[gmaj == lab].to_numpy()
            gl = gl[rng.permutation(len(gl))]
            tot = gsize[gl].sum()
            fill = np.zeros(3)
            target = f * tot
            for g in gl:
                ratio = np.where(target > 0, fill / np.maximum(target, 1e-9), np.inf)
                j = int(np.argmin(ratio))
                assign[g] = names[j]
                fill[j] += gsize[g]
        new = np.array([assign[g] for g in group], dtype=object)
        moved = new != split
        reason[moved] = 'moved_with_group'
    if ref_drop.any():
        new[ref_drop] = 'dropped'
        reason[ref_drop] = 'copy_of_reference_test'
    # outputs
    raw = task.raw.reset_index(drop=True)
    fmt = args.out_format
    if fmt == 'auto':
        fmt = task.formats[0] if len(task.formats) == 1 else 'parquet'
    ext = dict(parquet='parquet', csv='csv', tsv='tsv', fasta='fa')[fmt]
    files = {}
    splitcol = task.columns.get('split')
    for s in ('train', 'dev', 'test'):
        sel = new == s
        if not sel.any():
            continue
        d = raw[sel].copy()
        if splitcol and splitcol in d.columns:
            d[splitcol] = s
        p = os.path.join(out, f'{s}.{ext}')
        _write(d, p, fmt, task)
        files[s] = p
    man = pd.DataFrame(dict(item_id=ctx.ids, orig_split=split, new_split=new, group=group, reason=reason,
                            linked_item=linked))
    mp = os.path.join(out, 'manifest.tsv')
    man.to_csv(mp, sep='\t', index=False)
    files['manifest'] = mp
    counts = dict(before={s: int((split == s).sum()) for s in ('train', 'dev', 'test')},
                  after={s: int((new == s).sum()) for s in ('train', 'dev', 'test', 'dropped')},
                  reasons=man['reason'].value_counts().to_dict(), n_groups=int(group.max()) + 1,
                  largest_group=int(np.bincount(group).max()))
    if ctx.classification:
        counts['labels_after'] = {s: {str(c): int(((new == s) & (ctx.yi == i)).sum()) for i, c in enumerate(ctx.classes)}
                                  for s in ('train', 'dev', 'test')}
    rep = dict(schema='seqleak.clean/1', tool=dict(name='seqleak', version=version, argv=argv),
               task=task.name, inputs=task.inputs, settings=dict(mode=args.mode, group_by=group_by, links=links,
                                                                 link_max_df=2 * max_df,
                                                                 near_threshold=args.near_threshold,
                                                                 overlap_threshold=args.overlap_threshold,
                                                                 locus_gap=args.locus_gap, fractions=args.fractions,
                                                                 drop_reference=bool(args.drop_reference), k=k,
                                                                 max_df=max_df, seed=args.seed),
               relations=rel_counts, counts=counts, outputs={k_: dict(path=os.path.abspath(v), sha256=_sha(v))
                                                             for k_, v in files.items()})
    if not args.no_reaudit and 'train' in files and 'test' in files:
        from .audit import load_task, run_audit, to_markdown, write_json
        t2 = load_task(train=[files['train']], dev=[files['dev']] if 'dev' in files else None, test=[files['test']],
                       fmt=fmt, seq_col=','.join(task.columns['seq']), label_col=task.columns['label'],
                       id_col=task.columns.get('id'), coords=args.coords if args.coords != 'auto' else 'auto',
                       task_name=task.name + ' (cleaned)', task_type=task.task_type)
        kw = dict(audit_kw or {})
        kw['quiet'] = True
        rep2, _ = run_audit(t2, refs, argv=argv, version=version, **kw)
        ap = os.path.join(out, 'audit_after.json')
        write_json(rep2, ap)
        with open(os.path.join(out, 'audit_after.md'), 'w') as fh:
            fh.write(to_markdown(rep2))
        rep['audit_after'] = dict(path=os.path.abspath(ap), verdict=rep2['verdict'],
                                  statuses={c: (r['status'], r['value']) for c, r in rep2['checks'].items()})
    rep['runtime'] = dict(seconds=round(time.time() - t0, 2))
    from .audit.report import write_json as wj
    wj(rep, os.path.join(out, 'clean.json'))
    _log(f"{task.name}: {counts['after']} (before {counts['before']}); groups {counts['n_groups']}; "
         f"{rep['runtime']['seconds']} s", quiet)
    return rep
