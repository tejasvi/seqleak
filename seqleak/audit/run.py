"""seqleak audit: run every applicable check on one task and assemble the report (schema seqleak.audit/1)."""
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

from . import stats as S
from . import thresholds as TH
from .kmers import auto_scale, best_per_query, build_index, query
from .seqs import canonical
from .shortcuts import ARTEFACT_GROUPS, COMPOSITION, features, markov_rule, motif_rule, rule_eval
from .twins import composition_twins, positional_twins
from .construct import acgt_fraction, library_groups, opposite_twin_items, unmatched_site
from . import genome as G

SCHEMA = 'seqleak.audit/1'
FAMILY = {'overlap.exact': 'overlap', 'overlap.near': 'overlap', 'overlap.windows': 'overlap', 'overlap.pair': 'overlap',
          'overlap.homology': 'overlap', 'reference.copies': 'reference', 'shortcut.length': 'shortcut',
          'shortcut.ambiguous': 'shortcut', 'shortcut.case': 'shortcut', 'shortcut.lowinfo': 'shortcut',
          'shortcut.composition': 'shortcut', 'shortcut.kmer': 'shortcut', 'shortcut.position_motif': 'shortcut',
          'construction.shuffle': 'construction', 'construction.segment': 'construction',
          'construction.synthetic_negatives': 'construction', 'construction.unmatched_site': 'construction',
          'construction.library': 'construction', 'construction.genome_membership': 'construction',
          'shortcut.anchored_motif': 'shortcut', 'position.distance': 'position', 'position.row_zero': 'position',
          'baseline.lookup': 'baseline'}
TITLES = {
    'overlap.exact': 'Exact or reverse-complement duplicates across splits',
    'overlap.near': 'Near-duplicates (k-mer containment)',
    'overlap.windows': 'Overlapping or shifted genome windows',
    'overlap.pair': 'Multi-element items copied from one training item',
    'overlap.homology': 'Homology (alignment)',
    'reference.copies': "Copies of another benchmark's sequences",
    'shortcut.length': 'Length shortcut',
    'shortcut.ambiguous': 'N / padding / ambiguity-code shortcut',
    'shortcut.case': 'Lowercase (soft-masking) shortcut',
    'shortcut.lowinfo': 'Low-information inputs (windows made mostly of N)',
    'shortcut.composition': 'GC / CpG / naturalness composition rule',
    'shortcut.kmer': 'k-mer composition rule (order-3 Markov)',
    'shortcut.position_motif': 'Fixed-position motif table',
    'construction.shuffle': 'Dinucleotide-shuffle negatives (composition twins)',
    'construction.segment': 'Segment-substitution negatives (positional twins)',
    'construction.synthetic_negatives': 'Constructed negatives anywhere in the data (opposite-label twin pairs)',
    'construction.unmatched_site': 'Negatives that lack the site defining the positives',
    'construction.library': 'Pooled-library structure (shared constant segments predict the label)',
    'construction.genome_membership': 'Negatives absent from the reference genome (membership predicts the label)',
    'shortcut.anchored_motif': 'Motif table anchored at the item ends (variable-length items)',
    'position.distance': 'Position confound (distance fields)',
    'position.row_zero': 'Position confound (gene-pair row zero)',
    'baseline.lookup': 'Nearest-training lookup baseline',
}
ALL_CHECKS = list(FAMILY)


def _log(msg, quiet):
    if not quiet:
        print(f'[seqleak audit] {msg}', file=sys.stderr, flush=True)


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union_pairs(self, a, b):
        for x, y in zip(a, b):
            rx, ry = self.find(x), self.find(y)
            if rx != ry:
                self.p[max(rx, ry)] = min(rx, ry)

    def labels(self):
        return np.array([self.find(i) for i in range(len(self.p))])


def components(n, a, b):
    """Connected components of an undirected graph on n nodes (scipy sparse)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    if n == 0:
        return np.zeros(0, np.int64)
    g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1].astype(np.int64)


def _check(cid, status, statistic, value, ci=None, n=None, th=None, reason='', evidence=None, examples=None,
           seconds=0.0):
    thr = None
    if th is not None:
        thr = dict(warn=th.get('warn'), fail=th.get('fail'), direction='>=')
    return dict(family=FAMILY[cid], title=TITLES[cid], status=status, statistic=statistic,
                value=None if value is None else float(value), ci=ci, n=n, threshold=thr, reason=reason,
                evidence=evidence or {}, examples=examples or [], seconds=round(float(seconds), 3))


def _skip(cid, reason):
    return _check(cid, 'SKIP', None, None, reason=reason)


class Ctx:
    """Shared state of one audit."""

    def __init__(self, task, k, sketch_scale, max_df, seed, n_boot, n_folds, examples, quiet):
        it = task.items
        self.task = task
        self.it = it
        self.n = len(it)
        self.split = it['split'].to_numpy()
        self.tr = self.split == 'train'
        self.dv = self.split == 'dev'
        self.te = self.split == 'test'
        self.fit = self.tr | self.dv
        # test-only (zero-shot) task: rules are cross-validated within the test items; overlap checks are skipped
        self.test_only = not self.tr.any()
        if self.test_only:
            self.tr = self.te.copy()
            self.fit = self.te.copy()
        self.E = len(task.elements)
        self.k, self.max_df, self.seed, self.n_boot = k, max_df, seed, n_boot
        self.n_folds, self.n_examples, self.quiet = n_folds, examples, quiet
        self.ids = it['item_id'].to_numpy()
        self.classification = task.task_type == 'classification'
        if self.classification:
            self.classes = sorted(it['label'].unique(), key=lambda v: (str(type(v)), v))
            self.K = len(self.classes)
            self.yi = S.encode_labels(it['label'].to_numpy(), self.classes)
        else:
            self.classes, self.K = [], 0
            self.yi = np.zeros(self.n, np.int64)
        self.yv = it['label'].to_numpy()
        # a training split with a single class cannot fit or cross-validate a rule: the rules are then
        # cross-validated within the test items, as for test-only tasks (the overlap checks still use the training split)
        self.single_class_train = bool(self.classification and not self.test_only and
                                       len(np.unique(self.yi[self.tr])) < 2)
        self.rules_within_test = self.test_only or self.single_class_train
        self.minority = None
        if self.classification and self.K == 2:
            c = np.bincount(self.yi[self.te if self.single_class_train else self.tr], minlength=2)
            self.minority = float(c.min() / max(c.sum(), 1))
        # unique sequences per element (upper case) and canonical (reverse-complement aware) ids
        self.useq, self.uid, self.cid = [], [], []
        for e in range(self.E):
            up = it[f'seq{e}'].str.upper()
            codes, uniq = pd.factorize(up)
            uniq = list(uniq)
            can = [canonical(s) for s in uniq]
            ccodes, _ = pd.factorize(pd.Series(can))
            self.useq.append(uniq)
            self.uid.append(codes.astype(np.int64))
            self.cid.append(ccodes[codes].astype(np.int64))
        if self.E == 1:
            self.key = self.cid[0]
        else:
            self.key = pd.factorize(pd.Series(['|'.join(map(str, t)) for t in zip(*self.cid)]))[0].astype(np.int64)
        lens = it['seq0'].str.len().to_numpy()
        self.scale = [auto_scale(it[f'seq{e}'].str.len().to_numpy()) if sketch_scale == 'auto' else int(sketch_scale)
                      for e in range(self.E)]
        self.lens = lens
        self.acgt = np.min(np.column_stack([acgt_fraction(it[f'seq{e}'].tolist()) for e in range(self.E)]), 1)
        self.lowinfo = np.zeros(self.n, bool)
        self.unmatched = None
        self.published_best = None
        self.metric = 'mcc'
        self.near = {}       # element -> dict of per-item arrays (test items)
        self.pairs = {}      # element -> Hits mapped to uids
        self.twin_groups = None
        self.feat = None
        self.folds = None


def _examples(ctx, rows, extra=None, n=None):
    n = ctx.n_examples if n is None else n
    out = []
    for r in list(rows)[:n]:
        d = dict(item_id=str(ctx.ids[r]), split=str(ctx.split[r]), label=_jsonable(ctx.yv[r]))
        if extra:
            d.update({k: _jsonable(v[r]) if hasattr(v, '__len__') and not isinstance(v, str) else v
                      for k, v in extra.items()})
        out.append(d)
    return out


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ----------------------------------------------------------------------------------------------- overlap.exact
def check_exact(ctx, th):
    t0 = time.time()
    key = ctx.key.copy()
    if ctx.lowinfo.any():   # windows made mostly of N are reported by shortcut.lowinfo, not as copies
        key[ctx.lowinfo] = -1 - np.arange(int(ctx.lowinfo.sum()))
    fitkeys = set(key[ctx.fit])
    trkeys = set(key[ctx.tr])
    te = np.flatnonzero(ctx.te)
    dup = np.array([key[i] in fitkeys for i in te], bool)
    sh = S.share(dup)
    ev = dict(test_in_train=S.share(np.array([key[i] in trkeys for i in te], bool))['value'])
    if ctx.dv.any():
        dvi = np.flatnonzero(ctx.dv)
        ev['dev_in_train'] = S.share(np.array([key[i] in trkeys for i in dvi], bool))['value']
        devkeys = set(key[ctx.dv])
        ev['test_in_dev'] = S.share(np.array([key[i] in devkeys for i in te], bool))['value']
    ev['test_within_duplicates'] = float(1 - len(set(key[te])) / max(len(te), 1))
    examples = []
    if ctx.classification:
        fk = pd.DataFrame(dict(key=key[ctx.fit], y=ctx.yi[ctx.fit]))
        nlab = fk.groupby('key')['y'].nunique()
        ev['fit_groups_with_conflicting_labels'] = int((nlab > 1).sum())
        if dup.any():
            tot = fk.groupby('key').size()
            same = fk.groupby(['key', 'y']).size()
            kk = key[te[dup]]
            yy = ctx.yi[te[dup]]
            num = same.reindex(pd.MultiIndex.from_arrays([kk, yy])).fillna(0).to_numpy()
            den = tot.reindex(kk).to_numpy()
            ev['label_agreement'] = float(np.mean(num / den))
        # example partners
        first_fit = pd.Series(np.flatnonzero(ctx.fit), index=key[ctx.fit])
        first_fit = first_fit[~first_fit.index.duplicated()]
        rows = te[dup][:ctx.n_examples]
        for r in rows:
            j = int(first_fit[key[r]])
            rc = any(ctx.it[f'seq{e}'].iat[r].upper() != ctx.it[f'seq{e}'].iat[j].upper() for e in range(ctx.E))
            examples.append(dict(test_item=str(ctx.ids[r]), test_label=_jsonable(ctx.yv[r]), fit_item=str(ctx.ids[j]),
                                 fit_split=str(ctx.split[j]), fit_label=_jsonable(ctx.yv[j]), reverse_complement=bool(rc)))
    st = TH.status(sh['value'], th, True)
    if st != 'PASS' and sh['k'] < th.get('min_items', 0):
        st = 'PASS'
    reason = f"{sh['k']} of {sh['n']} test items have an exact or reverse-complement copy in train/dev"
    if 'label_agreement' in ev:
        reason += f"; their copies carry the same label {ev['label_agreement']:.3f} of the time"
    ctx.exact_dup = np.zeros(ctx.n, bool)
    ctx.exact_dup[te[dup]] = True
    return _check('overlap.exact', st, 'share of test items with an exact or reverse-complement copy in a fit split',
                  sh['value'], sh['ci'], sh['n'], th, reason, ev, examples, time.time() - t0)


# ----------------------------------------------------------------------------------------------- k-mer engine
def run_engine(ctx, max_df, quiet):
    """Test (and dev) unique sequences vs fit-split unique sequences, per element. Stores per-item best partner."""
    te = np.flatnonzero(ctx.te)
    for e in range(ctx.E):
        uid = ctx.uid[e]
        ref_u = np.unique(uid[ctx.fit])
        q_u = np.unique(uid[te])
        seqs = ctx.useq[e]
        idx = build_index([seqs[u] for u in ref_u], k=ctx.k, scale=ctx.scale[e])
        h = query(idx, [seqs[u] for u in q_u], max_df=max_df, keep_top=8, keep_c=0.2)
        b = best_per_query(h, len(q_u))
        # map to uid space
        h.q_uid = q_u[h.q]
        h.r_uid = ref_u[h.r]
        ctx.pairs[e] = h
        pos = np.full(len(seqs), -1, np.int64)
        pos[q_u] = np.arange(len(q_u))
        qi = pos[uid[te]]
        n_inf = h.n_inf[qi]
        n_all = h.n_all[qi]
        c = b['shared'][qi] / np.maximum(n_inf, 1)
        cl = b['max_coll'][qi] / np.maximum(n_inf, 1)
        ctx.near[e] = dict(test_rows=te, c=c, coll=cl, partner_uid=np.where(b['r'][qi] >= 0, ref_u[np.maximum(b['r'][qi], 0)], -1),
                           coll_uid=np.where(b['max_coll_r'][qi] >= 0, ref_u[np.maximum(b['max_coll_r'][qi], 0)], -1),
                           shift=b['max_coll_shift'][qi], same=b['max_coll_same'][qi], shared=b['shared'][qi],
                           n_inf=n_inf, n_all=n_all, ref_u=ref_u, q_u=q_u, qpos=pos)
        _log(f'element {e}: {len(ref_u)} reference and {len(q_u)} query sequences, scale {ctx.scale[e]}, '
             f'{len(h.q)} kept pairs', quiet)


def _first_fit_item_of_uid(ctx, e):
    s = pd.Series(np.flatnonzero(ctx.fit), index=ctx.uid[e][ctx.fit])
    return s[~s.index.duplicated()]


def check_near(ctx, th):
    t0 = time.time()
    te = np.flatnonzero(ctx.te)
    near_any = np.zeros(len(te), bool)
    ev = dict(per_element={})
    na_all = np.zeros(len(te), bool)
    for e in range(ctx.E):
        d = ctx.near[e]
        c = d['c']
        na = (d['n_all'] == 0) | (d['n_inf'] < th['min_informative'] * d['n_all']) | (d['n_inf'] == 0)
        nr = (c >= th['near']) & ~na
        near_any |= nr
        na_all |= na
        el = dict(share_c50=S.share(nr)['value'], share_c20=float(np.mean((c >= 0.2) & ~na)),
                  share_c80=float(np.mean((c >= 0.8) & ~na)), not_assessable=float(np.mean(na)),
                  median_c=float(np.median(c[~na])) if (~na).any() else None,
                  near_excluding_exact=float(np.mean(nr & ~ctx.exact_dup[te])))
        ev['per_element'][ctx.task.elements[e]] = el
    sh = S.share(near_any)
    ev['not_assessable'] = float(np.mean(na_all))
    examples = []
    if ctx.E >= 1:
        d = ctx.near[0]
        ff = _first_fit_item_of_uid(ctx, 0)
        rows = np.flatnonzero(near_any)
        o = rows[np.argsort(-d['c'][rows], kind='stable')]
        for r in o[:ctx.n_examples]:
            pu = d['partner_uid'][r]
            j = int(ff[pu]) if pu in ff.index else -1
            examples.append(dict(test_item=str(ctx.ids[te[r]]), test_label=_jsonable(ctx.yv[te[r]]),
                                 fit_item=str(ctx.ids[j]) if j >= 0 else None,
                                 fit_label=_jsonable(ctx.yv[j]) if j >= 0 else None,
                                 containment=round(float(d['c'][r]), 4), collinear=round(float(d['coll'][r]), 4)))
    st = TH.status(sh['value'], th)
    reason = f"{sh['k']} of {sh['n']} test items share >= {th['near']:.0%} of their k-mers with one fit-split sequence"
    if ev['not_assessable'] >= th['not_assessable_warn'] and st == 'PASS':
        st = 'WARN'
        reason += f"; {ev['not_assessable']:.0%} of test items are not assessable (k-mers shared by > max_df " \
                  f"training sequences: a highly redundant set)"
    ctx.near_item = np.zeros(ctx.n, bool)
    ctx.near_item[te[near_any]] = True
    return _check('overlap.near', st, f"share of test items with k-mer containment >= {th['near']} in a fit split",
                  sh['value'], sh['ci'], sh['n'], th, reason, ev, examples, time.time() - t0)


def _interval_overlap(chrom_q, s_q, e_q, chrom_r, s_r, e_r, pad=0):
    """For each query interval: does any reference interval on the same chromosome overlap [s-pad, e+pad)?"""
    out = np.zeros(len(s_q), bool)
    rdf = pd.DataFrame(dict(c=chrom_r, s=s_r, e=e_r)).dropna()
    groups = {c: g for c, g in rdf.groupby('c')}
    for c in np.unique(chrom_q):
        if c not in groups:
            continue
        g = groups[c].sort_values('s')
        rs = g['s'].to_numpy(np.float64)
        pmax = np.maximum.accumulate(g['e'].to_numpy(np.float64))
        sel = np.flatnonzero(chrom_q == c)
        qs = s_q[sel] - pad
        qe = e_q[sel] + pad
        j = np.searchsorted(rs, qe, 'left') - 1
        ok = j >= 0
        hit = np.zeros(len(sel), bool)
        hit[ok] = pmax[j[ok]] > qs[ok]
        out[sel] = hit
    return out


def check_windows(ctx, th):
    t0 = time.time()
    te = np.flatnonzero(ctx.te)
    coll_any = np.zeros(len(te), bool)
    shifted = np.zeros(len(te), bool)
    for e in range(ctx.E):
        d = ctx.near[e]
        na = (d['n_inf'] == 0)
        f = (d['coll'] >= th['collinear']) & ~na
        coll_any |= f
        shifted |= f & ~((d['shift'] == 0) & d['same'])
    ev = dict(alignment_free=dict(share=S.share(coll_any)['value'], shifted=float(np.mean(shifted)),
                                  median_overlap=float(np.median(np.concatenate([ctx.near[e]['coll'][coll_any]
                                                                                 for e in range(ctx.E)])))
                                  if coll_any.any() else None))
    examples = []
    d = ctx.near[0]
    ff = _first_fit_item_of_uid(ctx, 0)
    if ctx.task.has_coords:
        it = ctx.it
        cq = it['chrom'].to_numpy()[te]
        sq = it['start'].to_numpy(np.float64)[te]
        eq = it['end'].to_numpy(np.float64)[te]
        fi = np.flatnonzero(ctx.fit)
        cr, sr, er = it['chrom'].to_numpy()[fi], it['start'].to_numpy(np.float64)[fi], it['end'].to_numpy(np.float64)[fi]
        ov = _interval_overlap(cq, sq, eq, cr, sr, er)
        near1k = _interval_overlap(cq, sq, eq, cr, sr, er, pad=1000)
        sh = S.share(ov)
        ev['coordinates'] = dict(overlap=sh['value'], within_1kb=float(np.mean(near1k)),
                                 test_chromosomes_in_fit=float(np.mean(np.isin(cq, np.unique(cr)))))
        ev['agreement_alignment_free_vs_coordinates'] = float(np.mean(ov == coll_any))
        stat = 'share of test intervals overlapping a fit-split interval (coordinates)'
        rows = np.flatnonzero(ov)
        for r in rows[:ctx.n_examples]:
            examples.append(dict(test_item=str(ctx.ids[te[r]]), chrom=str(cq[r]), start=_jsonable(sq[r]),
                                 end=_jsonable(eq[r])))
        reason = f"{sh['k']} of {sh['n']} test intervals overlap a training/dev interval"
    else:
        sh = S.share(coll_any)
        stat = f"share of test items with a collinear shared run >= {th['collinear']} of their k-mers (alignment-free)"
        rows = np.flatnonzero(coll_any)
        o = rows[np.argsort(-d['coll'][rows], kind='stable')]
        for r in o[:ctx.n_examples]:
            pu = d['coll_uid'][r]
            j = int(ff[pu]) if pu in ff.index else -1
            examples.append(dict(test_item=str(ctx.ids[te[r]]), fit_item=str(ctx.ids[j]) if j >= 0 else None,
                                 overlap=round(float(d['coll'][r]), 4),
                                 shift=int(d['shift'][r]) if d['same'][r] else None,
                                 strand='same' if d['same'][r] else 'opposite'))
        reason = f"{sh['k']} of {sh['n']} test items share a collinear (overlapping or shifted) segment with a " \
                 f"training/dev sequence; {ev['alignment_free']['shifted']:.3f} of test items at a non-zero offset"
    st = TH.status(sh['value'], th)
    return _check('overlap.windows', st, stat, sh['value'], sh['ci'], sh['n'], th, reason, ev, examples,
                  time.time() - t0)


def _item_pairs_from_hits(ctx, e, cmin, cap=200):
    """(test item, fit item, containment) for element e hits with containment >= cmin, expanded from unique
    sequences to items (at most ``cap`` items per unique sequence, seeded)."""
    h = ctx.pairs[e]
    c = h.containment()
    sel = c >= cmin
    qu, ru, cc = h.q_uid[sel], h.r_uid[sel], c[sel]
    rng = np.random.default_rng(ctx.seed)
    te_rows = pd.Series(np.flatnonzero(ctx.te)).groupby(ctx.uid[e][ctx.te]).apply(lambda s: s.to_numpy())
    fit_rows = pd.Series(np.flatnonzero(ctx.fit)).groupby(ctx.uid[e][ctx.fit]).apply(lambda s: s.to_numpy())
    A, B, C = [], [], []
    for q, r, v in zip(qu, ru, cc):
        ta = te_rows.get(q)
        fb = fit_rows.get(r)
        if ta is None or fb is None:
            continue
        if len(fb) > cap:
            fb = np.sort(rng.choice(fb, cap, replace=False))
        A.append(np.repeat(ta, len(fb)))
        B.append(np.tile(fb, len(ta)))
        C.append(np.full(len(ta) * len(fb), v))
    if not A:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0)
    return np.concatenate(A), np.concatenate(B), np.concatenate(C)


def check_pair(ctx, th):
    t0 = time.time()
    if ctx.E < 2:
        return _skip('overlap.pair', 'single-element items')
    te = np.flatnonzero(ctx.te)
    keys = None
    for e in range(ctx.E):
        a, b, _ = _item_pairs_from_hits(ctx, e, th['near'])
        k = a * ctx.n + b
        keys = set(k.tolist()) if keys is None else keys & set(k.tolist())
    covered = np.zeros(ctx.n, bool)
    partner = {}
    for kk in keys:
        a, b = divmod(kk, ctx.n)
        covered[a] = True
        partner.setdefault(a, []).append(b)
    sh = S.share(covered[te])
    ev = {}
    examples = []
    if ctx.classification and sh['k']:
        rows = [r for r in te if covered[r]]
        agree = [np.mean(ctx.yi[partner[r]] == ctx.yi[r]) for r in rows]
        ev['label_agreement'] = float(np.mean(agree))
        ev['positive_rate_covered'] = float(np.mean(ctx.yi[rows] == ctx.K - 1)) if ctx.K == 2 else None
        for r in rows[:ctx.n_examples]:
            j = partner[r][0]
            examples.append(dict(test_item=str(ctx.ids[r]), test_label=_jsonable(ctx.yv[r]), fit_item=str(ctx.ids[j]),
                                 fit_label=_jsonable(ctx.yv[j])))
    ctx.pair_partner = partner
    st = TH.status(sh['value'], th)
    reason = f"{sh['k']} of {sh['n']} test items have every element near-copied (containment >= {th['near']}) " \
             f"from one training/dev item"
    return _check('overlap.pair', st, 'share of test items whose elements all come from one fit-split item',
                  sh['value'], sh['ci'], sh['n'], th, reason, ev, examples, time.time() - t0)


# ----------------------------------------------------------------------------------------------- reference copies
def check_reference(ctx, refs, th, max_df):
    t0 = time.time()
    if not refs:
        return _skip('reference.copies', 'no --reference given')
    te = np.flatnonzero(ctx.te)
    per = {}
    worst = (0.0, None)
    examples = []
    for ref in refs:
        R = ref['items']
        rsplit = R['split'].to_numpy()
        r_fit = np.isin(rsplit, ['train', 'dev', 'all'])
        r_te = np.isin(rsplit, ['test', 'all'])
        rs = R['seq0'].astype(str).str.upper().tolist()
        rcan = [canonical(s) for s in rs]
        ours = ctx.it['seq0'].str.upper().tolist()
        ocan = [canonical(s) for s in ours]
        rfitset = set(np.array(rcan, dtype=object)[r_fit])
        ofitset = set(np.array(ocan, dtype=object)[ctx.fit])
        ex_a = np.array([ocan[i] in rfitset for i in te], bool)
        rte = np.flatnonzero(r_te)
        if ctx.test_only:      # no fit split of our own: only the direction "our test items in the reference" applies
            rte = rte[:0]
        ex_b = np.array([rcan[i] in ofitset for i in rte], bool)
        # near copies (element 0)
        nr_a = np.zeros(len(te), bool)
        nr_b = np.zeros(len(rte), bool)
        if r_fit.any() and len(te):
            ru = sorted(set(np.array(rs, dtype=object)[r_fit]))
            idx = build_index(ru, k=ctx.k, scale=ctx.scale[0])
            qs = [ours[i] for i in te]
            h = query(idx, qs, max_df=max_df, keep_top=1, keep_c=1.1)
            b = best_per_query(h, len(qs))
            nr_a = b['shared'] / np.maximum(h.n_inf, 1) >= th['near']
        if len(rte) and ctx.fit.any():
            ou = sorted(set(np.array(ours, dtype=object)[ctx.fit]))
            idx = build_index(ou, k=ctx.k, scale=ctx.scale[0])
            qs = [rs[i] for i in rte]
            h = query(idx, qs, max_df=max_df, keep_top=1, keep_c=1.1)
            b = best_per_query(h, len(qs))
            nr_b = b['shared'] / np.maximum(h.n_inf, 1) >= th['near']
        a = S.share(ex_a | nr_a)
        bsh = S.share(ex_b | nr_b) if len(rte) else dict(value=None, ci=[None, None], k=0, n=0)
        d = dict(test_in_reference_fit=a['value'], test_in_reference_fit_ci=a['ci'],
                 test_in_reference_fit_exact=float(np.mean(ex_a)) if len(ex_a) else None,
                 reference_test_in_fit=bsh['value'], reference_test_in_fit_ci=bsh['ci'],
                 reference_test_in_fit_exact=float(np.mean(ex_b)) if len(ex_b) else None,
                 reference_n=int(len(R)), reference_has_split=bool(np.isin(rsplit, ['train', 'dev', 'test']).any()))
        # label agreement for exact copies of our test items in the reference fit split
        if ctx.classification and R['label'].notna().all() and ex_a.any():
            m = pd.DataFrame(dict(c=np.array(rcan, dtype=object)[r_fit], y=R['label'].to_numpy()[r_fit]))
            lab = m.drop_duplicates('c').set_index('c')['y']
            ag = [str(lab[ocan[i]]) == str(ctx.yv[i]) for i in te[ex_a]]
            d['label_agreement_exact'] = float(np.mean(ag))
        per[ref['name']] = d
        v = max(x for x in (a['value'], bsh['value']) if x is not None)
        if v >= worst[0]:
            worst = (v, ref['name'])
            examples = [dict(reference=ref['name'], test_item=str(ctx.ids[te[i]])) for i in
                        np.flatnonzero(ex_a | nr_a)[:ctx.n_examples]]
    st = TH.status(worst[0], th)
    reason = f"largest share {worst[0]:.3f} ({worst[1]}): test items with a copy in the reference's train/dev, or " \
             f"reference test items with a copy in this task's train/dev"
    return _check('reference.copies', st, "largest share of copies between this task and a reference benchmark",
                  worst[0], None, None, th, reason, dict(per_reference=per), examples, time.time() - t0)


# ----------------------------------------------------------------------------------------------- twins
def _twin_stats(ctx, pairs_u, cid, th, title_stat, default_direction='flip'):
    """Twin pairs over unique sequences of element 0 -> coverage of test items by fit-split twins, label relation,
    twin rule accuracy on covered test items."""
    t0 = time.time()
    uid = ctx.uid[0]
    nu = len(ctx.useq[0])
    te = np.flatnonzero(ctx.te)
    if len(pairs_u) == 0:
        ev = dict(n_pairs=0)
        return _check(cid, 'PASS', title_stat, 0.0, S.wilson(0, len(te)), len(te), th,
                      'no twin pairs', ev, [], time.time() - t0), np.zeros(ctx.n, bool)
    K = max(ctx.K, 1)
    # class histograms per unique sequence: fit items and all items
    fit_hist = np.zeros((nu, K), np.int64)
    all_hist = np.zeros((nu, K), np.int64)
    if ctx.classification:
        np.add.at(fit_hist, (uid[ctx.fit], ctx.yi[ctx.fit]), 1)
        np.add.at(all_hist, (uid, ctx.yi), 1)
    has_fit = fit_hist.sum(1) > 0
    a, b = pairs_u[:, 0], pairs_u[:, 1]
    # label relation from fit-fit twin pairs (majority labels)
    fl = fit_hist.argmax(1)
    ff = has_fit[a] & has_fit[b]
    p_same = float(np.mean(fl[a[ff]] == fl[b[ff]])) if ff.any() else 0.5
    al = all_hist.argmax(1)
    opp_all = float(np.mean(al[a] != al[b])) if len(a) else 0.0
    in_pair = np.zeros(nu, bool)
    in_pair[a] = True
    in_pair[b] = True
    within_share = float(np.mean(in_pair[uid]))
    # partners with fit labels for each unique sequence
    part_hist = np.zeros((nu, K), np.int64)
    np.add.at(part_hist, a, fit_hist[b])
    np.add.at(part_hist, b, fit_hist[a])
    has_part = part_hist.sum(1) > 0
    cov = has_part[uid[te]]
    sh = S.share(cov)
    ev = dict(n_pairs=int(len(a)), p_same_label_fit_pairs=p_same, opposite_label_share_all_pairs=opp_all,
              share_items_in_twin_pairs=within_share)
    rule_acc = None
    inf_acc = None
    if ctx.classification and cov.any():
        pl = part_hist[uid[te[cov]]].argmax(1)
        yt = ctx.yi[te[cov]]
        if ff.any():
            direction = 'copy' if p_same >= 0.5 else 'flip'
        else:
            direction = default_direction
        if ctx.K != 2 and direction == 'flip':
            direction = 'exclude'
        # the twin rule for the best-rule summary: direction learned on fit-split twin pairs only (the check's
        # default when the fit splits hold no twin pair: flip for composition twins, copy for positional twins)
        pred = pl if direction == 'copy' else (1 - pl if direction == 'flip' else np.full(len(pl), -1))
        rule_acc = float(np.mean(pred == yt)) if direction != 'exclude' else float(np.mean(pl != yt))
        # label-informativeness of the twin relation on test (audit statistic, direction-free)
        acc_copy = float(np.mean(pl == yt))
        # chance agreement of a test label with a random fit-split label (label frequencies of the two splits)
        pt = np.bincount(ctx.yi[te], minlength=K) / max(len(te), 1)
        pp = np.bincount(ctx.yi[ctx.fit], minlength=K) / max(int(ctx.fit.sum()), 1)
        p_e = float((pt * pp).sum())
        if th.get('chance_corrected'):
            k_copy = (acc_copy - p_e) / (1 - p_e) if p_e < 1 else 0.0
            k_flip = (p_e - acc_copy) / p_e if p_e > 0 else 0.0
            inf_dir = 'copy' if k_copy >= k_flip else 'flip'
            kappa = max(k_copy, k_flip)
            inf_acc = acc_copy if inf_dir == 'copy' else 1 - acc_copy
            chance = p_e if inf_dir == 'copy' else 1 - p_e
            ev['kappa'] = float(kappa)
            ev['chance_agreement'] = chance
        else:
            inf_acc = max(acc_copy, 1 - acc_copy)
            inf_dir = 'copy' if acc_copy >= 0.5 else 'flip'
        ev['twin_rule_accuracy_covered'] = rule_acc
        ev['twin_rule_direction'] = direction
        ev['label_determined_share_covered'] = inf_acc
        ev['label_determined_share_ci'] = S.wilson(int(round(inf_acc * len(yt))), len(yt))
        ev['label_relation_on_test'] = inf_dir
        tp = np.full(ctx.n, -1, np.int64)
        tp[te[cov]] = pred
        ctx.twin_pred = getattr(ctx, 'twin_pred', {})
        ctx.twin_pred[cid] = tp
    if th.get('chance_corrected') and inf_acc is not None:
        # kappa >= 2 * rule_acc - 1 (0.4: the same as agreement 0.7 at balance), agreement above chance, >= 10 items
        informative = ev['kappa'] >= 2 * th['rule_acc'] - 1 and int(cov.sum()) >= 10 and \
            ev['label_determined_share_ci'][0] > ev['chance_agreement']
    else:
        informative = inf_acc is not None and inf_acc >= th['rule_acc'] and \
            ev['label_determined_share_ci'][0] is not None and ev['label_determined_share_ci'][0] > 0.5
    constructed = informative and (th.get('fail_direction', 'flip') == 'any' or ev.get('label_relation_on_test') == 'flip')
    if constructed:
        st = TH.status(sh['value'], th, True)
    elif informative:
        st = 'WARN' if sh['value'] >= th['warn'] else 'PASS'   # same-label twins: relatedness at fixed offsets
    else:
        st = 'PASS'
    if st == 'PASS' and within_share >= th['within_warn']:
        if th.get('chance_corrected'):
            # v1.1: twin pairs inside the data must disagree beyond chance (flip kappa), not merely in half of the pairs
            pi = np.bincount(ctx.yi, minlength=K) / max(ctx.n, 1)
            p_same = float((pi * pi).sum())
            k_flip = (opp_all - (1 - p_same)) / p_same if p_same > 0 else 0.0
            ev['within_flip_kappa'] = float(k_flip)
            if k_flip >= 2 * th['rule_acc'] - 1:
                st = 'WARN'
        elif opp_all >= 0.5:
            st = 'WARN'
    reason = f"{sh['k']} of {sh['n']} test items have a twin in train/dev"
    if inf_acc is not None:
        reason += f"; their label is the partner's label{' flipped' if ev['label_relation_on_test'] == 'flip' else ''} " \
                  f"for {inf_acc:.3f} of them"
        if ev['label_relation_on_test'] == 'copy':
            reason += ' (same-label twins: relatedness at fixed offsets, not constructed negatives)'
    reason += f"; {within_share:.3f} of all items are in twin pairs, {opp_all:.3f} of pairs with different labels"
    examples = []
    pu = {}
    for x, y in zip(a, b):
        pu.setdefault(x, y)
        pu.setdefault(y, x)
    fitfirst = _first_fit_item_of_uid(ctx, 0)
    for r in te[cov][:ctx.n_examples]:
        u = uid[r]
        # a partner with a fit item
        partner = None
        for x, y in ((a, b), (b, a)):
            m = (x == u) & has_fit[y]
            if m.any():
                partner = int(fitfirst[y[m][0]])
                break
        examples.append(dict(test_item=str(ctx.ids[r]), test_label=_jsonable(ctx.yv[r]),
                             fit_item=str(ctx.ids[partner]) if partner is not None else None,
                             fit_label=_jsonable(ctx.yv[partner]) if partner is not None else None))
    covered = np.zeros(ctx.n, bool)
    covered[te[cov]] = True
    return _check(cid, st, title_stat, sh['value'], sh['ci'], sh['n'], th, reason, ev, examples,
                  time.time() - t0), covered


def check_twins(ctx, th_s, th_g, want_s, want_g):
    out = {}
    if ctx.E != 1:
        if want_s:
            out['construction.shuffle'] = _skip('construction.shuffle', 'multi-element items')
        if want_g:
            out['construction.segment'] = _skip('construction.segment', 'multi-element items')
        return out, None, None
    seqs = ctx.useq[0]
    cp = gp = np.zeros((0, 2), np.int64)
    if ctx.test_only:
        # no training split: twin pairs are still found (constructed negatives, CV folds) but nothing crosses a split
        cp, _ = composition_twins(seqs, max_identity=th_s['max_identity'])
        gp, _ = positional_twins(seqs, min_identity=th_g['min_identity'])
        for cid, want in (('construction.shuffle', want_s), ('construction.segment', want_g)):
            if want:
                out[cid] = _skip(cid, 'test-only task: no training split to cross')
        return out, cp, gp
    if want_s:
        t0 = time.time()
        cp, ci = composition_twins(seqs, max_identity=th_s['max_identity'])
        r, cov_s = _twin_stats(ctx, cp, 'construction.shuffle', th_s,
                               'share of test items with a composition twin (same dinucleotide counts) in a fit split')
        r['evidence']['median_identity'] = float(np.median(ci)) if len(ci) else None
        r['seconds'] = round(time.time() - t0, 3)
        out['construction.shuffle'] = r
    if want_g:
        t0 = time.time()
        gp, gi = positional_twins(seqs, min_identity=th_g['min_identity'])
        r, cov_g = _twin_stats(ctx, gp, 'construction.segment', th_g,
                               'share of test items with a positional twin (segments kept in place) in a fit split',
                               default_direction='copy')
        if len(gi):
            r['evidence']['identity_quartiles'] = [float(np.percentile(gi, q)) for q in (25, 50, 75)]
            r['evidence']['share_pairs_identity_ge_0.9'] = float(np.mean(gi >= 0.9))
        r['seconds'] = round(time.time() - t0, 3)
        out['construction.segment'] = r
    return out, cp, gp


# ----------------------------------------------------------------------------------------------- v1.1 construction
def check_lowinfo(ctx, th):
    t0 = time.time()
    te = np.flatnonzero(ctx.te)
    low = ctx.lowinfo[te]
    sh = S.share(low)
    ev = dict(min_acgt_fraction=th['min_acgt'], share_all_items=float(ctx.lowinfo.mean()))
    if ctx.classification and low.any():
        cnt = np.bincount(ctx.yi[te][low], minlength=ctx.K)
        base = np.bincount(ctx.yi[te], minlength=ctx.K) / max(len(te), 1)
        ev['label_counts'] = {str(ctx.classes[i]): int(c) for i, c in enumerate(cnt)}
        ev['majority_share'] = float(cnt.max() / cnt.sum())
        ev['that_label_overall'] = float(base[int(cnt.argmax())])
    st = TH.status(sh['value'], th)
    if st != 'PASS' and sh['k'] < th.get('min_items', 0):
        st = 'PASS'
    reason = f"{sh['k']} of {sh['n']} test items have less than {th['min_acgt']:.0%} A/C/G/T (windows made mostly of N " \
             f"or other ambiguity codes); they are left out of the overlap checks"
    if 'majority_share' in ev:
        reason += f"; {ev['majority_share']:.3f} of them carry one label (that label: {ev['that_label_overall']:.3f} of all test items)"
    examples = [dict(test_item=str(ctx.ids[te[i]]), label=_jsonable(ctx.yv[te[i]])) for i in np.flatnonzero(low)[:ctx.n_examples]]
    return _check('shortcut.lowinfo', st, 'share of test items made mostly of N or other ambiguity codes', sh['value'],
                  sh['ci'], sh['n'], th, reason, ev, examples, time.time() - t0)


def check_synthetic_negatives(ctx, cp, gp, th):
    t0 = time.time()
    cid = 'construction.synthetic_negatives'
    if ctx.E != 1:
        return _skip(cid, 'multi-element items')
    if not ctx.classification:
        return _skip(cid, 'regression labels')
    stat = 'share of test items in twin pairs with different labels (pairs anywhere in the data)'
    n_te = int(ctx.te.sum())
    kinds = [(nm, p) for nm, p in (('composition', cp), ('positional', gp)) if p is not None and len(p)]
    if not kinds:
        return _check(cid, 'PASS', stat, 0.0, S.wilson(0, n_te), n_te, th, 'no twin pairs', dict(n_twin_pairs=0), [],
                      time.time() - t0)
    # each twin kind on its own: a kind counts when its pairs disagree beyond chance (flip kappa), so same-label
    # positional twins (relatedness) do not dilute dinucleotide-shuffle twins, and vice versa
    mark = np.zeros(ctx.n, bool)
    ev = dict(kinds={})
    for nm, P in kinds:
        mk, npairs, kappa = opposite_twin_items(P, ctx.uid[0], ctx.yi, ctx.K)
        counted = kappa is not None and kappa >= th['kappa'] and npairs >= th.get('min_pairs', 10)
        ev['kinds'][nm] = dict(n_pairs=npairs, flip_kappa=kappa, share_items_opposite=float(mk.mean()), counted=bool(counted))
        if counted:
            mark |= mk
    sh = S.share(mark[ctx.te])
    st = TH.status(sh['value'], th)
    ev['share_all_items'] = float(mark.mean())
    desc = '; '.join(f"{nm}: {d['n_pairs']} pairs, flip kappa {d['flip_kappa']:.2f}" + (' (counted)' if d['counted'] else '')
                     for nm, d in ev['kinds'].items())
    reason = f"{sh['k']} of {sh['n']} test items are in twin pairs with different labels of a twin kind whose pairs " \
             f"disagree beyond chance (dinucleotide shuffles or segment substitutions of each other; {mark.mean():.3f} " \
             f"of all items); {desc}"
    rows = np.flatnonzero(mark & ctx.te)
    examples = [dict(item=str(ctx.ids[r]), split=str(ctx.split[r]), label=_jsonable(ctx.yv[r])) for r in rows[:ctx.n_examples]]
    return _check(cid, st, stat, sh['value'], sh['ci'], sh['n'], th, reason, ev, examples, time.time() - t0)


def check_unmatched_site(ctx, th):
    t0 = time.time()
    cid = 'construction.unmatched_site'
    if ctx.E != 1:
        return _skip(cid, 'multi-element items')
    if not ctx.classification:
        return _skip(cid, 'regression labels')
    r = unmatched_site(ctx.it['seq0'].tolist(), ctx.yi, ctx.K, np.flatnonzero(ctx.rule_tr), np.flatnonzero(ctx.te),
                       site_min=th['site_min'], seed=ctx.seed)
    if r is None:
        return _skip(cid, 'variable or long sequences')
    ctx.unmatched = r
    v = r['statistic']
    if not r['found']:
        return _check(cid, 'PASS', 'share of other-class test items lacking the site that defines a class', 0.0, None,
                      int(ctx.te.sum()), th, f"no fixed-position k-mer is in >= {th['site_min']:.0%} of a class",
                      dict(found=False), [], time.time() - t0)
    te = np.flatnonzero(ctx.te)
    n_other = int((ctx.yi[te] != r['cls']).sum())
    st = TH.status(v, th)
    reason = f"{r['kmer']} at position {r['pos']} is in {r['eval_rate_class']:.3f} of class {ctx.classes[r['cls']]} test " \
             f"items and {r['eval_rate_other']:.3f} of the others: {v:.3f} of the other test items lack the site " \
             f"({'within test' if ctx.rules_within_test else 'train'}: {r['fit_rate_class']:.3f} vs {r['fit_rate_other']:.3f})"
    if ctx.K > 2 and st == 'FAIL' and not th.get('multiclass_fail', False):
        # with more than two classes, classes can be defined by different sites (e.g. donor vs acceptor): biology and
        # construction cannot be told apart, so the check points but does not FAIL
        st = 'WARN'
        reason += '; more than two classes: WARN at most'
    if ctx.rules_within_test:
        reason += _within_note(ctx).replace('cross-validated', 'site chosen and scored')
    return _check(cid, st, 'share of other-class test items lacking the site that defines a class', v,
                  S.wilson(int(round(v * n_other)), n_other), n_other, th, reason,
                  dict({k_: (_jsonable(x) if not isinstance(x, str) else x) for k_, x in r.items()},
                       cls_label=_jsonable(ctx.classes[r['cls']])), [], time.time() - t0)


def check_library(ctx, th):
    t0 = time.time()
    cid = 'construction.library'
    if ctx.E != 1:
        return _skip(cid, 'multi-element items')
    sig, bbs = library_groups(ctx.it['seq0'].str.upper().tolist(), k=ctx.k, min_df=th['min_df'],
                              min_kmers=th['min_kmers'], pos_consistency=th['pos_consistency'],
                              max_items=th['max_items'], seed=ctx.seed)
    stat = 'test MCC (classification) or correlation (regression) of the library-group rule'
    ev = dict(n_backbones=len(bbs), backbones=[dict(n_kmers=b['n_kmers'], share=b['share'], offset=b['offset'])
                                              for b in bbs])
    if not bbs:
        return _check(cid, 'PASS', stat, 0.0, None, int(ctx.te.sum()), th,
                      f"no constant segment shared by >= {th['min_df']:.0%} of items at one offset", ev, [], time.time() - t0)
    g = pd.factorize(sig)[0]
    sizes = np.bincount(g)
    big = int((sizes >= th.get('min_group_share', 0.01) * ctx.n).sum())
    ev['n_groups'] = int(len(sizes))
    ev['groups_ge_1pct'] = big
    ev['group_sizes_top'] = [int(x) for x in np.sort(sizes)[::-1][:8]]
    if big < 2:
        return _check(cid, 'PASS', stat, 0.0, None, int(ctx.te.sum()), th,
                      'one library: the shared constant segment is in (almost) every item, so there is no '
                      'between-library structure', ev, [], time.time() - t0)
    te = np.flatnonzero(ctx.te)
    G = int(g.max()) + 1

    def fit_predict(tr_rows, ev_rows):
        if ctx.classification:
            cnt = np.zeros((G, ctx.K), np.int64)
            np.add.at(cnt, (g[tr_rows], ctx.yi[tr_rows]), 1)
            glob = int(np.argmax(np.bincount(ctx.yi[tr_rows], minlength=ctx.K)))
            lab = np.where(cnt.sum(1) > 0, cnt.argmax(1), glob)
            return lab[g[ev_rows]]
        yv = ctx.yv.astype(np.float64)
        s_ = np.bincount(g[tr_rows], weights=yv[tr_rows], minlength=G)
        c_ = np.bincount(g[tr_rows], minlength=G)
        mean = np.where(c_ > 0, s_ / np.maximum(c_, 1), yv[tr_rows].mean())
        return mean[g[ev_rows]]

    if ctx.rules_within_test:
        f = np.random.default_rng(ctx.seed).permutation(len(te)) % 5
        pred = np.zeros(len(te)) if not ctx.classification else np.zeros(len(te), np.int64)
        for q in range(5):
            pred[f == q] = fit_predict(te[f != q], te[f == q])
    else:
        pred = fit_predict(np.flatnonzero(ctx.tr), te)
    if ctx.classification:
        v = S.score(ctx.yi[te], pred, ctx.K, 'mcc')
        ci = S.bootstrap_ci(ctx.yi[te], pred, ctx.K, 'mcc', ctx.n_boot, ctx.seed)
        kind = 'MCC'
    else:
        yt = ctx.yv[te].astype(np.float64)
        sst = float(((yt - yt.mean()) ** 2).sum())
        r2 = 1 - float(((yt - pred) ** 2).sum()) / sst if sst > 0 else 0.0
        v = float(np.sqrt(max(r2, 0.0)))
        ci = None
        ev['test_r2'] = r2
        kind = 'correlation (square root of R2)'
    st = TH.status(v, th)
    reason = f"{len(bbs)} constant segment(s) shared by >= {th['min_df']:.0%} of items at a fixed offset split the items into " \
             f"{big} groups; predicting the label from the group " \
             f"({'5-fold CV within the test items' if ctx.rules_within_test else 'fitted on train'}) gives test {kind} {v:.3f}"
    return _check(cid, st, stat, v, ci, len(te), th, reason, ev, [], time.time() - t0)


def check_genome_membership(ctx, th, genome, registry=None, method='kmer', minimap2_bin=None, threads=1, workdir=None):
    """Rules v1.2: does membership in the reference genome(s) predict the label? (Negatives that were generated rather
    than drawn from the organism's genome are not members.)"""
    t0 = time.time()
    cid = 'construction.genome_membership'
    stat = 'test MCC of the train-fitted membership rule'
    if not genome:
        return _skip(cid, 'no --genome given')
    if ctx.E != 1:
        return _skip(cid, 'multi-element items')
    if not ctx.classification:
        return _skip(cid, 'regression labels')
    med = float(np.median(ctx.lens))
    if med < th['min_len']:
        return _skip(cid, f"items too short to place in a genome (median length {med:.0f} nt; the check needs "
                          f"{th['min_len']} nt)")
    try:
        paths = G.resolve(genome, registry)
    except FileNotFoundError as e:
        return _skip(cid, str(e))
    seqs = ctx.it['seq0'].tolist()
    try:
        if method == 'minimap2':
            memb, info = G.minimap2_membership(seqs, paths, minimap2_bin, threads, workdir, th['align_cov'], th['align_ident'])
        else:
            memb, info = G.kmer_membership(seqs, paths, th['k'], th['per_item'])
    except FileNotFoundError as e:
        return _skip(cid, str(e))
    ctx.genome_memb = memb
    m0 = np.nan_to_num(memb, nan=0.0)
    member = m0 >= th['member_min']
    ev = dict(info, references=[dict(file=os.path.basename(pth), bytes=os.path.getsize(pth)) for pth in paths],
              member_min=th['member_min'])
    shares = {}
    for sp, mask in (('train', ctx.split == 'train'), ('test', ctx.te)):
        shares[sp] = {str(_jsonable(ctx.classes[c])): dict(n=int((mask & (ctx.yi == c)).sum()),
                                                            members=float(member[mask & (ctx.yi == c)].mean()) if (mask & (ctx.yi == c)).any() else None,
                                                            mean_membership=float(m0[mask & (ctx.yi == c)].mean()) if (mask & (ctx.yi == c)).any() else None)
                      for c in range(ctx.K)}
    ev['shares'] = shares
    sh_txt = ', '.join(f"class {c} {d['members']:.2f}" for c, d in shares['test'].items() if d['members'] is not None)
    r = rule_eval(m0, ctx.yi, ctx.K, ctx.rule_tr, ctx.te, ctx.rule_folds, 'mcc', ctx.n_boot, ctx.seed,
                  cv_as_test=ctx.rules_within_test)
    if r.get('constant'):
        return _check(cid, 'PASS', stat, 0.0, [0.0, 0.0], int(ctx.te.sum()), th,
                      f'membership constant on train; near-full-length members among test items: {sh_txt}', ev, [],
                      time.time() - t0)
    ctx.rule_preds[cid] = ('membership', r['pred_test'], r['cv'])
    st, crit = _status_rule(r['test'], r['cv'], th, r, ctx.minority)
    ev.update(cv_mcc=r['cv'], test_auroc=r.get('test_auroc'), cv_auroc=r.get('cv_auroc'), threshold=r.get('threshold'),
              direction=r.get('direction'), status_from=crit)
    reason = f"membership rule (fitted on train): test MCC {r['test']:.3f} (CV {r['cv']:.3f})" + \
             (f"; test AUROC {r['test_auroc']:.3f}" + (' (sets the status)' if crit == 'auroc' else '')
              if r.get('test_auroc') is not None else '') + \
             f"; near-full-length members (membership >= {th['member_min']}) among test items: {sh_txt}" + \
             _within_note(ctx)
    return _check(cid, st, stat, r['test'], r['ci'], int(ctx.te.sum()), th, reason, ev, [], time.time() - t0)


def _within_note(ctx):
    if ctx.test_only:
        return ' (test-only task: cross-validated within the test items)'
    if ctx.single_class_train:
        return ' (training split has one class: cross-validated within the test items)'
    return ''


def _anchored_seqs(seqs, A, anchors=None):
    """First and last A bases of each item (right-aligned end, N where shorter), or the 2A bases around a given site."""
    out = []
    for i, x in enumerate(seqs):
        if anchors is not None:
            a = min(max(int(anchors[i]), 0), len(x))
            lo, hi = a - A, a + A
            out.append('N' * max(0, -lo) + x[max(lo, 0):min(hi, len(x))] + 'N' * max(0, hi - len(x)))
        else:
            out.append(x[:A].ljust(A, 'N') + x[-A:].rjust(A, 'N'))
    return out


# ----------------------------------------------------------------------------------------------- shortcuts
def _status_one(test, cv, warn, fail):
    if test is None:
        return 'SKIP'
    if fail is not None and test >= fail and (cv or 0) >= fail:
        return 'FAIL'
    if warn is not None and test >= warn and (cv or 0) >= warn:
        return 'WARN'
    return 'PASS'


RANK = {'SKIP': -1, 'PASS': 0, 'WARN': 1, 'FAIL': 2}


def _use_auroc(th, minority):
    """AUROC criterion for binary labels whose minority class (train share) is below th['auroc_max_minority']."""
    return bool(th.get('auroc')) and minority is not None and minority < th.get('auroc_max_minority', 0.2)


def _status_rule(test, cv, th, r=None, minority=None):
    """Status of a rule from its test and CV MCC; for imbalanced binary labels (see _use_auroc) also from its test
    and CV AUROC against (1 + MCC threshold) / 2, keeping the more severe (under strong imbalance a rule's MCC stays
    low even when its AUROC is high). Returns (status, criterion)."""
    st = _status_one(test, cv, th.get('warn'), th.get('fail'))
    if _use_auroc(th, minority) and r is not None and r.get('test_auroc') is not None:
        half = lambda v: None if v is None else (1 + v) / 2
        sa = _status_one(r['test_auroc'], r.get('cv_auroc'), half(th.get('warn')), half(th.get('fail')))
        if RANK[sa] > RANK[st]:
            return sa, 'auroc'
    return st, 'mcc'


def _metric_score(ctx, r, metric):
    """Rule score and chance in the published metric (mcc, acc, f1 from labels; auroc from the rule's scores)."""
    te = np.flatnonzero(ctx.te)
    yt = ctx.yi[te]
    if metric == 'auroc':
        if ctx.K != 2 or r.get('test_auroc') is None:
            return None, None
        return float(r['test_auroc']), 0.5
    maj = int(np.argmax(np.bincount(ctx.yi[ctx.fit], minlength=ctx.K)))
    return S.score(yt, r['pred_test'], ctx.K, metric), S.score(yt, np.full(len(te), maj), ctx.K, metric)


def _relative(ctx, r, th, st):
    """FAIL when a rule that already WARNs reaches th['relative_fail'] of the best published above-chance score
    (only with --published-best; the manual audits' convention for construction shortcuts)."""
    if ctx.published_best is None or th.get('relative_fail') is None:
        return st, None
    sc, ch = _metric_score(ctx, r, ctx.metric)
    if sc is None:
        return st, None
    pub = float(ctx.published_best)
    if pub <= ch:
        return st, dict(published_best=pub, metric=ctx.metric, rule_score=sc, chance=ch, ratio=None)
    ratio = (sc - ch) / (pub - ch)
    info = dict(published_best=pub, metric=ctx.metric, rule_score=sc, chance=ch, ratio=ratio,
                relative_fail=th['relative_fail'])
    if st in ('WARN', 'FAIL') and ratio >= th['relative_fail']:
        return 'FAIL', info
    return st, info


def check_shortcuts(ctx, want, ths):
    out = {}
    if not ctx.classification:
        for c in want:
            out[c] = _skip(c, 'regression labels: rules not implemented in this version')
        return out
    tr = ctx.rule_tr
    te = ctx.te
    K = ctx.K
    yi = ctx.yi
    folds = ctx.rule_folds
    feats = {}
    for e in range(ctx.E):
        f = features(ctx.it[f'seq{e}'].tolist())
        pre = '' if ctx.E == 1 else ctx.task.elements[e] + ':'
        for k_, v in f.items():
            feats[pre + k_] = v
    ctx.feat = feats
    if not hasattr(ctx, 'rule_preds'):
        ctx.rule_preds = {}
    groups = dict(ARTEFACT_GROUPS)
    groups['shortcut.composition'] = COMPOSITION
    for cid, names in groups.items():
        if cid not in want:
            continue
        t0 = time.time()
        th = ths[cid]
        res = {}
        for nm in feats:
            base = nm.split(':')[-1]
            if base in names:
                res[nm] = rule_eval(feats[nm], yi, K, tr, te, folds, 'mcc', ctx.n_boot, ctx.seed, cv_as_test=ctx.rules_within_test)
        if not res:
            out[cid] = _skip(cid, 'no feature')
            continue
        varying = {k_: v for k_, v in res.items() if not v.get('constant')}
        if not varying:
            out[cid] = _check(cid, 'PASS', 'test MCC of the CV-selected one-feature rule', 0.0, [0.0, 0.0],
                              int(te.sum()), th, 'feature constant' if ctx.rules_within_test else 'feature constant on train',
                              dict(features={k_: 'constant' for k_ in res}),
                              [], time.time() - t0)
            continue
        def _key(k_):
            v = varying[k_]
            c = v['cv'] if v['cv'] is not None else -1
            if _use_auroc(th, ctx.minority) and v.get('cv_auroc') is not None:
                c = max(c, 2 * v['cv_auroc'] - 1)    # AUROC on the MCC scale (Youden's J of a binary feature)
            return (c, k_)
        best = max(varying, key=_key)
        r = varying[best]
        ctx.rule_preds[cid] = (best, r['pred_test'], r['cv'])
        ev = dict(selected_feature=best, cv_mcc=r['cv'], test_auroc=r.get('test_auroc'),
                  threshold=r.get('threshold'), direction=r.get('direction'),
                  features={k_: dict(cv=v.get('cv'), test=v.get('test')) for k_, v in res.items()})
        st, crit = _status_rule(r['test'], r['cv'], th, r, ctx.minority)
        st, rel = _relative(ctx, r, th, st)
        ev['cv_auroc'] = r.get('cv_auroc')
        ev['status_from'] = crit
        reason = f"rule on {best}: test MCC {r['test']:.3f} (CV on train {r['cv']:.3f})"
        if r.get('test_auroc') is not None:
            reason += f"; test AUROC {r['test_auroc']:.3f}" + (' (sets the status)' if crit == 'auroc' else '')
        if rel is not None:
            ev['published_relative'] = rel
            if rel.get('ratio') is not None:
                reason += f"; {rel['ratio']:.2f} of the best published above-chance {rel['metric']} {rel['published_best']:.3f}"
        if ctx.rules_within_test:
            reason += _within_note(ctx)
        if cid == 'shortcut.composition':
            reason += '; composition can be real biology: report it as a baseline models must beat'
        out[cid] = _check(cid, st, 'test MCC of the CV-selected one-feature rule', r['test'], r['ci'], int(te.sum()),
                          th, reason, ev, [], time.time() - t0)
    if 'shortcut.kmer' in want:
        t0 = time.time()
        th = ths['shortcut.kmer']
        if ctx.E == 1:
            r = markov_rule(ctx.it['seq0'].tolist(), yi, K, tr, te, folds, 3, 'mcc', ctx.n_boot, ctx.seed,
                                cv_as_test=ctx.rules_within_test)
        else:
            joined = (ctx.it['seq0'] + 'N' + ctx.it['seq1']).tolist()
            r = markov_rule(joined, yi, K, tr, te, folds, 3, 'mcc', ctx.n_boot, ctx.seed, cv_as_test=ctx.rules_within_test)
        if r.get('constant'):
            out['shortcut.kmer'] = _skip('shortcut.kmer', 'sequences too short')
        else:
            ctx.rule_preds['shortcut.kmer'] = ('markov3', r['pred_test'], r['cv'])
            st, crit = _status_rule(r['test'], r['cv'], th, r, ctx.minority)
            out['shortcut.kmer'] = _check('shortcut.kmer', st, 'test MCC of the order-3 Markov Bayes rule', r['test'],
                                          r['ci'], int(te.sum()), th,
                                          f"order-3 Markov rule: test MCC {r['test']:.3f} (CV {r['cv']:.3f})"
                                          + (f"; test AUROC {r['test_auroc']:.3f}" if r.get('test_auroc') is not None else '')
                                          + _within_note(ctx)
                                          + "; composition can be real biology",
                                          dict(cv_mcc=r['cv'], test_auroc=r.get('test_auroc'), cv_auroc=r.get('cv_auroc'),
                                               status_from=crit), [], time.time() - t0)
    if 'shortcut.position_motif' in want:
        t0 = time.time()
        th = ths['shortcut.position_motif']
        if ctx.E != 1:
            out['shortcut.position_motif'] = _skip('shortcut.position_motif', 'multi-element items')
        else:
            r = motif_rule(ctx.it['seq0'].tolist(), yi, K, tr, te, folds, 'mcc', ctx.n_boot, ctx.seed,
                           cv_as_test=ctx.rules_within_test)
            if 'skipped' in r:
                out['shortcut.position_motif'] = _skip('shortcut.position_motif', r['skipped'])
            else:
                ctx.rule_preds['shortcut.position_motif'] = ('motif_table', r['pred_test'], r['cv'])
                st, crit = _status_rule(r['test'], r['cv'], th, r, ctx.minority)
                guard = None
                if st == 'FAIL' and crit == 'auroc' and th.get('auroc_requires_unmatched_site'):
                    um = ctx.unmatched or {}
                    if um.get('statistic', 0.0) < ths['construction.unmatched_site']['fail']:
                        st = 'WARN'      # imbalance criterion without an unmatched site: a consensus, not a construction
                        guard = 'AUROC criterion without an unmatched site: WARN'
                st, rel = _relative(ctx, r, th, st)
                ev = dict(cv_mcc=r['cv'], test_auroc=r.get('test_auroc'), cv_auroc=r.get('cv_auroc'), status_from=crit,
                          features=r['features'])
                reason = (f"table on {', '.join(f['kmer'] + '@' + str(f['pos']) for f in r['features'])}: test MCC "
                          f"{r['test']:.3f} (CV {r['cv']:.3f})"
                          + (f"; test AUROC {r['test_auroc']:.3f}" + (' (sets the status)' if crit == 'auroc' else '')
                             if r.get('test_auroc') is not None else ''))
                if guard:
                    ev['guard'] = guard
                    reason += '; ' + guard
                if rel is not None:
                    ev['published_relative'] = rel
                    if rel.get('ratio') is not None:
                        reason += f"; {rel['ratio']:.2f} of the best published above-chance {rel['metric']} " \
                                  f"{rel['published_best']:.3f}"
                if ctx.rules_within_test:
                    reason += _within_note(ctx)
                out['shortcut.position_motif'] = _check(
                    'shortcut.position_motif', st, 'test MCC of the greedy <= 4-feature fixed-position table', r['test'],
                    r['ci'], int(te.sum()), th, reason, ev, [], time.time() - t0)
    if 'shortcut.anchored_motif' in want:
        t0 = time.time()
        cid = 'shortcut.anchored_motif'
        th = ths[cid]
        anchors = getattr(ctx, 'anchor', None)
        if ctx.E != 1:
            out[cid] = _skip(cid, 'multi-element items')
        elif anchors is None and len(np.unique(ctx.lens)) == 1:
            out[cid] = _skip(cid, 'fixed-length items (see shortcut.position_motif)')
        else:
            A = th['anchor']
            r = motif_rule(_anchored_seqs(ctx.it['seq0'].tolist(), A, anchors), yi, K, tr, te, folds, 'mcc', ctx.n_boot,
                           ctx.seed, cv_as_test=ctx.rules_within_test)
            if 'skipped' in r:
                out[cid] = _skip(cid, r['skipped'])
            else:
                ctx.rule_preds[cid] = ('anchored_table', r['pred_test'], r['cv'])
                st, crit = _status_rule(r['test'], r['cv'], th, r, ctx.minority)
                guard = None
                if st == 'FAIL' and crit == 'auroc':
                    st = 'WARN'      # no unmatched-site evidence exists for variable-length items
                    guard = 'AUROC criterion without unmatched-site evidence (variable-length items): WARN'
                st, rel = _relative(ctx, r, th, st)

                def where(pos):
                    if anchors is not None:
                        return f'{pos - A:+d} from the site'
                    return f'start+{pos}' if pos < A else f'end-{2 * A - pos}'
                feats = [dict(f, anchored=where(f['pos'])) for f in r['features']]
                ev = dict(cv_mcc=r['cv'], test_auroc=r.get('test_auroc'), cv_auroc=r.get('cv_auroc'), status_from=crit,
                          features=feats, anchor='site column' if anchors is not None else 'item ends', window=A)
                reason = (f"table on {', '.join(f['kmer'] + '@' + f['anchored'] for f in feats)}: test MCC {r['test']:.3f} "
                          f"(CV {r['cv']:.3f})"
                          + (f"; test AUROC {r['test_auroc']:.3f}" + (' (sets the status)' if crit == 'auroc' else '')
                             if r.get('test_auroc') is not None else ''))
                if guard:
                    ev['guard'] = guard
                    reason += '; ' + guard
                if rel is not None:
                    ev['published_relative'] = rel
                    if rel.get('ratio') is not None:
                        reason += f"; {rel['ratio']:.2f} of the best published above-chance {rel['metric']} " \
                                  f"{rel['published_best']:.3f}"
                if ctx.rules_within_test:
                    reason += _within_note(ctx)
                out[cid] = _check(cid, st, 'test MCC of the greedy <= 4-feature table anchored at the item ends (or site)',
                                  r['test'], r['ci'], int(te.sum()), th, reason, ev, [], time.time() - t0)
    return out


# ----------------------------------------------------------------------------------------------- position
def check_position(ctx, th):
    t0 = time.time()
    if not ctx.task.distance_cols:
        return _skip('position.distance', 'no --distance-cols given')
    if not ctx.classification or ctx.K != 2:
        return _skip('position.distance', 'binary labels only')
    from ..report import evaluate
    cols = ['dist__' + c for c in ctx.task.distance_cols]
    F = ctx.it[cols].copy()
    F.columns = ctx.task.distance_cols
    ok = np.isfinite(F.to_numpy(np.float64)).all(1)
    y = ctx.yi.astype(np.float64)
    groups = ctx.it['group'].to_numpy() if 'group' in ctx.it else np.zeros(ctx.n, np.int64)
    keep = ok & (ctx.tr | ctx.te)
    idx = np.flatnonzero(keep)
    df = evaluate(y[idx], groups[idx], F.iloc[idx].reset_index(drop=True), {}, eval_mask=ctx.te[idx],
                  train_mask=ctx.rule_tr[idx], position_columns=list(F.columns), n_boot=ctx.n_boot, seed=ctx.seed,
                  rule=False, method='cluster' if 'group' in ctx.it and len(np.unique(groups[idx])) >= 10 else 'block')
    r = df.iloc[0]
    single = {}
    for c in F.columns:
        x = F[c].to_numpy(np.float64)
        trm = keep & ctx.rule_tr
        tem = keep & ctx.te
        a_tr = S.auroc(ctx.yi[trm], x[trm])
        sgn = 1 if (a_tr or 0.5) >= 0.5 else -1
        single[c] = S.auroc(ctx.yi[tem], sgn * x[tem])
    v = float(r['auroc'])
    ci = [float(r['auroc_lo']), float(r['auroc_hi'])]
    st = TH.status(v, th)
    ev = dict(single_column_auroc=single, n_missing=int((~ok).sum()), within=float(r.get('within', np.nan)))
    return _check('position.distance', st, 'test AUROC of the position-only model (distance fields)', v, ci,
                  int((keep & ctx.te).sum()), th,
                  f"position-only model on {', '.join(F.columns)}: test AUROC {v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", ev,
                  [], time.time() - t0)


def check_row_zero(task_dir, th, n_boot, seed):
    t0 = time.time()
    from ..report import row_zero
    df = row_zero(task_dir, models=[], n_boot=n_boot, seed=seed)
    r = df.iloc[0]
    v = float(r['auroc'])
    ci = [float(r['auroc_lo']), float(r['auroc_hi'])]
    ev = dict(table=df.drop(columns=[c for c in df.columns if df[c].dtype == object and c != 'model']).to_dict('records'))
    return _check('position.row_zero', TH.status(v, th), 'AUROC of the position-only model (seqleak row zero)', v, ci,
                  int(r.get('n_items', 0)) if 'n_items' in r else None, th,
                  f"row zero (position model) AUROC {v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", ev, [], time.time() - t0)


# ----------------------------------------------------------------------------------------------- lookup
def lookup(ctx, th, metric):
    t0 = time.time()
    te = np.flatnonzero(ctx.te)
    if ctx.test_only:
        ctx.lookup_pred = None
        return _check('baseline.lookup', 'INFO', 'nearest-training lookup', None, None, len(te), None,
                      'test-only task: no training split to look up', dict(metric=metric), [], time.time() - t0), None
    if not ctx.classification:
        d = ctx.near[0]
        ff = _first_fit_item_of_uid(ctx, 0)
        pv = np.array([ctx.yv[int(ff[u])] if u in ff.index else np.nan for u in d['partner_uid']], float)
        ok = np.isfinite(pv)
        from scipy.stats import pearsonr
        r = float(pearsonr(pv[ok], ctx.yv[te][ok].astype(float))[0]) if ok.sum() > 2 else None
        return _check('baseline.lookup', 'INFO', 'Pearson r of the nearest training label', r, None, int(ok.sum()),
                      None, 'regression lookup', {}, [], time.time() - t0), None
    K = ctx.K
    maj = int(np.argmax(np.bincount(ctx.yi[ctx.fit], minlength=K)))
    prev = np.bincount(ctx.yi[ctx.fit], minlength=K) / max(ctx.fit.sum(), 1)
    pred = np.full(len(te), maj, np.int64)
    soft = np.full(len(te), prev[-1] if K == 2 else 0.0)
    best_c = np.zeros(len(te))
    if ctx.E == 1:
        uid = ctx.uid[0]
        nu = len(ctx.useq[0])
        fit_hist = np.zeros((nu, K), np.int64)
        np.add.at(fit_hist, (uid[ctx.fit], ctx.yi[ctx.fit]), 1)
        h = ctx.pairs[0]
        d = ctx.near[0]
        # tied best partners per query
        if len(h.q):
            best_shared = np.zeros(len(d['q_u']), np.int64)
            np.maximum.at(best_shared, h.q, h.shared)
            tie = h.shared == best_shared[h.q]
            hist_q = np.zeros((len(d['q_u']), K), np.int64)
            np.add.at(hist_q, h.q[tie], fit_hist[h.r_uid[tie]])
            qi = d['qpos'][uid[te]]
            hq = hist_q[qi]
            okp = d['shared'] >= th['min_shared']
            pred = np.where(okp & (hq.sum(1) > 0), hq.argmax(1), maj)
            best_c = np.where(okp, d['c'], 0.0)
            if K == 2:
                fp = hq[:, 1] / np.maximum(hq.sum(1), 1)
                soft = np.where(okp & (hq.sum(1) > 0), best_c * fp + (1 - best_c) * prev[1], prev[1])
    else:
        A, B, C = [], [], []
        for e in range(ctx.E):
            a, b, c = _item_pairs_from_hits(ctx, e, 0.0)
            A.append(a)
            B.append(b)
            C.append(c)
        a = np.concatenate(A)
        b = np.concatenate(B)
        c = np.concatenate(C) / ctx.E
        if len(a):
            key = a * ctx.n + b
            u, inv = np.unique(key, return_inverse=True)
            tot = np.bincount(inv, weights=c)
            ua, ub = u // ctx.n, u % ctx.n
            o = np.lexsort((ub, -tot, ua))
            first = o[np.flatnonzero(np.r_[True, ua[o][1:] != ua[o][:-1]])]
            pos = pd.Series(np.arange(len(te)), index=te)
            rows = pos[ua[first]].to_numpy()
            pred[rows] = ctx.yi[ub[first]]
            best_c[rows] = tot[first]
            if K == 2:
                soft[rows] = tot[first] * ctx.yi[ub[first]] + (1 - tot[first]) * prev[1]
    yt = ctx.yi[te]
    if metric == 'auroc' and K == 2:
        v = S.auroc((yt == 1).astype(int), soft)
        ci = S.auroc_ci((yt == 1).astype(int), soft, ctx.n_boot, ctx.seed)
        chance = 0.5
    else:
        metric = 'mcc' if metric == 'auroc' else metric
        v = S.score(yt, pred, K, metric)
        ci = S.bootstrap_ci(yt, pred, K, metric, ctx.n_boot, ctx.seed)
        chance = S.score(yt, np.full(len(te), maj), K, metric)
    ev = dict(metric=metric, chance=chance, n_with_partner=int((best_c > 0).sum()),
              auroc=S.auroc((yt == K - 1).astype(int), soft) if K == 2 else None,
              mcc=S.score(yt, pred, K, 'mcc'), acc=S.score(yt, pred, K, 'acc'))
    ctx.lookup_pred = pred
    ctx.lookup_c = best_c
    return _check('baseline.lookup', 'INFO', f'{metric} of the nearest-training lookup on test', v, ci, len(te), None,
                  f"nearest-training lookup: {metric} {v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (chance {chance:.3f})", ev,
                  [], time.time() - t0), pred


def best_rule(ctx, metric, published_best, near_thr):
    if not ctx.classification:
        return None
    te = np.flatnonzero(ctx.te)
    yt = ctx.yi[te]
    K = ctx.K
    maj = int(np.argmax(np.bincount(ctx.yi[ctx.fit], minlength=K)))
    fams = {}
    rp = getattr(ctx, 'rule_preds', {})
    for cid, (nm, p, cv) in rp.items():
        fams[f'{cid}:{nm}'] = p
    lp = getattr(ctx, 'lookup_pred', None)
    if lp is not None:
        fams['lookup'] = lp
        # fallback: the shortcut rule with the best CV score
        if rp:
            fb_id = max(rp, key=lambda c: (rp[c][2] if rp[c][2] is not None else -1, c))
            fb = rp[fb_id][1]
            near = ctx.lookup_c >= near_thr
            fams[f'lookup_if_near_else:{fb_id}'] = np.where(near, lp, fb)
            for cid, tp in getattr(ctx, 'twin_pred', {}).items():
                t = tp[te]
                fams[f'twin:{cid}_else:{fb_id}'] = np.where(t >= 0, t, fb)
    for cid, tp in getattr(ctx, 'twin_pred', {}).items():
        t = tp[te]
        if f'twin:{cid}' not in fams:
            fams[f'twin:{cid}_else_majority'] = np.where(t >= 0, t, maj)
    if not fams:
        return None
    if metric == 'auroc' and K == 2:
        yb = (yt == 1).astype(int)
        scores = {k: S.auroc(yb, p) for k, p in fams.items()}
        scores = {k: (v if v is not None else 0.5) for k, v in scores.items()}
        bk = max(scores, key=lambda k: (scores[k], k))
        ci = S.auroc_ci(yb, fams[bk], ctx.n_boot, ctx.seed)
        chance = 0.5
    else:
        metric = 'mcc' if metric == 'auroc' else metric
        scores = {k: S.score(yt, p, K, metric) for k, p in fams.items()}
        bk = max(scores, key=lambda k: (scores[k], k))
        ci = S.bootstrap_ci(yt, fams[bk], K, metric, ctx.n_boot, ctx.seed)
        chance = S.score(yt, np.full(len(te), maj), K, metric)
    out = dict(family=bk, metric=metric, value=scores[bk], ci=ci, chance=chance, selected_on='test',
               all={k: round(v, 4) for k, v in sorted(scores.items(), key=lambda z: -z[1])})
    if published_best is not None:
        out['published_best'] = float(published_best)
        den = float(published_best) - chance
        out['ratio'] = float((scores[bk] - chance) / den) if den > 0 else None
    return out


# ----------------------------------------------------------------------------------------------- main
def run_audit(task, references=(), checks=None, skip=None, thresholds=None, k=21, sketch_scale='auto', max_df=64,
              n_boot=1000, folds=5, seed=0, threads=1, examples=5, homology='auto', homology_max_test=2000,
              blast_bin=None, mmseqs_bin=None, metric='auto', published_best=None, task_dir=None, workdir=None,
              quiet=False, argv=None, version='', genome=None, genome_registry=None, genome_method='kmer',
              minimap2_bin=None):
    t_start = time.time()
    th = thresholds or TH.load()
    metric = 'mcc' if metric == 'auto' else metric
    want = set(ALL_CHECKS)
    if checks:
        sel = set()
        for c in checks:
            sel |= {x for x in ALL_CHECKS if x == c or x.split('.')[0] == c}
        want = sel | {'baseline.lookup'}
    if skip:
        for c in skip:
            want -= {x for x in ALL_CHECKS if x == c or x.split('.')[0] == c}
    ctx = Ctx(task, k, sketch_scale, max_df, seed, n_boot, folds, examples, quiet)
    lo_th = th.get('shortcut.lowinfo', {}).get('min_acgt', 0.5)
    ctx.lowinfo = ctx.acgt < lo_th
    ctx.published_best = published_best
    ctx.metric = metric
    ctx.anchor = task.items['anchor'].to_numpy() if 'anchor' in task.items else None
    ctx.genome_memb = None
    ctx.rule_preds = {}
    res = {}
    _log(f'task {task.name}: {ctx.n} items (train {(ctx.split == "train").sum()}, dev {ctx.dv.sum()}, test {ctx.te.sum()}), '
         f'{ctx.E} element(s), {ctx.K} classes' + ('; test-only task' if ctx.test_only else ''), quiet)
    if ctx.test_only:
        ctx.exact_dup = np.zeros(ctx.n, bool)
        for c in ('overlap.exact', 'overlap.near', 'overlap.windows', 'overlap.pair', 'overlap.homology'):
            if c in want:
                res[c] = _skip(c, 'test-only task: no training split')
    else:
        r = check_exact(ctx, th['overlap.exact'])   # also sets per-item exact-copy flags used later
        if 'overlap.exact' in want:
            res['overlap.exact'] = r
        run_engine(ctx, max_df, quiet)
        r = check_near(ctx, th['overlap.near'])   # also sets per-item near flags used later
        if 'overlap.near' in want:
            res['overlap.near'] = r
        if 'overlap.windows' in want:
            res['overlap.windows'] = check_windows(ctx, th['overlap.windows'])
        if 'overlap.pair' in want:
            res['overlap.pair'] = check_pair(ctx, th['overlap.pair'])
        if 'overlap.homology' in want:
            from .homology import check_homology
            res['overlap.homology'] = check_homology(ctx, th['overlap.homology'], homology, homology_max_test, threads,
                                                     blast_bin, mmseqs_bin, workdir, _check, _skip)
    if 'reference.copies' in want:
        res['reference.copies'] = check_reference(ctx, list(references), th['reference.copies'], max_df)
    _log('overlap checks done', quiet)
    # twins (also define the clusters that group the CV folds)
    tw, cp, gp = check_twins(ctx, th['construction.shuffle'], th['construction.segment'],
                             'construction.shuffle' in want, 'construction.segment' in want)
    res.update(tw)
    # CV folds on train: grouped by exact-duplicate and twin clusters
    nu = len(ctx.useq[0])
    ua, ub = [], []
    for p in (cp, gp):
        if p is not None and len(p):
            ua.append(p[:, 0])
            ub.append(p[:, 1])
    if ctx.E == 1:
        # unique-sequence graph: twins + reverse-complement identity (canonical ids)
        can_u = np.zeros(nu, np.int64)
        can_u[ctx.uid[0]] = ctx.cid[0]
        comp_u = components(nu + int(can_u.max()) + 1,
                            np.concatenate(ua + [np.arange(nu)]),
                            np.concatenate(ub + [nu + can_u]))[:nu]
        group = comp_u[ctx.uid[0]]
    else:
        group = ctx.key
    ctx.group = group
    trn = np.flatnonzero(ctx.tr)
    ctx.folds = S.group_folds(group[trn], folds, seed)
    if ctx.single_class_train:
        ctx.rule_tr = ctx.te.copy()
        ctx.rule_folds = S.group_folds(group[np.flatnonzero(ctx.te)], folds, seed)
    else:
        ctx.rule_tr, ctx.rule_folds = ctx.tr, ctx.folds
    _log('twin checks done', quiet)
    if 'construction.synthetic_negatives' in want:
        res['construction.synthetic_negatives'] = check_synthetic_negatives(ctx, cp, gp, th['construction.synthetic_negatives'])
    # the unmatched-site statistic also guards the motif check's imbalance criterion, so it is computed first
    r = check_unmatched_site(ctx, th['construction.unmatched_site'])
    if 'construction.unmatched_site' in want:
        res['construction.unmatched_site'] = r
    if 'construction.library' in want:
        res['construction.library'] = check_library(ctx, th['construction.library'])
    if 'construction.genome_membership' in want:
        res['construction.genome_membership'] = check_genome_membership(
            ctx, th['construction.genome_membership'], genome, genome_registry, genome_method, minimap2_bin, threads,
            workdir)
        _log('genome membership done', quiet)
    if 'shortcut.lowinfo' in want:
        res['shortcut.lowinfo'] = check_lowinfo(ctx, th['shortcut.lowinfo'])
    sc_want = [c for c in ['shortcut.length', 'shortcut.ambiguous', 'shortcut.case', 'shortcut.composition',
                           'shortcut.kmer', 'shortcut.position_motif', 'shortcut.anchored_motif'] if c in want]
    if sc_want:
        res.update(check_shortcuts(ctx, sc_want, th))
    _log('shortcut checks done', quiet)
    if 'position.distance' in want:
        res['position.distance'] = check_position(ctx, th['position.distance'])
    if 'position.row_zero' in want:
        res['position.row_zero'] = check_row_zero(task_dir, th['position.row_zero'], n_boot, seed) if task_dir \
            else _skip('position.row_zero', 'no --task-dir given')
    lk, _ = lookup(ctx, th['baseline.lookup'], metric)
    res['baseline.lookup'] = lk
    br = best_rule(ctx, metric, published_best, th['overlap.near']['near'])
    order = [c for c in ALL_CHECKS if c in res]
    res = {c: res[c] for c in order}
    fails = [c for c, r in res.items() if r['status'] == 'FAIL']
    warns = [c for c, r in res.items() if r['status'] == 'WARN']
    verdict = 'FAIL' if fails else ('WARN' if warns else 'PASS')
    summ = []
    if fails:
        summ.append('leak checks failing: ' + ', '.join(fails))
    if warns:
        summ.append('warnings: ' + ', '.join(warns))
    if not summ:
        summ.append('no check reached its warning threshold')
    it = task.items
    rep = dict(
        schema=SCHEMA, tool=dict(name='seqleak', version=version, argv=argv),
        task=dict(name=task.name, inputs=task.inputs, kind='sequence' if ctx.E == 1 else 'multi_element',
                  elements=task.elements, task_type=task.task_type,
                  classes=[_jsonable(c) for c in ctx.classes],
                  n=dict(train=int((ctx.split == 'train').sum()), dev=int(ctx.dv.sum()), test=int(ctx.te.sum())),
                  label_counts={sp: {str(k_): int(v) for k_, v in it.loc[it['split'] == sp, 'label'].value_counts()
                                     .sort_index().items()} for sp in ('train', 'dev', 'test') if (it['split'] == sp).any()}
                  if ctx.classification else None,
                  length=dict(min=int(ctx.lens.min()), median=float(np.median(ctx.lens)), max=int(ctx.lens.max())),
                  metric=metric, has_coords=bool(task.has_coords), has_distance=bool(task.distance_cols),
                  mode='test_only' if ctx.test_only else 'standard', single_class_train=bool(ctx.single_class_train)),
        settings=dict(rules=TH.RULES_VERSION, thresholds=th, k=k, sketch_scale=ctx.scale, max_df=max_df, seed=seed,
                      n_boot=n_boot, genome=list(genome) if genome else None, genome_method=genome_method,
                      folds=folds, homology=homology, homology_max_test=homology_max_test,
                      fold_groups='exact-duplicate and twin clusters of unique sequences'),
        checks=res, lookup_baseline=dict(lk['evidence'], metric=metric, value=lk['value'], ci=lk['ci']),
        best_rule=br,
        verdict=dict(status=verdict, fail=fails, warn=warns, summary='; '.join(summ)),
        runtime=dict(seconds=round(time.time() - t_start, 2),
                     peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)))
    # per-item table
    items = pd.DataFrame(dict(item_id=ctx.ids, split=ctx.split, label=[str(v) for v in ctx.yv]))
    cvf = np.full(ctx.n, -1, np.int64)
    cvf[trn] = ctx.folds
    items['cv_fold'] = cvf
    items['group'] = ctx.group
    items['exact_dup'] = ctx.exact_dup
    te = np.flatnonzero(ctx.te)
    items['lowinfo'] = ctx.lowinfo
    if ctx.genome_memb is not None:
        items['genome_membership'] = ctx.genome_memb
    if ctx.near:
        for nm in ('c', 'coll'):
            col = np.full(ctx.n, np.nan)
            col[te] = np.max(np.column_stack([ctx.near[e][nm] for e in range(ctx.E)]), 1)
            items['containment' if nm == 'c' else 'collinear'] = col
        pu = np.full(ctx.n, '', object)
        ff = _first_fit_item_of_uid(ctx, 0)
        pid = ctx.near[0]['partner_uid']
        pu[te] = [str(ctx.ids[int(ff[u])]) if u >= 0 and u in ff.index else '' for u in pid]
        items['partner_item'] = pu
    if getattr(ctx, 'lookup_pred', None) is not None:
        lp = np.full(ctx.n, -1, np.int64)
        lp[te] = ctx.lookup_pred
        items['lookup_pred'] = [str(ctx.classes[v]) if v >= 0 else '' for v in lp]
    if ctx.feat:
        for k_, v in ctx.feat.items():
            items['f_' + k_] = v
    return rep, items
