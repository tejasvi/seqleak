"""The standard table: a position-only baseline as row zero, then every model with its raw metric, its metric with
position held fixed, its gain over the position model and its position share, all with 95% intervals and n."""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from . import metrics as M
from .baseline import make_folds, position_model
from .bootstrap import Resample, factorize_sorted
from .features import rule_score
from .strata import gene_scheme, make_strata, prognostic_strata, resolve_scheme
from .tasks import load_task

ROW_ZERO = 'position model (row zero)'
CACHE_VERSION = '0.2'   # out-of-fold fits are unchanged in 0.3   # bump when cached out-of-fold predictions change meaning (spec defaults, solvers)
RULE = 'distance rule'
ALIASES = {'k': 'genes_apart', 'same': 'same_strand', 'maxgap': 'max_intervening_gap', 'nsw': 'strand_switches',
           'n_switch': 'strand_switches', 'gene_len': 'length', 'ori_dist': 'dist_origin', 'leading': 'replication_strand'}


def _auto(n_groups, method, folds):
    if method == 'auto':
        method = 'cluster' if n_groups >= 10 else 'block'
    if isinstance(folds, str) and folds == 'auto':
        folds = 'contig5' if n_groups == 1 else 'logo' if n_groups <= 10 else 5
    return method, folds


def content_hash(a):
    """Fingerprint of an array's values (float64)."""
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()


class _Cache:
    def __init__(self, path, tag):
        self.path, self.tag = path, tag
        if path:
            os.makedirs(path, exist_ok=True)

    def get(self, name, fn, n, content=None):
        """Cached array for ``name``; ``content`` (e.g. the model score) is part of the key, so a changed score
        never reuses a stale fit."""
        if not self.path:
            return fn()
        key = f'{self.tag}|{name}'
        if content is not None:
            key += '|' + content_hash(content)
        h = hashlib.sha1(key.encode()).hexdigest()[:12]
        f = os.path.join(self.path, f'{name}.{h}.npy')
        if os.path.exists(f):
            a = np.load(f)
            if len(a) == n:
                return a
        a = fn()
        np.save(f, a)
        return a


def _combine(a, b):
    """One code per distinct (a, b) pair."""
    return pd.factorize(pd.MultiIndex.from_arrays([np.asarray(a), np.asarray(b)]))[0].astype(np.int64)


def _row(name, res):
    """Flatten MetricResults of one scorer into a dict."""
    r = dict(scorer=name)
    for key, m in res.items():
        if m is None:
            continue
        r[key] = m.estimate
        r[f'{key}_lo'], r[f'{key}_hi'] = m.ci
        if key == 'auroc':
            r['n_items'], r['n_pos'], r['n_genomes'] = m.n_items, m.n_pos, m.n_groups
        if key.startswith('gain'):
            r[f'{key}_p'] = m.info.get('p_wilcoxon', np.nan)
            r[f'{key}_genomes_up'] = m.info.get('groups_improved', np.nan)
        if key == 'share':
            r['share_flags'] = m.info.get('flags', '')
            r['share_upper'] = m.info.get('share_upper', np.nan)
            r['share_upper_lo'] = m.info.get('share_upper_lo', np.nan)
            r['share_upper_hi'] = m.info.get('share_upper_hi', np.nan)
    return r


def evaluate(y, groups, features, scores, *, strata=None, scheme='auto', spec='auto', folds='auto', seed=0,
             n_boot=2000, method='auto', block_size=50, order=None, aggregate='mean', chance=0.5, min_margin=0.02,
             interact='auto', train_mask=None, eval_mask=None, rule=True, position_columns=None, n_bins=20,
             cache_dir=None, cache_tag='', level=0.95, verbose=False):
    """Row-zero table for in-memory data. ``scores``: dict model -> array aligned with ``y``.

    ``eval_mask`` selects the evaluated items (default all). ``train_mask`` (optional) selects the items the position
    models are fitted on; when it is disjoint from ``eval_mask`` the models are fitted once on it (benchmark
    train/test split), otherwise by genome-grouped cross-validation (``folds``) restricted to it.
    ``position_columns``: use these columns as the position features (any names; numeric columns get splines,
    columns with <= 10 values indicators) with prognostic strata (``n_bins`` quantile bins of the out-of-fold
    position-model prediction within each genome) unless ``spec`` / ``scheme`` say otherwise. ``scheme='prognostic'``
    works with any features. When the cross-validation folds split genomes (single-genome tasks), every metric of the
    out-of-fold predictions (row zero, gains) compares items only within their fold (``df.attrs['fold_stratified']``)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    n = len(y)
    eval_mask = np.ones(n, bool) if eval_mask is None else np.asarray(eval_mask, bool)
    if position_columns is not None:
        features = features[list(position_columns)]
        spec = 'generic' if isinstance(spec, str) and spec == 'auto' else spec
        scheme = 'prognostic' if isinstance(scheme, str) and scheme == 'auto' else scheme
    scheme = resolve_scheme(features, scheme)
    if isinstance(scheme, str) and scheme == 'gene' and not gene_scheme(features):
        raise ValueError('no known position features in the table: pass position_columns=[...] (see seqleak.report.evaluate)')
    prognostic = isinstance(scheme, str) and scheme == 'prognostic'
    if strata is None and not prognostic:
        strata = make_strata(features, scheme)
    n_groups = len(np.unique(groups[eval_mask]))
    method, folds = _auto(n_groups, method, folds)
    is_pair = 'same_strand' in features and 'gap' in features
    if interact == 'auto':
        interact = is_pair and scheme in ('adjacent', 'operon_v1')
    fixed_split = train_mask is not None and not np.any(np.asarray(train_mask, bool) & eval_mask)
    tag = json.dumps(dict(t=cache_tag, n=n, y=float(np.nansum(np.asarray(y, np.float64))), spec=str(spec), folds=str(folds),
                          seed=seed, fixed=fixed_split, v=CACHE_VERSION), sort_keys=True)
    cache = _Cache(cache_dir, tag)

    def fit(extra=None):
        return position_model(features, y, groups, spec=spec, extra=extra, folds=None if fixed_split else folds,
                              seed=seed, train_mask=train_mask, order=order, verbose=verbose)

    ev = np.flatnonzero(eval_mask)
    ye, ge = y[ev], groups[ev]
    ordr = None
    if method == 'block' and order is not None:
        rank = np.empty(n, np.int64)
        rank[np.asarray(order)] = np.arange(n)       # position of every item in genome order
        ordr = np.argsort(rank[ev], kind='stable')   # evaluated items in genome order
    rs = Resample(ge, method=method, n_boot=n_boot, block_size=block_size, seed=seed, order=ordr) if n_boot else None
    kw = dict(resample=rs, level=level)
    pos = cache.get('position', fit, n)
    # Out-of-fold predictions of different fold models are compared only within their fold when the folds split
    # genomes (single-genome tasks): pooling them across folds biases the AUROC down (a fold rich in positives was
    # trained on fewer), most for coarse predictors such as the position model, which would inflate the gain.
    fold_ev = None
    if not fixed_split:
        f_all = make_folds(groups, folds, seed, order)[ev]
        if pd.Series(f_all).groupby(ge).nunique().max() > 1:
            fold_ev = f_all
    bins_by = ge if fold_ev is None else _combine(ge, fold_ev)
    ke = prognostic_strata(pos[ev], bins_by, n_bins) if prognostic else np.asarray(strata)[ev]
    ke_f = ke if fold_ev is None else _combine(ke, fold_ev)
    rows = []
    res0 = dict(auroc=M.within_auc(ye, pos[ev], fold_ev, ge, aggregate=aggregate, **kw),
                auprc=M.auprc(ye, pos[ev], ge, fold=fold_ev, **kw),
                within=M.within_auc(ye, pos[ev], ke_f, ge, aggregate=aggregate, **kw),
                share=M.position_share_bound(ye, pos[ev], ke, ge, aggregate=aggregate, fold=fold_ev, min_margin=min_margin, **kw))
    rows.append(_row(ROW_ZERO, res0))
    if rule and is_pair:
        rs_score = rule_score(features['same_strand'].to_numpy(), features['gap'].to_numpy())
        if 'genes_apart' in features:
            rs_score = rs_score - 1e-6 * features['genes_apart'].to_numpy()
        rr = dict(auroc=M.within_auc(ye, rs_score[ev], None, ge, aggregate=aggregate, **kw),
                  auprc=M.auprc(ye, rs_score[ev], ge, **kw),
                  within=M.within_auc(ye, rs_score[ev], ke, ge, aggregate=aggregate, **kw),
                  share=M.position_share_bound(ye, rs_score[ev], ke, ge, aggregate=aggregate, min_margin=min_margin, **kw))
        rows.append(_row(RULE, rr))
    same = features['same_strand'].to_numpy().astype(np.float64) if is_pair else None
    for m, s in scores.items():
        s = np.asarray(s, np.float64)
        if verbose:
            print(f'[seqleak] {m}', flush=True)
        res = dict(auroc=M.within_auc(ye, s[ev], None, ge, aggregate=aggregate, **kw),
                   auprc=M.auprc(ye, s[ev], ge, **kw),
                   within=M.within_auc(ye, s[ev], ke, ge, aggregate=aggregate, **kw))
        extra = np.column_stack([s, s * same]) if interact and same is not None else s
        pe = cache.get(f'position_plus_{m}', lambda: fit(extra), n, content=s)
        res['gain'] = M.gain_over_position(ye, pos[ev], pe[ev], ge, metric='auroc', aggregate=aggregate, fold=fold_ev, **kw)
        res['gain_auprc'] = M.gain_over_position(ye, pos[ev], pe[ev], ge, metric='auprc', fold=fold_ev, **kw)
        res['share'] = M.position_share_bound(ye, s[ev], ke, ge, aggregate=aggregate, min_margin=min_margin, **kw)
        rows.append(_row(m, res))
    df = pd.DataFrame(rows)
    df.attrs.update(dict(scheme=str(scheme), spec=str(spec), folds=str(folds), method=method, n_boot=n_boot,
                         aggregate=aggregate, interact=bool(interact), fixed_split=bool(fixed_split), level=level,
                         n_strata=int(len(np.unique(ke))), fold_stratified=fold_ev is not None))
    return df


def row_zero(task_dir, models=None, *, query=None, by=None, cache_dir=None, verbose=False, **kw):
    """Row-zero table for a task directory (README, "Task format"); ``models`` defaults to every ``scores/*.npy``.

    Evaluation follows the task's split when it has one (``split == 'test'`` evaluated; position models fitted on
    ``split == 'train'`` when the model scores cover the training items, else cross-validated within the test
    genomes); otherwise every item is evaluated with genome-grouped cross-validation (5 folds for > 10 genomes,
    leave-one-genome-out for 2-10, 5 contiguous chunks of the genome for a single genome).
    Column aliases are applied (``k`` -> genes_apart, ``same`` -> same_strand, ``maxgap`` -> max_intervening_gap,
    ``nsw`` / ``n_switch`` -> strand_switches, ``gene_len`` -> length, ``ori_dist`` -> dist_origin,
    ``leading`` -> replication_strand). ``query``: a pandas query selecting items (e.g. one dataset of a multi-dataset
    table); ``by``: column(s) evaluated separately (one table, ``by`` columns first). Other keyword arguments go to
    :func:`evaluate`. Returns a DataFrame (one row per scorer; ``df.attrs`` holds the settings)."""
    task = load_task(task_dir)
    it = task.items
    models = task.models if models is None else list(models)
    allscores = {m: task.scores(m) for m in models}
    order_all = task.order()
    rank = np.empty(len(it), np.int64)
    rank[order_all] = np.arange(len(it))
    keep = np.ones(len(it), bool) if query is None else it.eval(query).to_numpy(bool)
    by = [by] if isinstance(by, str) else (list(by) if by is not None else [])
    if by:
        keys = list(it.loc[keep, by].drop_duplicates().itertuples(index=False, name=None))
        parts = [(key, keep & np.logical_and.reduce([it[c].to_numpy() == v for c, v in zip(by, key)])) for key in keys]
    else:
        parts = [((), keep)]
    out = []
    for key, mask in parts:
        idx = np.flatnonzero(mask)
        sub = it.iloc[idx].reset_index(drop=True)
        sub = sub.rename(columns={a: b for a, b in ALIASES.items() if a in sub and b not in sub})
        y = sub['label'].to_numpy()
        groups = sub['genome_id'].to_numpy()
        feats = sub.drop(columns=[c for c in ('label', 'split') if c in sub])
        scores = {m: np.asarray(s)[idx] for m, s in allscores.items()}
        eval_mask = train_mask = None
        if 'split' in sub and (sub['split'] == 'test').any():
            eval_mask = (sub['split'] == 'test').to_numpy()
            tr = (sub['split'] == 'train').to_numpy()
            covered = bool(tr.any() and scores and all(np.isfinite(np.asarray(s, np.float64)[tr]).mean() > 0.5
                                                       for s in scores.values()))
            train_mask = tr if covered else eval_mask
        order = np.argsort(rank[idx], kind='stable')
        name = task.task_id + ''.join(f'_{v}' for v in key)
        cdir = os.path.join(cache_dir, name) if cache_dir else None
        df = evaluate(y, groups, feats, scores, eval_mask=eval_mask, train_mask=train_mask, order=order, cache_dir=cdir,
                      cache_tag=f'{task.task_id}|{query}|{key}', verbose=verbose, **kw)
        for pos_, (c, v) in enumerate(zip(by, key)):
            df.insert(pos_, c, v)
        out.append(df)
    res = pd.concat(out, ignore_index=True) if len(out) > 1 else out[0]
    res.attrs = dict(out[0].attrs)
    res.attrs['task'] = task.task_id
    if query is not None:
        res.attrs['query'] = query
    return res


def format_table(df, digits=3):
    """Human-readable table: estimates with [95% CI]."""
    def f(r, k, d=digits, sign=False):
        v = r.get(k, np.nan)
        if v is None or not np.isfinite(v):
            return ''
        lo, hi = r.get(f'{k}_lo', np.nan), r.get(f'{k}_hi', np.nan)
        fmt = f'{{:+.{d}f}}' if sign else f'{{:.{d}f}}'
        s = fmt.format(v)
        if np.isfinite(lo) or np.isfinite(hi):
            s += ' [' + fmt.format(lo) + ', ' + fmt.format(hi) + ']'
        return s
    out = []
    for _, r in df.iterrows():
        r = r.to_dict()
        lead = {c: r[c] for c in df.columns[:list(df.columns).index('scorer')]}
        out.append(dict(**lead, scorer=r['scorer'], n=int(r.get('n_items', 0)), genomes=int(r.get('n_genomes', 0)),
                        AUROC=f(r, 'auroc'), AUPRC=f(r, 'auprc'), within=f(r, 'within'),
                        gain=f(r, 'gain', digits + 1, True),
                        p=('' if not np.isfinite(r.get('gain_p', np.nan)) else f"{r['gain_p']:.1e}"),
                        share=f(r, 'share', 2), upper=f(r, 'share_upper', 2),
                        flags=r.get('share_flags', '') if isinstance(r.get('share_flags'), str) else ''))
    return pd.DataFrame(out).to_string(index=False)
