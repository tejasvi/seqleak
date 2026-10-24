"""Label-correlated shortcuts as cross-validated no-model rules: per-sequence artefact and composition features,
a class-conditional order-3 Markov rule (k-mer composition) and a greedy fixed-position motif table."""
import numpy as np

from . import stats as S
from .seqs import background, chunk_ranges, concat, kmer_counts, runs_max, seq_ids

ARTEFACT_GROUPS = {
    'shortcut.length': ['length'],
    'shortcut.ambiguous': ['n_N', 'frac_N', 'lead_N', 'trail_N', 'maxrun_N', 'n_other'],
    'shortcut.case': ['frac_lower'],
}
COMPOSITION = ['gc', 'cpg_oe', 'cpg_dens', 'natural']


def features(seqs):
    """Per-sequence features: length, N/ambiguity/padding, lowercase, GC, CpG and Markov naturalness."""
    parts = [_features(seqs[a:b]) for a, b in chunk_ranges(seqs)]
    if not parts:
        return {}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _features(seqs):
    codes, raw, off = concat(seqs)
    n = len(seqs)
    lens = np.diff(off).astype(np.float64)
    sid = seq_ids(off)
    isN = (raw == ord('N')) | (raw == ord('n'))
    islow = (raw >= 97) & (raw <= 122)
    other = (codes == 4) & ~isN
    nN = np.bincount(sid, weights=isN, minlength=n)
    nlow = np.bincount(sid, weights=islow, minlength=n)
    noth = np.bincount(sid, weights=other, minlength=n)
    c1 = np.zeros((n, 4))
    for b in range(4):
        c1[:, b] = np.bincount(sid, weights=(codes == b), minlength=n)
    acgt = c1.sum(1)
    m = len(codes) - 1
    if m > 0:
        valid = (codes[:-1] < 4) & (codes[1:] < 4) & (sid[:-1] == sid[1:])
        cg = np.bincount(sid[:-1], weights=valid & (codes[:-1] == 1) & (codes[1:] == 2), minlength=n)
    else:
        cg = np.zeros(n)
    gc = (c1[:, 1] + c1[:, 2]) / np.maximum(acgt, 1)
    cpg_oe = cg * np.maximum(acgt, 1) / np.maximum(c1[:, 1] * c1[:, 2], 1)
    cpg_dens = cg / np.maximum(acgt - 1, 1)
    lead = np.array([len(s) - len(s.lstrip('Nn')) for s in seqs], np.float64)
    trail = np.array([len(s) - len(s.rstrip('Nn')) for s in seqs], np.float64)
    m5, m1 = background()
    natural = m5.score(codes, off) - m1.score(codes, off)
    return dict(length=lens, n_N=nN, frac_N=nN / np.maximum(lens, 1), lead_N=lead, trail_N=trail,
                maxrun_N=runs_max(isN, off).astype(np.float64), n_other=noth, frac_lower=nlow / np.maximum(lens, 1),
                gc=gc, cpg_oe=cpg_oe, cpg_dens=cpg_dens, natural=natural)


def rule_eval(x, yi, K, tr, te, folds, metric='mcc', n_boot=1000, seed=0, cv_as_test=False):
    """One-feature rule: CV on train (folds = fold id per train item), refit on train, scored on test.
    ``cv_as_test`` (test-only tasks, where train and test are the same items): the test score is the CV score."""
    x = np.nan_to_num(np.asarray(x, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    xtr, ytr = x[tr], yi[tr]
    out = dict()
    if np.std(xtr) == 0:
        return dict(constant=True, cv=0.0, test=0.0, ci=[0.0, 0.0])

    def fit_predict(xa, ya, xb):
        if K == 2:
            t, d, _ = S.best_threshold(xa, ya)
            return S.apply_threshold(xb, t, d), (t, d)
        edges, lab = S.fit_bins(xa, ya, K)
        return S.apply_bins(xb, edges, lab), None

    cvp = np.full(len(xtr), -1)
    cvs = np.zeros(len(xtr))
    for f in np.unique(folds):
        a = folds != f
        if len(np.unique(ytr[a])) < 2:
            continue
        cvp[folds == f], par_f = fit_predict(xtr[a], ytr[a], xtr[folds == f])
        if par_f is not None:
            cvs[folds == f] = par_f[1] * xtr[folds == f]
    out['cv'] = S.score(ytr, cvp, K, metric)
    if K == 2:
        out['cv_auroc'] = S.auroc(ytr, cvs)
    pte, par = fit_predict(xtr, ytr, x[te])
    if cv_as_test:
        pte = cvp
    out['test'] = S.score(yi[te], pte, K, metric)
    out['ci'] = S.bootstrap_ci(yi[te], pte, K, metric, n_boot, seed)
    if K == 2 and par is not None:
        out['threshold'], out['direction'] = float(par[0]), int(par[1])
        out['test_auroc'] = out['cv_auroc'] if cv_as_test else S.auroc(yi[te], par[1] * x[te])
    out['pred_test'] = pte
    return out


def markov_rule(seqs, yi, K, tr, te, folds, order=3, metric='mcc', n_boot=1000, seed=0, cv_as_test=False):
    """Class-conditional order-``order`` Markov chains (add-one smoothing) fitted on train; Bayes rule with class
    priors. CV on train with the given folds; test score of the chains fitted on all of train."""
    k = order + 1
    M = 4 ** k
    C = np.concatenate([kmer_counts_of(seqs[a:b], k) for a, b in chunk_ranges(seqs)]).astype(np.float32)
    if C.sum() == 0:
        return dict(constant=True, cv=0.0, test=0.0, ci=[0.0, 0.0])
    trn = np.flatnonzero(tr)
    tei = np.flatnonzero(te)
    F = int(folds.max()) + 1 if len(folds) else 1
    cnt = np.zeros((F, K, M))
    for f in range(F):
        for c in range(K):
            rows = trn[(folds == f) & (yi[trn] == c)]
            if len(rows):
                cnt[f, c] = C[rows].sum(0, dtype=np.float64)
    tot = cnt.sum(0)

    def logp(counts):
        c = counts.reshape(K, M // 4, 4) + 1.0
        return np.log(c / c.sum(2, keepdims=True)).reshape(K, M)

    def logprior(rows):
        pr = np.bincount(yi[rows], minlength=K).astype(np.float64)
        return np.log(np.maximum(pr, 1e-12) / max(pr.sum(), 1e-12))

    cvp = np.full(len(trn), -1)
    cvs = np.zeros(len(trn))
    nk_tr = C[trn].sum(1)
    for f in range(F):
        held = folds == f
        if not held.any():
            continue
        ll = C[trn[held]].astype(np.float64) @ logp(tot - cnt[f]).T + logprior(trn[~held])[None, :]
        cvp[held] = ll.argmax(1)
        if K == 2:
            cvs[held] = (ll[:, 1] - ll[:, 0]) / np.maximum(nk_tr[held], 1)
    ll = C[tei].astype(np.float64) @ logp(tot).T + logprior(trn)[None, :]
    pte = cvp if cv_as_test else ll.argmax(1)
    out = dict(cv=S.score(yi[tr], cvp, K, metric), test=S.score(yi[te], pte, K, metric),
               ci=S.bootstrap_ci(yi[te], pte, K, metric, n_boot, seed), pred_test=pte)
    if K == 2:
        nk = C[tei].sum(1)
        out['cv_auroc'] = S.auroc(yi[tr], cvs)
        out['test_auroc'] = out['cv_auroc'] if cv_as_test else S.auroc(yi[te], (ll[:, 1] - ll[:, 0]) / np.maximum(nk, 1))
    return out


def kmer_counts_of(seqs, k):
    codes, _, off = concat(seqs)
    return kmer_counts(codes, off, k)


def _motif_select(X, yi, K, rows, metric, n_cand=60, n_greedy=40, max_feats=4):
    """Greedy (position, k-mer) indicator table selected on ``rows`` (forward selection, vectorised)."""
    n, L = X.shape
    cands = []
    for k in (1, 2, 3):
        npos = L - k + 1
        if npos <= 0:
            continue
        code = np.zeros((len(rows), npos), np.int64)
        bad = np.zeros((len(rows), npos), bool)
        for j in range(k):
            c = X[rows, j:j + npos]
            bad |= c > 3
            code = code * 4 + np.minimum(c, 3)
        code = np.where(bad, 4 ** k, code)   # invalid -> extra symbol
        yr = yi[rows]
        A = 4 ** k + 1
        flat = (np.arange(npos)[None, :] * A + code).ravel()
        for cl in range(K):
            m1 = yr == cl
            if m1.sum() == 0 or (~m1).sum() == 0:
                continue
            f1 = np.bincount((np.arange(npos)[None, :] * A + code[m1]).ravel(), minlength=npos * A) / m1.sum()
            f0 = np.bincount((np.arange(npos)[None, :] * A + code[~m1]).ravel(), minlength=npos * A) / (~m1).sum()
            diff = np.abs(f1 - f0).reshape(npos, A)[:, :4 ** k]
            for km in range(4 ** k):
                top = np.argsort(-diff[:, km], kind='stable')[:3]
                for p in top:
                    cands.append((float(diff[p, km]), k, int(p), int(km)))
        del flat
    cands.sort(key=lambda z: (-z[0], z[1], z[2], z[3]))
    seen = set()
    uniq = []
    for c in cands:
        if (c[1], c[2], c[3]) not in seen:
            seen.add((c[1], c[2], c[3]))
            uniq.append(c)
    cands = uniq[:n_cand]

    def feat(k, p, km):
        code = np.zeros(n, np.int64)
        bad = np.zeros(n, bool)
        for j in range(k):
            c = X[:, p + j]
            bad |= c > 3
            code = code * 4 + np.minimum(c, 3)
        return (code == km) & ~bad

    feats = [(c[1], c[2], c[3], feat(c[1], c[2], c[3])) for c in cands]

    def table(cols):
        key = np.zeros(n, np.int64)
        for i, c in enumerate(cols):
            key |= c.astype(np.int64) << i
        cnt = np.zeros((2 ** len(cols), K), np.int64)
        np.add.at(cnt, (key[rows], yi[rows]), 1)
        if K == 2:
            # per-cell positive rate (add-one) as the score; labels from the train MCC-optimal cut of the score
            rate = (cnt[:, 1] + 1.0) / (cnt.sum(1) + 2.0)
            sc = rate[key]
            t, d, _ = S.best_threshold(sc[rows], yi[rows])
            return S.apply_threshold(sc, t, d), sc
        glob = int(np.argmax(np.bincount(yi[rows], minlength=K)))
        lab = np.where(cnt.sum(1) > 0, cnt.argmax(1), glob)
        return lab[key], None

    chosen = []
    pred = score = None
    for _ in range(max_feats):
        best = None
        for i in range(min(n_greedy, len(feats))):
            if i in chosen:
                continue
            p, sc = table([feats[j][3] for j in chosen + [i]])
            v = S.auroc(yi[rows], sc[rows]) if K == 2 else S.score(yi[rows], p[rows], K, metric)
            v = -1.0 if v is None else v
            if best is None or v > best[0]:
                best = (v, i, p, sc)
        if best is None:
            break
        chosen.append(best[1])
        pred, score = best[2], best[3]
    return [feats[i][:3] for i in chosen], pred, score


def motif_rule(seqs, yi, K, tr, te, folds, metric='mcc', n_boot=1000, seed=0, max_len=2000, max_rows=50_000,
               cv_as_test=False):
    """Fixed-position motif table (fixed-length tasks <= max_len). Selection uses at most ``max_rows`` train items
    (seeded subsample) per fit."""
    lens = np.fromiter((len(s) for s in seqs), np.int64, count=len(seqs))
    if len(np.unique(lens)) != 1 or lens[0] > max_len or lens[0] < 4:
        return dict(skipped='variable or long sequences')
    L = int(lens[0])
    codes, _, off = concat(seqs)
    X = codes.reshape(len(seqs), L)
    rng = np.random.default_rng(seed)
    trn = np.flatnonzero(tr)

    def sub(rows):
        return np.sort(rng.choice(rows, max_rows, replace=False)) if len(rows) > max_rows else rows

    cvp = np.full(len(trn), -1)
    cvs = np.zeros(len(trn))
    for f in np.unique(folds):
        rows = sub(trn[folds != f])
        _, pred, sc = _motif_select(X, yi, K, rows, metric)
        if pred is not None:
            cvp[folds == f] = pred[trn[folds == f]]
        if sc is not None:
            cvs[folds == f] = sc[trn[folds == f]]
    feats, pred, score = _motif_select(X, yi, K, sub(trn), metric)
    alph = 'ACGT'

    def kmstr(k, km):
        return ''.join(alph[(km >> (2 * (k - 1 - j))) & 3] for j in range(k))

    pte = pred[te] if pred is not None else np.zeros(int(np.sum(te)), np.int64)
    if cv_as_test:
        pte = cvp
    out = dict(cv=S.score(yi[tr], cvp, K, metric), test=S.score(yi[te], pte, K, metric),
               ci=S.bootstrap_ci(yi[te], pte, K, metric, n_boot, seed), pred_test=pte,
               features=[dict(k=k, pos=p, kmer=kmstr(k, km)) for k, p, km in feats])
    if K == 2 and score is not None:
        out['cv_auroc'] = S.auroc(yi[tr], cvs)
        out['test_auroc'] = out['cv_auroc'] if cv_as_test else S.auroc(yi[te], score[te])
    return out
