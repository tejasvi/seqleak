"""Default thresholds. ``warn`` / ``fail`` apply to the check's statistic (``>=``); None = that level is never reached.
Overrides: --thresholds FILE (JSON with the same keys). The FAIL levels of overlap.near (0.10), overlap.windows and
overlap.pair (0.25) follow the operating points of the manual audit the linter was validated against; the twin
checks FAIL only when twins carry different labels (``fail_direction='flip'``: constructed negatives), and judge the
twin label relation against chance agreement (``chance_corrected``). For binary labels whose minority class is below
20% of train (``auroc_max_minority``) the shortcut checks also score each rule by AUROC against (1 + MCC threshold) / 2
(``auroc``), so imbalance does not hide a shortcut.
Rules v1.1 adds: low-information inputs (shortcut.lowinfo), constructed negatives anywhere in the data
(construction.synthetic_negatives), negatives lacking the site that defines the positives (construction.unmatched_site;
WARN at most with more than two classes, ``multiclass_fail``), pooled-library structure (construction.library); the motif check's imbalance criterion FAILs only with an unmatched
site (``auroc_requires_unmatched_site``); with --published-best, a construction shortcut that WARNs FAILs when it reaches
``relative_fail`` of the best published above-chance score (motif 0.75, artefacts 0.5).
Rules v1.2 adds: membership in the reference genome(s) given with --genome (construction.genome_membership: a
train-fitted rule on each item's share of canonical k-mers found in the references, ``k``, ``per_item``; members have
share >= ``member_min``; items shorter than ``min_len`` cannot be placed); synthetic_negatives scored on the share of
test items (``statistic``); motif tables anchored at the ends of variable-length items or at a site column
(shortcut.anchored_motif; ``anchor`` bases from each end; the AUROC criterion can only WARN there).
Rules v1.2.1 changes no threshold: in multi-class tasks the one-feature rules give a point mass at the largest feature
value its own quantile bin (stats.fit_bins)."""
import copy
import json

RULES_VERSION = 'v1.2.1'

DEFAULTS = {
    'overlap.exact': dict(warn=0.005, fail=0.05, min_items=3),
    'overlap.near': dict(warn=0.01, fail=0.10, near=0.5, not_assessable_warn=0.5, min_informative=0.2),
    'overlap.windows': dict(warn=0.01, fail=0.25, collinear=0.2),
    'overlap.pair': dict(warn=0.01, fail=0.25, near=0.5),
    'overlap.homology': dict(warn=0.05, fail=0.25, homolog_id=80.0, homolog_cov=0.5, near_id=95.0, near_cov=0.8),
    'reference.copies': dict(warn=0.01, fail=0.05, near=0.5),
    'shortcut.length': dict(warn=0.1, fail=0.3, auroc=True, auroc_max_minority=0.2, relative_fail=0.5),
    'shortcut.ambiguous': dict(warn=0.1, fail=0.3, auroc=True, auroc_max_minority=0.2, relative_fail=0.5),
    'shortcut.case': dict(warn=0.1, fail=0.3, auroc=True, auroc_max_minority=0.2, relative_fail=0.5),
    'shortcut.lowinfo': dict(warn=0.005, fail=0.05, min_items=3, min_acgt=0.5),
    'shortcut.composition': dict(warn=0.5, fail=None, auroc=True, auroc_max_minority=0.2),
    'shortcut.kmer': dict(warn=0.5, fail=None, auroc=True, auroc_max_minority=0.2),
    'shortcut.position_motif': dict(warn=0.4, fail=0.7, auroc=True, auroc_max_minority=0.2,
                                    auroc_requires_unmatched_site=True, relative_fail=0.75),
    'construction.shuffle': dict(warn=0.01, fail=0.05, rule_acc=0.7, within_warn=0.05, max_identity=0.6,
                                 fail_direction='flip', chance_corrected=True),
    'construction.segment': dict(warn=0.01, fail=0.05, rule_acc=0.7, within_warn=0.05, min_identity=0.4,
                                 fail_direction='flip', chance_corrected=True),
    'construction.synthetic_negatives': dict(warn=0.05, fail=0.25, kappa=0.4, min_pairs=10, statistic='test_share'),
    'construction.unmatched_site': dict(warn=0.05, fail=0.25, site_min=0.95, multiclass_fail=False),
    'construction.library': dict(warn=0.1, fail=0.3, min_df=0.05, min_kmers=5, pos_consistency=0.8, max_items=40000,
                                 min_group_share=0.01),
    'construction.genome_membership': dict(warn=0.2, fail=0.5, auroc=True, auroc_max_minority=0.2, k=21, per_item=32,
                                           member_min=0.9, min_len=30, align_cov=0.95, align_ident=0.95),
    'shortcut.anchored_motif': dict(warn=0.4, fail=0.7, auroc=True, auroc_max_minority=0.2, anchor=100,
                                    relative_fail=0.75),
    'position.distance': dict(warn=0.6, fail=0.75),
    'position.row_zero': dict(warn=0.6, fail=0.75),
    'baseline.lookup': dict(min_shared=2),
}


def load(path=None):
    t = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            over = json.load(f)
        for k, v in over.items():
            if k not in t:
                raise ValueError(f'unknown check in thresholds file: {k}')
            t[k].update(v)
    return t


def status(value, th, extra_fail=True):
    """PASS / WARN / FAIL of a statistic against a threshold dict (>=)."""
    if value is None:
        return 'SKIP'
    if th.get('fail') is not None and value >= th['fail'] and extra_fail:
        return 'FAIL'
    if th.get('warn') is not None and value >= th['warn']:
        return 'WARN'
    return 'PASS'
