"""seqleak: gene-position baselines and position-held-fixed metrics for genomic language model benchmarks.

Modules: ``features`` (pair and gene position features), ``strata`` (position strata), ``metrics`` (AUROC/AUPRC
with position held fixed, gain over position, position share), ``bootstrap`` (shared resampling plans),
``baseline`` (position-only model with genome-grouped CV), ``tasks`` (task directories), ``report`` (row-zero table),
``audit`` (leak linter, ``seqleak audit``) and ``clean`` (reduced-leakage splits, ``seqleak clean``).
"""
from . import baseline, bootstrap, features, metrics, report, strata, tasks
from .baseline import position_model
from .bootstrap import Resample
from .features import gene_features, pair_features, rule_score
from .metrics import (auprc, auroc, equalized_score, explained_share, explained_share_ovr, gain_over_position,
                      position_share, position_share_bound, position_share_result, within_auc)
from .report import evaluate, format_table, row_zero
from .strata import make_strata

__version__ = '0.4.1'
