# posbench audit: gb__human_enhancers_cohn

**Verdict: WARN**. warnings: overlap.windows, overlap.homology.

Items: train 20843, dev 0, test 6948; classification, classes [0, 1]; length 500-500 (median 500); elements seq.
Nearest-training lookup: acc 0.501 [0.489, 0.513] (chance 0.500).
Best no-model rule on test (lookup_if_near_else:shortcut.kmer): acc 0.681 [0.671, 0.692] (selected on test: an upper bound for no-model rules).

| check | status | statistic | value [95% CI] | warn / fail |
|---|---|---|---|---|
| overlap.exact | PASS | share of test items with an exact or reverse-complement copy in a fit split | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| overlap.near | PASS | share of test items with k-mer containment >= 0.5 in a fit split | 0.007 [0.005, 0.009] | 0.010 / 0.100 |
| overlap.windows | WARN | share of test items with a collinear shared run >= 0.2 of their k-mers (alignment-free) | 0.022 [0.018, 0.025] | 0.010 / 0.250 |
| overlap.pair | SKIP | single-element items |  |  |
| overlap.homology | WARN | share of test items with a fit-split hit of >= 80% identity over >= 50% of the query | 0.114 [0.100, 0.128] | 0.050 / 0.250 |
| reference.copies | SKIP | no --reference given |  |  |
| shortcut.length | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.ambiguous | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.case | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.lowinfo | PASS | share of test items made mostly of N or other ambiguity codes | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| shortcut.composition | PASS | test MCC of the CV-selected one-feature rule | 0.339 [0.318, 0.362] | 0.500 / - |
| shortcut.kmer | PASS | test MCC of the order-3 Markov Bayes rule | 0.364 [0.343, 0.387] | 0.500 / - |
| shortcut.position_motif | PASS | test MCC of the greedy <= 4-feature fixed-position table | 0.069 [0.046, 0.091] | 0.400 / 0.700 |
| construction.shuffle | PASS | share of test items with a composition twin (same dinucleotide counts) in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| construction.segment | PASS | share of test items with a positional twin (segments kept in place) in a fit split | 0.081 [0.075, 0.087] | 0.010 / 0.050 |
| construction.synthetic_negatives | PASS | share of test items in twin pairs with different labels (pairs anywhere in the data) | 0.000 [0.000, 0.001] | 0.050 / 0.250 |
| construction.unmatched_site | PASS | share of other-class test items lacking the site that defines a class | 0.000 | 0.050 / 0.250 |
| construction.library | PASS | test MCC (classification) or correlation (regression) of the library-group rule | 0.000 | 0.100 / 0.300 |
| construction.genome_membership | PASS | test MCC of the train-fitted membership rule | 0.000 [0.000, 0.000] | 0.200 / 0.500 |
| shortcut.anchored_motif | SKIP | fixed-length items (see shortcut.position_motif) |  |  |
| position.distance | SKIP | no --distance-cols given |  |  |
| position.row_zero | SKIP | no --task-dir given |  |  |
| baseline.lookup | INFO | acc of the nearest-training lookup on test | 0.501 [0.489, 0.513] |  |

## overlap.windows (WARN): Overlapping or shifted genome windows
150 of 6948 test items share a collinear (overlapping or shifted) segment with a training/dev sequence; 0.022 of test items at a non-zero offset

Evidence: `{"alignment_free": {"median_overlap": 0.3785495849716033, "share": 0.02158894645941278, "shifted": 0.02158894645941278}}`

Examples:
- test_item=#22323, fit_item=#1359, overlap=0.9919, shift=-4, strand=same
- test_item=#22235, fit_item=#3374, overlap=0.9916, shift=-4, strand=same
- test_item=#25820, fit_item=#16656, overlap=0.9542, shift=1, strand=same
- test_item=#20979, fit_item=#2807, overlap=0.8704, shift=67, strand=same
- test_item=#21785, fit_item=#4580, overlap=0.8509, shift=-94, strand=same

## overlap.homology (WARN): Homology (alignment)
227 of 2000 sampled test items have a homologous training/dev sequence (BLASTN dc-megablast)

Evidence: `{"any_hit": 0.392, "homolog_label_agreement": 0.7312775330396476, "n_subsample": 2000, "near_duplicate_share": 0.0085, "tool": "BLASTN dc-megablast"}`

## baseline.lookup (INFO): Nearest-training lookup baseline
nearest-training lookup: acc 0.501 [0.489, 0.513] (chance 0.500)

Evidence: `{"acc": 0.5010074841681059, "auroc": 0.5773483769056357, "chance": 0.5, "mcc": 0.0035314521224739073, "metric": "acc", "n_with_partner": 2090}`

Runtime 1223.35 s, peak memory 2206.8 MB. posbench 0.4.0.dev2, schema posbench.audit/1.
