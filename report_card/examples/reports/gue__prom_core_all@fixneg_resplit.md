# posbench audit: gue__prom_core_all@fixneg_resplit

**Verdict: WARN**. warnings: shortcut.position_motif.

Items: train 44297, dev 5572, test 5601; classification, classes [0, 1]; length 70-70 (median 70); elements seq.
Nearest-training lookup: mcc 0.049 [0.029, 0.067] (chance 0.000).
Best no-model rule on test (shortcut.kmer:markov3): mcc 0.492 [0.470, 0.513] (selected on test: an upper bound for no-model rules).

| check | status | statistic | value [95% CI] | warn / fail |
|---|---|---|---|---|
| overlap.exact | PASS | share of test items with an exact or reverse-complement copy in a fit split | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| overlap.near | PASS | share of test items with k-mer containment >= 0.5 in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.100 |
| overlap.windows | PASS | share of test items with a collinear shared run >= 0.2 of their k-mers (alignment-free) | 0.000 [0.000, 0.001] | 0.010 / 0.250 |
| overlap.pair | SKIP | single-element items |  |  |
| overlap.homology | PASS | share of test items with a fit-split hit of >= 80% identity over >= 50% of the query | 0.000 [0.000, 0.002] | 0.050 / 0.250 |
| reference.copies | SKIP | no --reference given |  |  |
| shortcut.length | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.ambiguous | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.case | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.lowinfo | PASS | share of test items made mostly of N or other ambiguity codes | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| shortcut.composition | PASS | test MCC of the CV-selected one-feature rule | 0.404 [0.378, 0.427] | 0.500 / - |
| shortcut.kmer | PASS | test MCC of the order-3 Markov Bayes rule | 0.492 [0.470, 0.513] | 0.500 / - |
| shortcut.position_motif | WARN | test MCC of the greedy <= 4-feature fixed-position table | 0.452 [0.429, 0.476] | 0.400 / 0.700 |
| construction.shuffle | PASS | share of test items with a composition twin (same dinucleotide counts) in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| construction.segment | PASS | share of test items with a positional twin (segments kept in place) in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| construction.synthetic_negatives | PASS | share of test items in twin pairs with different labels (pairs anywhere in the data) | 0.000 [0.000, 0.001] | 0.050 / 0.250 |
| construction.unmatched_site | PASS | share of other-class test items lacking the site that defines a class | 0.000 | 0.050 / 0.250 |
| construction.library | PASS | test MCC (classification) or correlation (regression) of the library-group rule | 0.000 | 0.100 / 0.300 |
| construction.genome_membership | PASS | test MCC of the train-fitted membership rule | 0.000 [0.000, 0.000] | 0.200 / 0.500 |
| shortcut.anchored_motif | SKIP | fixed-length items (see shortcut.position_motif) |  |  |
| position.distance | SKIP | no --distance-cols given |  |  |
| position.row_zero | SKIP | no --task-dir given |  |  |
| baseline.lookup | INFO | mcc of the nearest-training lookup on test | 0.049 [0.029, 0.067] |  |

## shortcut.position_motif (WARN): Fixed-position motif table
table on A@33, CG@32, C@32, A@32: test MCC 0.452 (CV 0.464); test AUROC 0.768

Evidence: `{"cv_auroc": 0.7722447382202102, "cv_mcc": 0.4642191489089416, "features": [{"k": 1, "kmer": "A", "pos": 33}, {"k": 2, "kmer": "CG", "pos": 32}, {"k": 1, "kmer": "C", "pos": 32}, {"k": 1, "kmer": "A", "pos": 32}], "status_from": "mcc", "test_auroc": 0.7683107468508471}`

## baseline.lookup (INFO): Nearest-training lookup baseline
nearest-training lookup: mcc 0.049 [0.029, 0.067] (chance 0.000)

Evidence: `{"acc": 0.5016961256918407, "auroc": 0.5030252804375611, "chance": 0.0, "mcc": 0.04949765256490385, "metric": "mcc", "n_with_partner": 31}`

Runtime 1076.49 s, peak memory 2182.0 MB. posbench 0.4.0.dev2, schema posbench.audit/1.
