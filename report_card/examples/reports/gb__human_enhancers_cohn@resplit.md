# posbench audit: gb__human_enhancers_cohn@resplit

**Verdict: PASS**. no check reached its warning threshold.

Items: train 16137, dev 0, test 6292; classification, classes [0, 1]; length 500-500 (median 500); elements seq.
Nearest-training lookup: acc 0.493 [0.479, 0.505] (chance 0.500).
Best no-model rule on test (twin:construction.segment_else:shortcut.kmer): acc 0.647 [0.635, 0.658] (selected on test: an upper bound for no-model rules).

| check | status | statistic | value [95% CI] | warn / fail |
|---|---|---|---|---|
| overlap.exact | PASS | share of test items with an exact or reverse-complement copy in a fit split | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| overlap.near | PASS | share of test items with k-mer containment >= 0.5 in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.100 |
| overlap.windows | PASS | share of test items with a collinear shared run >= 0.2 of their k-mers (alignment-free) | 0.001 [0.000, 0.002] | 0.010 / 0.250 |
| overlap.pair | SKIP | single-element items |  |  |
| overlap.homology | PASS | share of test items with a fit-split hit of >= 80% identity over >= 50% of the query | 0.001 [0.000, 0.003] | 0.050 / 0.250 |
| reference.copies | SKIP | no --reference given |  |  |
| shortcut.length | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.ambiguous | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.case | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.lowinfo | PASS | share of test items made mostly of N or other ambiguity codes | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| shortcut.composition | PASS | test MCC of the CV-selected one-feature rule | 0.274 [0.251, 0.296] | 0.500 / - |
| shortcut.kmer | PASS | test MCC of the order-3 Markov Bayes rule | 0.297 [0.274, 0.319] | 0.500 / - |
| shortcut.position_motif | PASS | test MCC of the greedy <= 4-feature fixed-position table | 0.052 [0.025, 0.076] | 0.400 / 0.700 |
| construction.shuffle | PASS | share of test items with a composition twin (same dinucleotide counts) in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| construction.segment | PASS | share of test items with a positional twin (segments kept in place) in a fit split | 0.002 [0.001, 0.004] | 0.010 / 0.050 |
| construction.synthetic_negatives | PASS | share of test items in twin pairs with different labels (pairs anywhere in the data) | 0.000 [0.000, 0.001] | 0.050 / 0.250 |
| construction.unmatched_site | PASS | share of other-class test items lacking the site that defines a class | 0.000 | 0.050 / 0.250 |
| construction.library | PASS | test MCC (classification) or correlation (regression) of the library-group rule | 0.000 | 0.100 / 0.300 |
| construction.genome_membership | PASS | test MCC of the train-fitted membership rule | 0.000 [0.000, 0.000] | 0.200 / 0.500 |
| shortcut.anchored_motif | SKIP | fixed-length items (see shortcut.position_motif) |  |  |
| position.distance | SKIP | no --distance-cols given |  |  |
| position.row_zero | SKIP | no --task-dir given |  |  |
| baseline.lookup | INFO | acc of the nearest-training lookup on test | 0.493 [0.479, 0.505] |  |

## baseline.lookup (INFO): Nearest-training lookup baseline
nearest-training lookup: acc 0.493 [0.479, 0.505] (chance 0.500)

Evidence: `{"acc": 0.4928480610298792, "auroc": 0.5453864967027424, "chance": 0.5, "mcc": -0.025728609510453845, "metric": "acc", "n_with_partner": 1556}`

Runtime 1017.51 s, peak memory 2165.2 MB. posbench 0.4.0.dev2, schema posbench.audit/1.
