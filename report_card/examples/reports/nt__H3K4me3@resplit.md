# posbench audit: nt__H3K4me3@resplit

**Verdict: PASS**. no check reached its warning threshold.

Items: train 31826, dev 0, test 3592; classification, classes [0, 1]; length 310-500 (median 500); elements seq.
Nearest-training lookup: mcc 0.046 [0.016, 0.078] (chance 0.000).
Best no-model rule on test (shortcut.kmer:markov3): mcc 0.229 [0.196, 0.262] (selected on test: an upper bound for no-model rules).

| check | status | statistic | value [95% CI] | warn / fail |
|---|---|---|---|---|
| overlap.exact | PASS | share of test items with an exact or reverse-complement copy in a fit split | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| overlap.near | PASS | share of test items with k-mer containment >= 0.5 in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.100 |
| overlap.windows | PASS | share of test items with a collinear shared run >= 0.2 of their k-mers (alignment-free) | 0.003 [0.002, 0.005] | 0.010 / 0.250 |
| overlap.pair | SKIP | single-element items |  |  |
| overlap.homology | PASS | share of test items with a fit-split hit of >= 80% identity over >= 50% of the query | 0.000 [0.000, 0.002] | 0.050 / 0.250 |
| reference.copies | SKIP | no --reference given |  |  |
| shortcut.length | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.ambiguous | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.case | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.lowinfo | PASS | share of test items made mostly of N or other ambiguity codes | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| shortcut.composition | PASS | test MCC of the CV-selected one-feature rule | 0.110 [0.080, 0.142] | 0.500 / - |
| shortcut.kmer | PASS | test MCC of the order-3 Markov Bayes rule | 0.229 [0.196, 0.262] | 0.500 / - |
| shortcut.position_motif | SKIP | variable or long sequences |  |  |
| construction.shuffle | PASS | share of test items with a composition twin (same dinucleotide counts) in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| construction.segment | PASS | share of test items with a positional twin (segments kept in place) in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| construction.synthetic_negatives | PASS | share of test items in twin pairs with different labels (pairs anywhere in the data) | 0.000 [0.000, 0.001] | 0.050 / 0.250 |
| construction.unmatched_site | SKIP | variable or long sequences |  |  |
| construction.library | PASS | test MCC (classification) or correlation (regression) of the library-group rule | 0.000 | 0.100 / 0.300 |
| construction.genome_membership | PASS | test MCC of the train-fitted membership rule | -0.023 [-0.055, 0.011] | 0.200 / 0.500 |
| shortcut.anchored_motif | PASS | test MCC of the greedy <= 4-feature table anchored at the item ends (or site) | 0.020 [-0.013, 0.053] | 0.400 / 0.700 |
| position.distance | SKIP | no --distance-cols given |  |  |
| position.row_zero | SKIP | no --task-dir given |  |  |
| baseline.lookup | INFO | mcc of the nearest-training lookup on test | 0.046 [0.016, 0.078] |  |

## baseline.lookup (INFO): Nearest-training lookup baseline
nearest-training lookup: mcc 0.046 [0.016, 0.078] (chance 0.000)

Evidence: `{"acc": 0.5381403118040089, "auroc": 0.5137579403267882, "chance": 0.0, "mcc": 0.04643935045232176, "metric": "mcc", "n_with_partner": 179}`

Runtime 207.85 s, peak memory 2171.2 MB. posbench 0.4.0.dev2, schema posbench.audit/1.
