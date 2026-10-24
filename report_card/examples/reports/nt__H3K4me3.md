# posbench audit: nt__H3K4me3

**Verdict: FAIL**. leak checks failing: overlap.near, overlap.windows, overlap.homology.

Items: train 33119, dev 0, test 3680; classification, classes [0, 1]; length 290-500 (median 500); elements seq.
Nearest-training lookup: mcc 0.823 [0.804, 0.841] (chance 0.000).
Best no-model rule on test (lookup): mcc 0.823 [0.804, 0.841] (selected on test: an upper bound for no-model rules).

| check | status | statistic | value [95% CI] | warn / fail |
|---|---|---|---|---|
| overlap.exact | PASS | share of test items with an exact or reverse-complement copy in a fit split | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| overlap.near | FAIL | share of test items with k-mer containment >= 0.5 in a fit split | 0.570 [0.554, 0.586] | 0.010 / 0.100 |
| overlap.windows | FAIL | share of test items with a collinear shared run >= 0.2 of their k-mers (alignment-free) | 0.930 [0.921, 0.938] | 0.010 / 0.250 |
| overlap.pair | SKIP | single-element items |  |  |
| overlap.homology | FAIL | share of test items with a fit-split hit of >= 80% identity over >= 50% of the query | 0.611 [0.589, 0.632] | 0.050 / 0.250 |
| reference.copies | SKIP | no --reference given |  |  |
| shortcut.length | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.ambiguous | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.case | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.lowinfo | PASS | share of test items made mostly of N or other ambiguity codes | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| shortcut.composition | PASS | test MCC of the CV-selected one-feature rule | 0.071 [0.038, 0.102] | 0.500 / - |
| shortcut.kmer | PASS | test MCC of the order-3 Markov Bayes rule | 0.221 [0.191, 0.249] | 0.500 / - |
| shortcut.position_motif | SKIP | variable or long sequences |  |  |
| construction.shuffle | PASS | share of test items with a composition twin (same dinucleotide counts) in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| construction.segment | PASS | share of test items with a positional twin (segments kept in place) in a fit split | 0.001 [0.000, 0.003] | 0.010 / 0.050 |
| construction.synthetic_negatives | PASS | share of test items in twin pairs with different labels (pairs anywhere in the data) | 0.000 [0.000, 0.001] | 0.050 / 0.250 |
| construction.unmatched_site | SKIP | variable or long sequences |  |  |
| construction.library | PASS | test MCC (classification) or correlation (regression) of the library-group rule | 0.000 | 0.100 / 0.300 |
| construction.genome_membership | PASS | test MCC of the train-fitted membership rule | 0.023 [-0.009, 0.054] | 0.200 / 0.500 |
| shortcut.anchored_motif | PASS | test MCC of the greedy <= 4-feature table anchored at the item ends (or site) | 0.037 [0.006, 0.069] | 0.400 / 0.700 |
| position.distance | SKIP | no --distance-cols given |  |  |
| position.row_zero | SKIP | no --task-dir given |  |  |
| baseline.lookup | INFO | mcc of the nearest-training lookup on test | 0.823 [0.804, 0.841] |  |

## overlap.near (FAIL): Near-duplicates (k-mer containment)
2098 of 3680 test items share >= 50% of their k-mers with one fit-split sequence

Evidence: `{"not_assessable": 0.0, "per_element": {"seq": {"median_c": 0.5323160826177767, "near_excluding_exact": 0.5701086956521739, "not_assessable": 0.0, "share_c20": 0.9307065217391305, "share_c50": 0.5701086956521739, "share_c80": 0.034782608695652174}}}`

Examples:
- test_item=YDR213W_UPC2_890049|1, test_label=1, fit_item=YDR213W_UPC2_889984|1, fit_label=1, containment=0.9, collinear=0.9
- test_item=YBL017C_PEP1_191478|1, test_label=1, fit_item=iYBL017C_191549|1, fit_label=1, containment=0.8991, collinear=0.8991
- test_item=iYGL045W_415880|0, test_label=0, fit_item=iYGL045W_415940|0, fit_label=0, containment=0.8966, collinear=0.8966
- test_item=YHR150W_PEX28_397546|1, test_label=1, fit_item=YHR150W_PEX28_397482|1, fit_label=1, containment=0.8889, collinear=0.8889
- test_item=iYMR021C_318411|1, test_label=1, fit_item=YMR021C_MAC1_318321|1, fit_label=1, containment=0.8819, collinear=0.8819

## overlap.windows (FAIL): Overlapping or shifted genome windows
3423 of 3680 test items share a collinear (overlapping or shifted) segment with a training/dev sequence; 0.930 of test items at a non-zero offset

Evidence: `{"alignment_free": {"median_overlap": 0.5476190476190477, "share": 0.9301630434782608, "shifted": 0.9301630434782608}}`

Examples:
- test_item=YDR213W_UPC2_890049|1, fit_item=YDR213W_UPC2_889984|1, overlap=0.9, shift=-65, strand=same
- test_item=YBL017C_PEP1_191478|1, fit_item=iYBL017C_191549|1, overlap=0.8991, shift=71, strand=same
- test_item=iYGL045W_415880|0, fit_item=iYGL045W_415940|0, overlap=0.8966, shift=60, strand=same
- test_item=YHR150W_PEX28_397546|1, fit_item=YHR150W_PEX28_397482|1, overlap=0.8889, shift=-64, strand=same
- test_item=iYMR021C_318411|1, fit_item=YMR021C_MAC1_318321|1, overlap=0.8819, shift=-90, strand=same

## overlap.homology (FAIL): Homology (alignment)
1222 of 2000 sampled test items have a homologous training/dev sequence (BLASTN dc-megablast)

Evidence: `{"any_hit": 0.977, "homolog_label_agreement": 0.9541734860883797, "n_subsample": 2000, "near_duplicate_share": 0.0365, "tool": "BLASTN dc-megablast"}`

## baseline.lookup (INFO): Nearest-training lookup baseline
nearest-training lookup: mcc 0.823 [0.804, 0.841] (chance 0.000)

Evidence: `{"acc": 0.9119565217391304, "auroc": 0.9465138526340768, "chance": 0.0, "mcc": 0.8231216298497681, "metric": "mcc", "n_with_partner": 3587}`

Runtime 139.03 s, peak memory 2225.5 MB. posbench 0.4.0.dev2, schema posbench.audit/1.
