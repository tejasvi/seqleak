# posbench audit: gue__prom_core_all

**Verdict: FAIL**. leak checks failing: construction.segment, construction.synthetic_negatives, construction.genome_membership; warnings: overlap.exact, overlap.near, overlap.windows, overlap.homology, shortcut.kmer.

Items: train 47356, dev 5920, test 5920; classification, classes [0, 1]; length 70-70 (median 70); elements seq.
Nearest-training lookup: mcc -0.311 [-0.334, -0.290] (chance 0.000).
Best no-model rule on test (construction.genome_membership:membership): mcc 0.901 [0.891, 0.911] (selected on test: an upper bound for no-model rules).

| check | status | statistic | value [95% CI] | warn / fail |
|---|---|---|---|---|
| overlap.exact | WARN | share of test items with an exact or reverse-complement copy in a fit split | 0.009 [0.007, 0.011] | 0.005 / 0.050 |
| overlap.near | WARN | share of test items with k-mer containment >= 0.5 in a fit split | 0.065 [0.059, 0.072] | 0.010 / 0.100 |
| overlap.windows | WARN | share of test items with a collinear shared run >= 0.2 of their k-mers (alignment-free) | 0.245 [0.234, 0.256] | 0.010 / 0.250 |
| overlap.pair | SKIP | single-element items |  |  |
| overlap.homology | WARN | share of test items with a fit-split hit of >= 80% identity over >= 50% of the query | 0.176 [0.160, 0.194] | 0.050 / 0.250 |
| reference.copies | SKIP | no --reference given |  |  |
| shortcut.length | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.ambiguous | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.case | PASS | test MCC of the CV-selected one-feature rule | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| shortcut.lowinfo | PASS | share of test items made mostly of N or other ambiguity codes | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| shortcut.composition | PASS | test MCC of the CV-selected one-feature rule | 0.410 [0.386, 0.431] | 0.500 / - |
| shortcut.kmer | WARN | test MCC of the order-3 Markov Bayes rule | 0.536 [0.515, 0.557] | 0.500 / - |
| shortcut.position_motif | PASS | test MCC of the greedy <= 4-feature fixed-position table | 0.325 [0.300, 0.349] | 0.400 / 0.700 |
| construction.shuffle | PASS | share of test items with a composition twin (same dinucleotide counts) in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| construction.segment | FAIL | share of test items with a positional twin (segments kept in place) in a fit split | 0.672 [0.660, 0.684] | 0.010 / 0.050 |
| construction.synthetic_negatives | FAIL | share of test items in twin pairs with different labels (pairs anywhere in the data) | 0.723 [0.711, 0.734] | 0.050 / 0.250 |
| construction.unmatched_site | PASS | share of other-class test items lacking the site that defines a class | 0.000 | 0.050 / 0.250 |
| construction.library | PASS | test MCC (classification) or correlation (regression) of the library-group rule | 0.000 | 0.100 / 0.300 |
| construction.genome_membership | FAIL | test MCC of the train-fitted membership rule | 0.901 [0.891, 0.911] | 0.200 / 0.500 |
| shortcut.anchored_motif | SKIP | fixed-length items (see shortcut.position_motif) |  |  |
| position.distance | SKIP | no --distance-cols given |  |  |
| position.row_zero | SKIP | no --task-dir given |  |  |
| baseline.lookup | INFO | mcc of the nearest-training lookup on test | -0.311 [-0.334, -0.290] |  |

## overlap.exact (WARN): Exact or reverse-complement duplicates across splits
51 of 5920 test items have an exact or reverse-complement copy in train/dev; their copies carry the same label 0.863 of the time

Evidence: `{"dev_in_train": 0.009121621621621622, "fit_groups_with_conflicting_labels": 56, "label_agreement": 0.8627450980392157, "test_in_dev": 0.0013513513513513514, "test_in_train": 0.007263513513513514, "test_within_duplicates": 0.0006756756756757243}`

Examples:
- test_item=#53286, test_label=0, fit_item=#28568, fit_split=train, fit_label=0, reverse_complement=False
- test_item=#53336, test_label=1, fit_item=#46477, fit_split=train, fit_label=1, reverse_complement=False
- test_item=#53347, test_label=1, fit_item=#8281, fit_split=train, fit_label=1, reverse_complement=False
- test_item=#53355, test_label=1, fit_item=#37989, fit_split=train, fit_label=1, reverse_complement=False
- test_item=#53424, test_label=1, fit_item=#16704, fit_split=train, fit_label=1, reverse_complement=False

## overlap.near (WARN): Near-duplicates (k-mer containment)
387 of 5920 test items share >= 50% of their k-mers with one fit-split sequence

Evidence: `{"not_assessable": 0.0, "per_element": {"seq": {"median_c": 0.0, "near_excluding_exact": 0.05675675675675676, "not_assessable": 0.0, "share_c20": 0.2456081081081081, "share_c50": 0.06537162162162162, "share_c80": 0.018243243243243244}}}`

Examples:
- test_item=#53286, test_label=0, fit_item=#28568, fit_label=0, containment=1.0, collinear=1.0
- test_item=#53336, test_label=1, fit_item=#46477, fit_label=1, containment=1.0, collinear=1.0
- test_item=#53347, test_label=1, fit_item=#8281, fit_label=1, containment=1.0, collinear=1.0
- test_item=#53355, test_label=1, fit_item=#37989, fit_label=1, containment=1.0, collinear=1.0
- test_item=#53424, test_label=1, fit_item=#16704, fit_label=1, containment=1.0, collinear=1.0

## overlap.windows (WARN): Overlapping or shifted genome windows
1451 of 5920 test items share a collinear (overlapping or shifted) segment with a training/dev sequence; 0.016 of test items at a non-zero offset

Evidence: `{"alignment_free": {"median_overlap": 0.24, "share": 0.24510135135135136, "shifted": 0.016047297297297296}}`

Examples:
- test_item=#53286, fit_item=#28568, overlap=1.0, shift=0, strand=same
- test_item=#53336, fit_item=#46477, overlap=1.0, shift=0, strand=same
- test_item=#53347, fit_item=#8281, overlap=1.0, shift=0, strand=same
- test_item=#53355, fit_item=#37989, overlap=1.0, shift=0, strand=same
- test_item=#53424, fit_item=#16704, overlap=1.0, shift=0, strand=same

## overlap.homology (WARN): Homology (alignment)
353 of 2000 sampled test items have a homologous training/dev sequence (BLASTN blastn)

Evidence: `{"any_hit": 0.3415, "homolog_label_agreement": 0.23229461756373937, "n_subsample": 2000, "near_duplicate_share": 0.0245, "tool": "BLASTN blastn"}`

## shortcut.kmer (WARN): k-mer composition rule (order-3 Markov)
order-3 Markov rule: test MCC 0.536 (CV 0.532); test AUROC 0.844; composition can be real biology

Evidence: `{"cv_auroc": 0.842231180939884, "cv_mcc": 0.5316022602029972, "status_from": "mcc", "test_auroc": 0.8440481783102871}`

## construction.segment (FAIL): Segment-substitution negatives (positional twins)
3981 of 5920 test items have a twin in train/dev; their label is the partner's label flipped for 0.935 of them; 0.747 of all items are in twin pairs, 0.795 of pairs with different labels

Evidence: `{"chance_agreement": 0.5000004058416853, "identity_quartiles": [0.45714285714285713, 0.5428571428571428, 0.6285714285714286], "kappa": 0.8703842203393722, "label_determined_share_ci": [0.9271173700607409, 0.942427889331527], "label_determined_share_covered": 0.9351921627731725, "label_relation_on_test": "flip", "n_pairs": 29612, "opposite_label_share_all_pairs": 0.79504930433608, "p_same_label_fit_pairs": 0.20380231300440968, "share_items_in_twin_pairs": 0.7470775052368404, "share_pairs_identity_ge_0.9": 0.011380521410239092, "twin_rule_accuracy_covered": 0.9351921627731725, "twin_rule_direction": "flip"}`

Examples:
- test_item=#53299, test_label=1, fit_item=#10625, fit_label=1
- test_item=#53310, test_label=0, fit_item=#14300, fit_label=0
- test_item=#53327, test_label=0, fit_item=#53219, fit_label=0
- test_item=#53336, test_label=1, fit_item=#1257, fit_label=1
- test_item=#53343, test_label=1, fit_item=#44869, fit_label=1

## construction.synthetic_negatives (FAIL): Constructed negatives anywhere in the data (opposite-label twin pairs)
4279 of 5920 test items are in twin pairs with different labels of a twin kind whose pairs disagree beyond chance (dinucleotide shuffles or segment substitutions of each other; 0.728 of all items); composition: 2 pairs, flip kappa 0.00; positional: 29612 pairs, flip kappa 0.59 (counted)

Evidence: `{"kinds": {"composition": {"counted": false, "flip_kappa": 0.0, "n_pairs": 2, "share_items_opposite": 3.3786066626123385e-05}, "positional": {"counted": true, "flip_kappa": 0.59009860867216, "n_pairs": 29612, "share_items_opposite": 0.7277349820933847}}, "share_all_items": 0.7277349820933847}`

Examples:
- item=#53356, split=test, label=0
- item=#53430, split=test, label=1
- item=#53434, split=test, label=0
- item=#53453, split=test, label=1
- item=#53474, split=test, label=1

## construction.genome_membership (FAIL): Negatives absent from the reference genome (membership predicts the label)
membership rule (fitted on train): test MCC 0.901 (CV 0.900); test AUROC 0.948; near-full-length members (membership >= 0.9) among test items: class 0 0.10, class 1 1.00

Evidence: `{"cv_auroc": 0.9476270828939043, "cv_mcc": 0.9000703276530987, "direction": 1, "k": 21, "member_min": 0.9, "method": "kmer", "per_item": 32, "query_kmers": 1789633, "reference_bases": 3099925961, "references": [{"bytes": 3161924569, "file": "hg38.fa"}], "shares": {"test": {"0": {"mean_membership": 0.19412686314363142, "members": 0.1043360433604336, "n": 2952}, "1": {"mean_membership": 1.0, "members": 1.0, "n": 2968}}, "train": {"0": {"mean_membership": 0.195138156507066, "members": 0.10478801940518878, "n": 23705}, "1": {"mean_membership": 1.0, "members": 1.0, "n": 23651}}}, "status_from": "mcc", "test_auroc": 0.9478319783197832, "threshold": 0.9375, "unplaced_items": 0}`

## baseline.lookup (INFO): Nearest-training lookup baseline
nearest-training lookup: mcc -0.311 [-0.334, -0.290] (chance 0.000)

Evidence: `{"acc": 0.3802364864864865, "auroc": 0.29141728117079013, "chance": 0.0, "mcc": -0.3109097495660567, "metric": "mcc", "n_with_partner": 1890}`

Runtime 948.88 s, peak memory 2181.2 MB. posbench 0.4.0.dev2, schema posbench.audit/1.
