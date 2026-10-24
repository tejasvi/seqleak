# posbench audit: gue__prom_core_all@resplit_union

**Verdict: FAIL**. leak checks failing: construction.synthetic_negatives, construction.genome_membership; warnings: shortcut.kmer, construction.segment.

Items: train 44543, dev 5594, test 5601; classification, classes [0, 1]; length 70-70 (median 70); elements seq.
Nearest-training lookup: mcc 0.055 [0.038, 0.071] (chance 0.000).
Best no-model rule on test (lookup_if_near_else:construction.genome_membership): mcc 0.893 [0.882, 0.904] (selected on test: an upper bound for no-model rules).

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
| shortcut.composition | PASS | test MCC of the CV-selected one-feature rule | 0.380 [0.357, 0.403] | 0.500 / - |
| shortcut.kmer | WARN | test MCC of the order-3 Markov Bayes rule | 0.525 [0.503, 0.546] | 0.500 / - |
| shortcut.position_motif | PASS | test MCC of the greedy <= 4-feature fixed-position table | 0.321 [0.295, 0.344] | 0.400 / 0.700 |
| construction.shuffle | PASS | share of test items with a composition twin (same dinucleotide counts) in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| construction.segment | WARN | share of test items with a positional twin (segments kept in place) in a fit split | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| construction.synthetic_negatives | FAIL | share of test items in twin pairs with different labels (pairs anywhere in the data) | 0.699 [0.687, 0.711] | 0.050 / 0.250 |
| construction.unmatched_site | PASS | share of other-class test items lacking the site that defines a class | 0.000 | 0.050 / 0.250 |
| construction.library | PASS | test MCC (classification) or correlation (regression) of the library-group rule | 0.000 | 0.100 / 0.300 |
| construction.genome_membership | FAIL | test MCC of the train-fitted membership rule | 0.893 [0.882, 0.904] | 0.200 / 0.500 |
| shortcut.anchored_motif | SKIP | fixed-length items (see shortcut.position_motif) |  |  |
| position.distance | SKIP | no --distance-cols given |  |  |
| position.row_zero | SKIP | no --task-dir given |  |  |
| baseline.lookup | INFO | mcc of the nearest-training lookup on test | 0.055 [0.038, 0.071] |  |

## shortcut.kmer (WARN): k-mer composition rule (order-3 Markov)
order-3 Markov rule: test MCC 0.525 (CV 0.532); test AUROC 0.836; composition can be real biology

Evidence: `{"cv_auroc": 0.8431139507295791, "cv_mcc": 0.531910775756802, "status_from": "mcc", "test_auroc": 0.8356779208909535}`

## construction.segment (WARN): Segment-substitution negatives (positional twins)
0 of 5601 test items have a twin in train/dev; 0.706 of all items are in twin pairs, 0.915 of pairs with different labels

Evidence: `{"identity_quartiles": [0.4714285714285714, 0.5571428571428572, 0.6571428571428571], "n_pairs": 21799, "opposite_label_share_all_pairs": 0.9148126060828479, "p_same_label_fit_pairs": 0.0873905072316154, "share_items_in_twin_pairs": 0.7056944992644156, "share_pairs_identity_ge_0.9": 0.011881278957750355, "within_flip_kappa": 0.8296252121656957}`

## construction.synthetic_negatives (FAIL): Constructed negatives anywhere in the data (opposite-label twin pairs)
3916 of 5601 test items are in twin pairs with different labels of a twin kind whose pairs disagree beyond chance (dinucleotide shuffles or segment substitutions of each other; 0.692 of all items); composition: 1 pairs, flip kappa 1.00; positional: 21799 pairs, flip kappa 0.83 (counted)

Evidence: `{"kinds": {"composition": {"counted": false, "flip_kappa": 1.0, "n_pairs": 1, "share_items_opposite": 3.588216297678424e-05}, "positional": {"counted": true, "flip_kappa": 0.8296252121656957, "n_pairs": 21799, "share_items_opposite": 0.692489863288959}}, "share_all_items": 0.692489863288959}`

Examples:
- item=#14, split=test, label=0
- item=#25, split=test, label=1
- item=#29, split=test, label=1
- item=#46, split=test, label=1
- item=#95, split=test, label=0

## construction.genome_membership (FAIL): Negatives absent from the reference genome (membership predicts the label)
membership rule (fitted on train): test MCC 0.893 (CV 0.905); test AUROC 0.944; near-full-length members (membership >= 0.9) among test items: class 0 0.11, class 1 1.00

Evidence: `{"cv_auroc": 0.9500829708032471, "cv_mcc": 0.9045915425302986, "direction": 1, "k": 21, "member_min": 0.9, "method": "kmer", "per_item": 32, "query_kmers": 1692572, "reference_bases": 3099925961, "references": [{"bytes": 3161924569, "file": "hg38.fa"}], "shares": {"test": {"0": {"mean_membership": 0.1986439312567132, "members": 0.11278195488721804, "n": 2793}, "1": {"mean_membership": 1.0, "members": 1.0, "n": 2808}}, "train": {"0": {"mean_membership": 0.18916081759878012, "members": 0.09983405839350586, "n": 22297}, "1": {"mean_membership": 1.0, "members": 1.0, "n": 22246}}}, "status_from": "mcc", "test_auroc": 0.943609022556391, "threshold": 0.875, "unplaced_items": 0}`

## baseline.lookup (INFO): Nearest-training lookup baseline
nearest-training lookup: mcc 0.055 [0.038, 0.071] (chance 0.000)

Evidence: `{"acc": 0.5020532047848598, "auroc": 0.5033812145340968, "chance": 0.0, "mcc": 0.05534023442863752, "metric": "mcc", "n_with_partner": 25}`

Runtime 962.07 s, peak memory 2217.7 MB. posbench 0.4.0.dev2, schema posbench.audit/1.
