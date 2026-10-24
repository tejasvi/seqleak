### seqleak audit: gue__prom_core_all

**Verdict: FAIL.** Leak checks failing: construction.segment, construction.synthetic_negatives, construction.genome_membership; warnings: overlap.exact, overlap.near, overlap.windows, overlap.homology, shortcut.kmer.

Items: train 47,356, dev 5,920, test 5,920; classification, classes 0, 1; length 70-70. Checks: 3 FAIL, 5 WARN, 9 PASS, 5 SKIP, 1 INFO.

- Nearest-training lookup (label of the most similar training sequence): MCC -0.311 [-0.334, -0.290]; chance 0.000.
- Best no-model rule, chosen on test (an upper bound): MCC 0.901 [0.891, 0.911]; negatives absent from the reference genome (membership predicts the label).

| | check | value [95% CI] | warn / fail | finding |
|---|---|---|---|---|
| **FAIL** | Segment-substitution negatives (positional twins) (`construction.segment`) | 0.672 [0.660, 0.684] | 0.010 / 0.050 | 3981 of 5920 test items have a twin in train/dev; their label is the partner's label flipped for 0.935 of them; 0.747 of all items are in twin pairs, 0.795 of pairs with different labels |
| **FAIL** | Constructed negatives anywhere in the data (opposite-label twin pairs) (`construction.synthetic_negatives`) | 0.723 [0.711, 0.734] | 0.050 / 0.250 | 4279 of 5920 test items are in twin pairs with different labels of a twin kind whose pairs disagree beyond chance (dinucleotide shuffles or segment substitutions of each other; 0.728 of all items); composition: 2 pairs, flip kappa 0.00; positional: 29612 pairs, flip kappa 0.59 (counted) |
| **FAIL** | Negatives absent from the reference genome (membership predicts the label) (`construction.genome_membership`) | 0.901 [0.891, 0.911] | 0.200 / 0.500 | membership rule (fitted on train): test MCC 0.901 (CV 0.900); test AUROC 0.948; near-full-length members (membership >= 0.9) among test items: class 0 0.10, class 1 1.00 |
| **WARN** | Exact or reverse-complement duplicates across splits (`overlap.exact`) | 0.009 [0.007, 0.011] | 0.005 / 0.050 | 51 of 5920 test items have an exact or reverse-complement copy in train/dev; their copies carry the same label 0.863 of the time |
| **WARN** | Near-duplicates (k-mer containment) (`overlap.near`) | 0.065 [0.059, 0.072] | 0.010 / 0.100 | 387 of 5920 test items share >= 50% of their k-mers with one fit-split sequence |
| **WARN** | Overlapping or shifted genome windows (`overlap.windows`) | 0.245 [0.234, 0.256] | 0.010 / 0.250 | 1451 of 5920 test items share a collinear (overlapping or shifted) segment with a training/dev sequence; 0.016 of test items at a non-zero offset |
| **WARN** | Homology (alignment) (`overlap.homology`) | 0.176 [0.160, 0.194] | 0.050 / 0.250 | 353 of 2000 sampled test items have a homologous training/dev sequence (BLASTN blastn) |
| **WARN** | k-mer composition rule (order-3 Markov) (`shortcut.kmer`) | 0.536 [0.515, 0.557] | 0.500 / - | order-3 Markov rule: test MCC 0.536 (CV 0.532); test AUROC 0.844; composition can be real biology |

What to do:
- `construction.segment`: Negatives were made by substituting segments of positives, and twin pairs cross the split. Keeping twins in one split (seqleak clean) removes the cross-split leak, but the negatives stay constructed: genomic background windows matched on length and GC remove the shortcut.
- `construction.synthetic_negatives`: Negatives are dinucleotide shuffles or segment substitutions of positives, anywhere in the data (within a split too). A re-split does not remove them: replace them with genomic background windows matched on length and GC.
- `construction.genome_membership`: Membership in the reference genome alone gives the label: the negatives are not genome sequence (shuffled, substituted or otherwise generated). Draw negatives from the genome (windows matched on length and GC), or report the membership rule as a baseline.

<details><summary>All checks</summary>

| check | status | value [95% CI] | warn / fail |
|---|---|---|---|
| Exact or reverse-complement duplicates across splits (`overlap.exact`) | WARN | 0.009 [0.007, 0.011] | 0.005 / 0.050 |
| Near-duplicates (k-mer containment) (`overlap.near`) | WARN | 0.065 [0.059, 0.072] | 0.010 / 0.100 |
| Overlapping or shifted genome windows (`overlap.windows`) | WARN | 0.245 [0.234, 0.256] | 0.010 / 0.250 |
| Multi-element items copied from one training item (`overlap.pair`) | SKIP | single-element items |  |
| Homology (alignment) (`overlap.homology`) | WARN | 0.176 [0.160, 0.194] | 0.050 / 0.250 |
| Copies of another benchmark's sequences (`reference.copies`) | SKIP | no --reference given |  |
| Length shortcut (`shortcut.length`) | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| N / padding / ambiguity-code shortcut (`shortcut.ambiguous`) | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| Lowercase (soft-masking) shortcut (`shortcut.case`) | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| Low-information inputs (windows made mostly of N) (`shortcut.lowinfo`) | PASS | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| GC / CpG / naturalness composition rule (`shortcut.composition`) | PASS | 0.410 [0.386, 0.431] | 0.500 / - |
| k-mer composition rule (order-3 Markov) (`shortcut.kmer`) | WARN | 0.536 [0.515, 0.557] | 0.500 / - |
| Fixed-position motif table (`shortcut.position_motif`) | PASS | 0.325 [0.300, 0.349] | 0.400 / 0.700 |
| Dinucleotide-shuffle negatives (composition twins) (`construction.shuffle`) | PASS | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| Segment-substitution negatives (positional twins) (`construction.segment`) | FAIL | 0.672 [0.660, 0.684] | 0.010 / 0.050 |
| Constructed negatives anywhere in the data (opposite-label twin pairs) (`construction.synthetic_negatives`) | FAIL | 0.723 [0.711, 0.734] | 0.050 / 0.250 |
| Negatives that lack the site defining the positives (`construction.unmatched_site`) | PASS | 0.000 | 0.050 / 0.250 |
| Pooled-library structure (shared constant segments predict the label) (`construction.library`) | PASS | 0.000 | 0.100 / 0.300 |
| Negatives absent from the reference genome (membership predicts the label) (`construction.genome_membership`) | FAIL | 0.901 [0.891, 0.911] | 0.200 / 0.500 |
| Motif table anchored at the item ends (variable-length items) (`shortcut.anchored_motif`) | SKIP | fixed-length items (see shortcut.position_motif) |  |
| Position confound (distance fields) (`position.distance`) | SKIP | no --distance-cols given |  |
| Position confound (gene-pair row zero) (`position.row_zero`) | SKIP | no --task-dir given |  |
| Nearest-training lookup baseline (`baseline.lookup`) | INFO | -0.311 [-0.334, -0.290] |  |

</details>

Linter rules v1.2; posbench 0.4.0.dev2; schema posbench.audit/1. Reading: FAIL marks a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of test items in the training data, negatives constructed from positives, artefact or motif shortcuts, or position confounds. WARN marks the same signals below that level, or a composition baseline that a model must beat. It is common on tasks without a documented leak, so treat it as a reason to look closer. PASS means that no check reached its warning threshold, and SKIP that the input lacks what the check needs (coordinates, distance fields, an aligner, or items with several sequences).
