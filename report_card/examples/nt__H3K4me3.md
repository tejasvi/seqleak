### seqleak audit: nt__H3K4me3

**Verdict: FAIL.** Leak checks failing: overlap.near, overlap.windows, overlap.homology.

Items: train 33,119, dev 0, test 3,680; classification, classes 0, 1; length 290-500. Checks: 3 FAIL, 13 PASS, 6 SKIP, 1 INFO.

- Nearest-training lookup (label of the most similar training sequence): MCC 0.823 [0.804, 0.841]; chance 0.000.
- Best no-model rule, chosen on test (an upper bound): MCC 0.823 [0.804, 0.841]; label of the nearest training sequence.

| | check | value [95% CI] | warn / fail | finding |
|---|---|---|---|---|
| **FAIL** | Near-duplicates (k-mer containment) (`overlap.near`) | 0.570 [0.554, 0.586] | 0.010 / 0.100 | 2098 of 3680 test items share >= 50% of their k-mers with one fit-split sequence |
| **FAIL** | Overlapping or shifted genome windows (`overlap.windows`) | 0.930 [0.921, 0.938] | 0.010 / 0.250 | 3423 of 3680 test items share a collinear (overlapping or shifted) segment with a training/dev sequence; 0.930 of test items at a non-zero offset |
| **FAIL** | Homology (alignment) (`overlap.homology`) | 0.611 [0.589, 0.632] | 0.050 / 0.250 | 1222 of 2000 sampled test items have a homologous training/dev sequence (BLASTN dc-megablast) |

What to do:
- `overlap.near`: Test items share most of their k-mers with a training sequence. Keep each group of near-duplicates in one split (seqleak clean --mode resplit groups them).
- `overlap.windows`: Test windows overlap, or are shifted copies of, training windows. Split by genomic locus or chromosome so that overlapping windows stay together (seqleak clean --group-by locus).
- `overlap.homology`: Test items have close homologs in the training data. Cluster by sequence identity and keep each cluster in one split.

<details><summary>All checks</summary>

| check | status | value [95% CI] | warn / fail |
|---|---|---|---|
| Exact or reverse-complement duplicates across splits (`overlap.exact`) | PASS | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| Near-duplicates (k-mer containment) (`overlap.near`) | FAIL | 0.570 [0.554, 0.586] | 0.010 / 0.100 |
| Overlapping or shifted genome windows (`overlap.windows`) | FAIL | 0.930 [0.921, 0.938] | 0.010 / 0.250 |
| Multi-element items copied from one training item (`overlap.pair`) | SKIP | single-element items |  |
| Homology (alignment) (`overlap.homology`) | FAIL | 0.611 [0.589, 0.632] | 0.050 / 0.250 |
| Copies of another benchmark's sequences (`reference.copies`) | SKIP | no --reference given |  |
| Length shortcut (`shortcut.length`) | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| N / padding / ambiguity-code shortcut (`shortcut.ambiguous`) | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| Lowercase (soft-masking) shortcut (`shortcut.case`) | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| Low-information inputs (windows made mostly of N) (`shortcut.lowinfo`) | PASS | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| GC / CpG / naturalness composition rule (`shortcut.composition`) | PASS | 0.071 [0.038, 0.102] | 0.500 / - |
| k-mer composition rule (order-3 Markov) (`shortcut.kmer`) | PASS | 0.221 [0.191, 0.249] | 0.500 / - |
| Fixed-position motif table (`shortcut.position_motif`) | SKIP | variable or long sequences |  |
| Dinucleotide-shuffle negatives (composition twins) (`construction.shuffle`) | PASS | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| Segment-substitution negatives (positional twins) (`construction.segment`) | PASS | 0.001 [0.000, 0.003] | 0.010 / 0.050 |
| Constructed negatives anywhere in the data (opposite-label twin pairs) (`construction.synthetic_negatives`) | PASS | 0.000 [0.000, 0.001] | 0.050 / 0.250 |
| Negatives that lack the site defining the positives (`construction.unmatched_site`) | SKIP | variable or long sequences |  |
| Pooled-library structure (shared constant segments predict the label) (`construction.library`) | PASS | 0.000 | 0.100 / 0.300 |
| Negatives absent from the reference genome (membership predicts the label) (`construction.genome_membership`) | PASS | 0.023 [-0.009, 0.054] | 0.200 / 0.500 |
| Motif table anchored at the item ends (variable-length items) (`shortcut.anchored_motif`) | PASS | 0.037 [0.006, 0.069] | 0.400 / 0.700 |
| Position confound (distance fields) (`position.distance`) | SKIP | no --distance-cols given |  |
| Position confound (gene-pair row zero) (`position.row_zero`) | SKIP | no --task-dir given |  |
| Nearest-training lookup baseline (`baseline.lookup`) | INFO | 0.823 [0.804, 0.841] |  |

</details>

Linter rules v1.2; posbench 0.4.0.dev2; schema posbench.audit/1. Reading: FAIL marks a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of test items in the training data, negatives constructed from positives, artefact or motif shortcuts, or position confounds. WARN marks the same signals below that level, or a composition baseline that a model must beat. It is common on tasks without a documented leak, so treat it as a reason to look closer. PASS means that no check reached its warning threshold, and SKIP that the input lacks what the check needs (coordinates, distance fields, an aligner, or items with several sequences).
