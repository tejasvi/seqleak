### seqleak audit: gb__human_enhancers_cohn

**Verdict: WARN.** Warnings: overlap.windows, overlap.homology.

Items: train 20,843, dev 0, test 6,948; classification, classes 0, 1; length 500-500. Checks: 2 WARN, 15 PASS, 5 SKIP, 1 INFO.

- Nearest-training lookup (label of the most similar training sequence): accuracy 0.501 [0.489, 0.513]; chance 0.500.
- Best no-model rule, chosen on test (an upper bound): accuracy 0.681 [0.671, 0.692]; label of the nearest training sequence when it is a near copy, otherwise the k-mer composition rule (order-3 Markov).

| | check | value [95% CI] | warn / fail | finding |
|---|---|---|---|---|
| **WARN** | Overlapping or shifted genome windows (`overlap.windows`) | 0.022 [0.018, 0.025] | 0.010 / 0.250 | 150 of 6948 test items share a collinear (overlapping or shifted) segment with a training/dev sequence; 0.022 of test items at a non-zero offset |
| **WARN** | Homology (alignment) (`overlap.homology`) | 0.114 [0.100, 0.128] | 0.050 / 0.250 | 227 of 2000 sampled test items have a homologous training/dev sequence (BLASTN dc-megablast) |

<details><summary>All checks</summary>

| check | status | value [95% CI] | warn / fail |
|---|---|---|---|
| Exact or reverse-complement duplicates across splits (`overlap.exact`) | PASS | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| Near-duplicates (k-mer containment) (`overlap.near`) | PASS | 0.007 [0.005, 0.009] | 0.010 / 0.100 |
| Overlapping or shifted genome windows (`overlap.windows`) | WARN | 0.022 [0.018, 0.025] | 0.010 / 0.250 |
| Multi-element items copied from one training item (`overlap.pair`) | SKIP | single-element items |  |
| Homology (alignment) (`overlap.homology`) | WARN | 0.114 [0.100, 0.128] | 0.050 / 0.250 |
| Copies of another benchmark's sequences (`reference.copies`) | SKIP | no --reference given |  |
| Length shortcut (`shortcut.length`) | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| N / padding / ambiguity-code shortcut (`shortcut.ambiguous`) | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| Lowercase (soft-masking) shortcut (`shortcut.case`) | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| Low-information inputs (windows made mostly of N) (`shortcut.lowinfo`) | PASS | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| GC / CpG / naturalness composition rule (`shortcut.composition`) | PASS | 0.339 [0.318, 0.362] | 0.500 / - |
| k-mer composition rule (order-3 Markov) (`shortcut.kmer`) | PASS | 0.364 [0.343, 0.387] | 0.500 / - |
| Fixed-position motif table (`shortcut.position_motif`) | PASS | 0.069 [0.046, 0.091] | 0.400 / 0.700 |
| Dinucleotide-shuffle negatives (composition twins) (`construction.shuffle`) | PASS | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| Segment-substitution negatives (positional twins) (`construction.segment`) | PASS | 0.081 [0.075, 0.087] | 0.010 / 0.050 |
| Constructed negatives anywhere in the data (opposite-label twin pairs) (`construction.synthetic_negatives`) | PASS | 0.000 [0.000, 0.001] | 0.050 / 0.250 |
| Negatives that lack the site defining the positives (`construction.unmatched_site`) | PASS | 0.000 | 0.050 / 0.250 |
| Pooled-library structure (shared constant segments predict the label) (`construction.library`) | PASS | 0.000 | 0.100 / 0.300 |
| Negatives absent from the reference genome (membership predicts the label) (`construction.genome_membership`) | PASS | 0.000 [0.000, 0.000] | 0.200 / 0.500 |
| Motif table anchored at the item ends (variable-length items) (`shortcut.anchored_motif`) | SKIP | fixed-length items (see shortcut.position_motif) |  |
| Position confound (distance fields) (`position.distance`) | SKIP | no --distance-cols given |  |
| Position confound (gene-pair row zero) (`position.row_zero`) | SKIP | no --task-dir given |  |
| Nearest-training lookup baseline (`baseline.lookup`) | INFO | 0.501 [0.489, 0.513] |  |

</details>

Linter rules v1.2; posbench 0.4.0.dev2; schema posbench.audit/1. Reading: FAIL marks a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of test items in the training data, negatives constructed from positives, artefact or motif shortcuts, or position confounds. WARN marks the same signals below that level, or a composition baseline that a model must beat. It is common on tasks without a documented leak, so treat it as a reason to look closer. PASS means that no check reached its warning threshold, and SKIP that the input lacks what the check needs (coordinates, distance fields, an aligner, or items with several sequences).
