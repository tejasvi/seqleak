## Leakage audit

The splits in this repository were checked for data leakage with `posbench audit` (linter rules v1.2; posbench 0.4.0.dev2).

**Verdict: FAIL.** Leak checks failing: overlap.near, overlap.windows, overlap.homology.

| check | id | result | value [95% CI] | warn / fail |
|---|---|---|---|---|
| Exact or reverse-complement duplicates across splits | `overlap.exact` | PASS | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| Near-duplicates (k-mer containment) | `overlap.near` | FAIL | 0.570 [0.554, 0.586] | 0.010 / 0.100 |
| Overlapping or shifted genome windows | `overlap.windows` | FAIL | 0.930 [0.921, 0.938] | 0.010 / 0.250 |
| Homology (alignment) | `overlap.homology` | FAIL | 0.611 [0.589, 0.632] | 0.050 / 0.250 |
| Length shortcut | `shortcut.length` | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| N / padding / ambiguity-code shortcut | `shortcut.ambiguous` | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| Lowercase (soft-masking) shortcut | `shortcut.case` | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| Low-information inputs (windows made mostly of N) | `shortcut.lowinfo` | PASS | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| GC / CpG / naturalness composition rule | `shortcut.composition` | PASS | 0.071 [0.038, 0.102] | 0.500 / - |
| k-mer composition rule (order-3 Markov) | `shortcut.kmer` | PASS | 0.221 [0.191, 0.249] | 0.500 / - |
| Dinucleotide-shuffle negatives (composition twins) | `construction.shuffle` | PASS | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| Segment-substitution negatives (positional twins) | `construction.segment` | PASS | 0.001 [0.000, 0.003] | 0.010 / 0.050 |
| Constructed negatives anywhere in the data (opposite-label twin pairs) | `construction.synthetic_negatives` | PASS | 0.000 [0.000, 0.001] | 0.050 / 0.250 |
| Pooled-library structure (shared constant segments predict the label) | `construction.library` | PASS | 0.000 | 0.100 / 0.300 |
| Negatives absent from the reference genome (membership predicts the label) | `construction.genome_membership` | PASS | 0.023 [-0.009, 0.054] | 0.200 / 0.500 |
| Motif table anchored at the item ends (variable-length items) | `shortcut.anchored_motif` | PASS | 0.037 [0.006, 0.069] | 0.400 / 0.700 |

Skipped (SKIP): `overlap.pair` (single-element items); `reference.copies` (no --reference given); `shortcut.position_motif` (variable or long sequences); `construction.unmatched_site` (variable or long sequences); `position.distance` (no --distance-cols given); `position.row_zero` (no --task-dir given).

- Nearest-training lookup (label of the most similar training sequence): MCC 0.823 [0.804, 0.841]; chance 0.000.
- Best no-model rule: the nearest-training lookup above. It is chosen on the test split, so it is an upper bound.

How to read the verdict. FAIL marks a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of test items in the training data, negatives constructed from positives, artefact or motif shortcuts, or position confounds. WARN marks the same signals below that level, or a composition baseline that a model must beat. It is common on tasks without a documented leak, so treat it as a reason to look closer. PASS means that no check reached its warning threshold, and SKIP that the input lacks what the check needs (coordinates, distance fields, an aligner, or items with several sequences).

The linter reports only what its checks measure. It checks position only when distance fields are given, copies of other benchmarks only with --reference, membership in a reference genome only with --genome, and has no check for transductive arrangements; see "Known limitations" in the seqleak README.

To reproduce:

```
posbench audit --data nt__H3K4me3.parquet --out nt__H3K4me3.json --task-name nt__H3K4me3 --threads 1 --homology blast --homology-max-test 2000 --genome sacCer3
```
