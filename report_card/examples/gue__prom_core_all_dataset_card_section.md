## Leakage audit

The splits in this repository were checked for data leakage with `posbench audit` (linter rules v1.2; posbench 0.4.0.dev2).

**Verdict: FAIL.** Leak checks failing: construction.segment, construction.synthetic_negatives, construction.genome_membership; warnings: overlap.exact, overlap.near, overlap.windows, overlap.homology, shortcut.kmer.

| check | id | result | value [95% CI] | warn / fail |
|---|---|---|---|---|
| Exact or reverse-complement duplicates across splits | `overlap.exact` | WARN | 0.009 [0.007, 0.011] | 0.005 / 0.050 |
| Near-duplicates (k-mer containment) | `overlap.near` | WARN | 0.065 [0.059, 0.072] | 0.010 / 0.100 |
| Overlapping or shifted genome windows | `overlap.windows` | WARN | 0.245 [0.234, 0.256] | 0.010 / 0.250 |
| Homology (alignment) | `overlap.homology` | WARN | 0.176 [0.160, 0.194] | 0.050 / 0.250 |
| Length shortcut | `shortcut.length` | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| N / padding / ambiguity-code shortcut | `shortcut.ambiguous` | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| Lowercase (soft-masking) shortcut | `shortcut.case` | PASS | 0.000 [0.000, 0.000] | 0.100 / 0.300 |
| Low-information inputs (windows made mostly of N) | `shortcut.lowinfo` | PASS | 0.000 [0.000, 0.001] | 0.005 / 0.050 |
| GC / CpG / naturalness composition rule | `shortcut.composition` | PASS | 0.410 [0.386, 0.431] | 0.500 / - |
| k-mer composition rule (order-3 Markov) | `shortcut.kmer` | WARN | 0.536 [0.515, 0.557] | 0.500 / - |
| Fixed-position motif table | `shortcut.position_motif` | PASS | 0.325 [0.300, 0.349] | 0.400 / 0.700 |
| Dinucleotide-shuffle negatives (composition twins) | `construction.shuffle` | PASS | 0.000 [0.000, 0.001] | 0.010 / 0.050 |
| Segment-substitution negatives (positional twins) | `construction.segment` | FAIL | 0.672 [0.660, 0.684] | 0.010 / 0.050 |
| Constructed negatives anywhere in the data (opposite-label twin pairs) | `construction.synthetic_negatives` | FAIL | 0.723 [0.711, 0.734] | 0.050 / 0.250 |
| Negatives that lack the site defining the positives | `construction.unmatched_site` | PASS | 0.000 | 0.050 / 0.250 |
| Pooled-library structure (shared constant segments predict the label) | `construction.library` | PASS | 0.000 | 0.100 / 0.300 |
| Negatives absent from the reference genome (membership predicts the label) | `construction.genome_membership` | FAIL | 0.901 [0.891, 0.911] | 0.200 / 0.500 |

Skipped (SKIP): `overlap.pair` (single-element items); `reference.copies` (no --reference given); `shortcut.anchored_motif` (fixed-length items (see shortcut.position_motif)); `position.distance` (no --distance-cols given); `position.row_zero` (no --task-dir given).

- Nearest-training lookup (label of the most similar training sequence): MCC -0.311 [-0.334, -0.290]; chance 0.000.
- Best no-model rule (negatives absent from the reference genome (membership predicts the label)): MCC 0.901 [0.891, 0.911]. It is chosen on the test split, so it is an upper bound.

How to read the verdict. FAIL marks a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of test items in the training data, negatives constructed from positives, artefact or motif shortcuts, or position confounds. WARN marks the same signals below that level, or a composition baseline that a model must beat. It is common on tasks without a documented leak, so treat it as a reason to look closer. PASS means that no check reached its warning threshold, and SKIP that the input lacks what the check needs (coordinates, distance fields, an aligner, or items with several sequences).

The linter reports only what its checks measure. It checks position only when distance fields are given, copies of other benchmarks only with --reference, membership in a reference genome only with --genome, and has no check for transductive arrangements; see "Known limitations" in the seqleak README.

To reproduce:

```
posbench audit --data gue__prom_core_all.parquet --out gue__prom_core_all.json --task-name gue__prom_core_all --threads 1 --homology blast --homology-max-test 2000 --genome hg38
```
