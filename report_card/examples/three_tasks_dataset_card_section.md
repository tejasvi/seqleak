## Leakage audit

The splits in this repository were checked for data leakage with `posbench audit` (linter rules v1.2; posbench 0.4.0.dev2).

**Verdict: FAIL.** 7 tasks: 3 FAIL, 2 WARN, 2 PASS.

| task | verdict | failing checks | nearest-training lookup (chance) | best no-model rule |
|---|---|---|---|---|
| gue__prom_core_all | FAIL | construction.segment, construction.synthetic_negatives, construction.genome_membership | -0.311 (0.000) | 0.901 |
| gue__prom_core_all@resplit_union | FAIL | construction.synthetic_negatives, construction.genome_membership | 0.055 (0.000) | 0.893 |
| gue__prom_core_all@fixneg_resplit | WARN | - | 0.049 (0.000) | 0.492 |
| nt__H3K4me3 | FAIL | overlap.near, overlap.windows, overlap.homology | 0.823 (0.000) | 0.823 |
| nt__H3K4me3@resplit | PASS | - | 0.046 (0.000) | 0.229 |
| gb__human_enhancers_cohn | WARN | - | 0.501 (0.500) | 0.681 |
| gb__human_enhancers_cohn@resplit | PASS | - | 0.493 (0.500) | 0.647 |

Lookup: label of the most similar training sequence. Best no-model rule: chosen on the test split, an upper bound for rules without a model. Metric: MCC, accuracy.

How to read the verdict. FAIL marks a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of test items in the training data, negatives constructed from positives, artefact or motif shortcuts, or position confounds. WARN marks the same signals below that level, or a composition baseline that a model must beat. It is common on tasks without a documented leak, so treat it as a reason to look closer. PASS means that no check reached its warning threshold, and SKIP that the input lacks what the check needs (coordinates, distance fields, an aligner, or items with several sequences).

The linter reports only what its checks measure. It checks position only when distance fields are given, copies of other benchmarks only with --reference, membership in a reference genome only with --genome, and has no check for transductive arrangements; see "Known limitations" in the seqleak README.

To reproduce:

```
posbench audit --data gue__prom_core_all.parquet --out gue__prom_core_all.json --task-name gue__prom_core_all --threads 1 --homology blast --homology-max-test 2000 --genome hg38
posbench audit --data gue__prom_core_all@resplit_union.parquet --out gue__prom_core_all@resplit_union.json --task-name gue__prom_core_all@resplit_union --threads 1 --homology blast --homology-max-test 2000 --genome hg38
posbench audit --data gue__prom_core_all@fixneg_resplit.parquet --out gue__prom_core_all@fixneg_resplit.json --task-name gue__prom_core_all@fixneg_resplit --threads 1 --homology blast --homology-max-test 2000 --genome hg38
...
```
