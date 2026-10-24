### seqleak audit: Three tasks, original splits and cleaned versions

Task verdicts: 3 FAIL, 2 WARN, 2 PASS.

| task | verdict | failing checks | warnings | lookup [95% CI] (chance) | best rule | test items |
|---|---|---|---|---|---|---|
| gue__prom_core_all | **FAIL** | construction.segment, construction.synthetic_negatives, construction.genome_membership | overlap.exact, overlap.near, overlap.windows, overlap.homology, shortcut.kmer | -0.311 [-0.334, -0.290] (0.000) | 0.901 | 5,920 |
| gue__prom_core_all@resplit_union | **FAIL** | construction.synthetic_negatives, construction.genome_membership | shortcut.kmer, construction.segment | 0.055 [0.038, 0.071] (0.000) | 0.893 | 5,601 |
| gue__prom_core_all@fixneg_resplit | **WARN** | - | shortcut.position_motif | 0.049 [0.029, 0.067] (0.000) | 0.492 | 5,601 |
| nt__H3K4me3 | **FAIL** | overlap.near, overlap.windows, overlap.homology | - | 0.823 [0.804, 0.841] (0.000) | 0.823 | 3,680 |
| nt__H3K4me3@resplit | **PASS** | - | - | 0.046 [0.016, 0.078] (0.000) | 0.229 | 3,592 |
| gb__human_enhancers_cohn | **WARN** | - | overlap.windows, overlap.homology | 0.501 [0.489, 0.513] (0.500) | 0.681 | 6,948 |
| gb__human_enhancers_cohn@resplit | **PASS** | - | - | 0.493 [0.479, 0.505] (0.500) | 0.647 | 6,292 |

Failing checks:
- gue__prom_core_all: Segment-substitution negatives (positional twins) (`construction.segment`) 0.672 [0.660, 0.684]: 3981 of 5920 test items have a twin in train/dev; their label is the partner's label flipped for 0.935 of them; 0.747 of all items are in twin pairs, 0.795 of pairs with different labels
- gue__prom_core_all: Constructed negatives anywhere in the data (opposite-label twin pairs) (`construction.synthetic_negatives`) 0.723 [0.711, 0.734]: 4279 of 5920 test items are in twin pairs with different labels of a twin kind whose pairs disagree beyond chance (dinucleotide shuffles or segment substitutions of each other; 0.728 of all items); composition: 2 pairs, flip kappa 0.00; positional: 29612 pairs, flip kappa 0.59 (counted)
- gue__prom_core_all: Negatives absent from the reference genome (membership predicts the label) (`construction.genome_membership`) 0.901 [0.891, 0.911]: membership rule (fitted on train): test MCC 0.901 (CV 0.900); test AUROC 0.948; near-full-length members (membership >= 0.9) among test items: class 0 0.10, class 1 1.00
- gue__prom_core_all@resplit_union: Constructed negatives anywhere in the data (opposite-label twin pairs) (`construction.synthetic_negatives`) 0.699 [0.687, 0.711]: 3916 of 5601 test items are in twin pairs with different labels of a twin kind whose pairs disagree beyond chance (dinucleotide shuffles or segment substitutions of each other; 0.692 of all items); composition: 1 pairs, flip kappa 1.00; positional: 21799 pairs, flip kappa 0.83 (counted)
- gue__prom_core_all@resplit_union: Negatives absent from the reference genome (membership predicts the label) (`construction.genome_membership`) 0.893 [0.882, 0.904]: membership rule (fitted on train): test MCC 0.893 (CV 0.905); test AUROC 0.944; near-full-length members (membership >= 0.9) among test items: class 0 0.11, class 1 1.00
- nt__H3K4me3: Near-duplicates (k-mer containment) (`overlap.near`) 0.570 [0.554, 0.586]: 2098 of 3680 test items share >= 50% of their k-mers with one fit-split sequence
- nt__H3K4me3: Overlapping or shifted genome windows (`overlap.windows`) 0.930 [0.921, 0.938]: 3423 of 3680 test items share a collinear (overlapping or shifted) segment with a training/dev sequence; 0.930 of test items at a non-zero offset
- nt__H3K4me3: Homology (alignment) (`overlap.homology`) 0.611 [0.589, 0.632]: 1222 of 2000 sampled test items have a homologous training/dev sequence (BLASTN dc-megablast)

Linter rules v1.2. Reading: FAIL marks a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of test items in the training data, negatives constructed from positives, artefact or motif shortcuts, or position confounds. WARN marks the same signals below that level, or a composition baseline that a model must beat. It is common on tasks without a documented leak, so treat it as a reason to look closer. PASS means that no check reached its warning threshold, and SKIP that the input lacks what the check needs (coordinates, distance fields, an aligner, or items with several sequences).
