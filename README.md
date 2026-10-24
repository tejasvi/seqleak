# seqleak

seqleak checks DNA sequence-classification benchmarks for leaks: copies, near-copies and relatives of test
sequences in the training data, negatives built from the positives, label-correlated artefacts such as length or
padding, and position confounds. `seqleak audit` reports a statistic, a fixed threshold and a verdict for each check,
and `seqleak clean` writes a split without the leaks it finds. For benchmarks of gene pairs, such as operons and
protein interactions, the package also computes a position-only baseline and model metrics with gene position held
fixed (see "Gene-position metrics" below).

seqleak was called posbench up to version 0.4.0.dev2. The `posbench` command still works: it prints a one-line
notice and then runs `seqleak`.

## Install

```
pip install seqleak                # Python 3.9 or later
```

numpy, pandas, scipy, scikit-learn and pyarrow are installed with it. The optional homology check runs when BLAST+
(`blastn` and `makeblastdb`) or MMseqs2 is on the PATH, and `--genome-method minimap2` needs minimap2. In a copy of
this repository, `pip install -e .` installs the package for development and `pytest tests` runs the unit tests.

Two tiny tasks in `examples/tiny/` check an installation. The second has planted copies of training items in its
test set:

```
cd examples/tiny
seqleak audit --train clean/train.csv --dev clean/dev.csv --test clean/test.csv --out clean.json --fail-on fail   # PASS, exit status 0
seqleak audit --train leaky/train.csv --dev leaky/dev.csv --test leaky/test.csv --out leaky.json --fail-on fail   # FAIL, exit status 1
```

## Leak linter

`seqleak audit` looks for the leaks that let a rule without a model score high on a sequence-classification task.
Each check reports a statistic, a fixed threshold, a 95% interval where one applies, example records and a verdict.
The verdict is PASS, WARN or FAIL, or SKIP when the input lacks what the check needs. The task verdict is the worst
check verdict. Two baselines come with every report: the nearest-training lookup, which gives each test item the label
of its most similar training sequence, and the best rule that uses no model.

```
seqleak audit --data task.parquet --out report.json                  # one table with sequence, label and split columns
seqleak audit --train train.csv --dev dev.csv --test test.csv --out report.json
seqleak audit --train pos.fa@1 neg.fa@0 --test test_pos.fa@1 test_neg.fa@0 --out report.json   # one FASTA file per class
seqleak audit --data epi.parquet --seq-col enhancer,promoter --out report.json                 # items with two sequences
seqleak audit --data task.parquet --reference other=other_benchmark.parquet --out report.json  # copies of another benchmark
seqleak audit --data loci.parquet --distance-cols distance_to_tss --group-col chrom --out report.json
seqleak audit --data task.parquet --genome hg38.fa.gz --out report.json          # membership in a reference genome
python -m seqleak audit ...                                           # the same, without the console script
```

Inputs are FASTA, CSV, TSV or parquet files. In FASTA headers, tokens of the form `key=value` become columns, for
example `label=1 split=test`. Columns are found by name (`sequence` or `seq`, `label`, and `split` with the values
train, dev and test) or named with `--seq-col`, `--label-col` and `--split-col`. Genomic coordinates are used when the
table has chrom, start and end columns, or ids of the form `chr1:100-600`.

The report is written as JSON with the schema id `seqleak.audit/1`, as Markdown, and as a table with one row per item,
`<out>.items.parquet`. That table holds each item's cross-validation fold, near-duplicate group, nearest training
item, k-mer containment, twin flags and features. Reports written by posbench carry the schema id `posbench.audit/1`.

### Checks

| check | what it measures | warn / fail |
|---|---|---|
| overlap.exact | test items with an exact or reverse-complement copy in train or dev | 0.005 / 0.05 |
| overlap.near | test items sharing >= 50% of their canonical 21-mers with one training sequence | 0.01 / 0.10 |
| overlap.windows | test windows overlapping a training window (coordinates), or sharing a collinear run of k-mers covering >= 20% of the item (shifted windows) | 0.01 / 0.25 |
| overlap.pair | items with several sequences whose every element is copied from one training item | 0.01 / 0.25 |
| overlap.homology | test items with a training hit of >= 80% identity over >= 50% (BLASTN or MMseqs2, when installed) | 0.05 / 0.25 |
| reference.copies | copies between this task and another benchmark (both directions) | 0.01 / 0.05 |
| shortcut.length, .ambiguous, .case | one-feature rules on length, N runs and padding, ambiguity codes, lowercase; test MCC | 0.1 / 0.3 |
| shortcut.composition, .kmer | GC, CpG and naturalness rules; class-conditional order-3 Markov rule; test MCC (composition can be biology: warning only) | 0.5 / - |
| shortcut.position_motif | a table of <= 4 (position, k-mer) indicators chosen on train (fixed-length tasks); test MCC | 0.4 / 0.7 |
| shortcut.anchored_motif | the motif table on the first and last 100 bases of variable-length items (or on the 200 bases around a site column, `--anchor-col`); test MCC | 0.4 / 0.7 |
| shortcut.lowinfo | test items with less than half A, C, G or T (windows made of N); they are left out of the overlap checks | 0.005 / 0.05 |
| construction.shuffle | test items with a composition twin in train with the opposite label (same length, dinucleotide counts and end bases, as a dinucleotide shuffle leaves them) | 0.01 / 0.05 |
| construction.segment | test items with a positional twin in train with the opposite label (segments kept in place, identity >= 0.4) | 0.01 / 0.05 |
| construction.synthetic_negatives | test items in composition or positional twin pairs with different labels anywhere in the data, counted for a twin kind when its pairs disagree beyond chance (kappa >= 0.4, at least 10 pairs): constructed negatives, also inside one split | 0.05 / 0.25 |
| construction.genome_membership | with `--genome` (FASTA files or registry names): each item's share of up to 32 canonical 21-mers found in the reference genome(s); a rule on that share fitted on train; test MCC; negatives that were generated rather than drawn from the organism's genome are not members; items need >= 30 nt | 0.2 / 0.5 |
| construction.unmatched_site | other-class test items that lack the fixed-position k-mer (k = 1-3) found in >= 95% of one class on train (fixed-length tasks up to 2 kb; WARN at most with more than two classes) | 0.05 / 0.25 |
| construction.library | constant segments (>= 5 canonical 21-mers present together in >= 5% of items, at one offset) that split the items into libraries; the label predicted from the library (fitted on train) and scored on test: MCC, or the square root of R2 for regression | 0.1 / 0.3 |
| position.distance, .row_zero | a position-only model on supplied distance fields, or the row zero of a gene-pair task directory (`--task-dir`); test AUROC | 0.6 / 0.75 |

### How the rules are scored

Rules are fitted on the training split. The feature or table of each rule is chosen by 5-fold cross-validation on
the training split. The folds keep exact copies and twins together and are written to the per-item table. The
reported statistic is the rule's score on the test split, and a shortcut check reaches a level only when the
cross-validation score reaches it too.

For binary labels whose minority class is below 20% of the training split, a shortcut rule is also scored by its
AUROC against (1 + threshold) / 2, and the more severe status counts. Under strong imbalance the MCC of a rule stays
low even when the rule separates the classes well.

Twins count as constructed negatives when the test label is the opposite of the twin's more often than chance allows,
with Cohen's kappa of at least 0.4 against the agreement of random label pairs and at least 10 covered items. Twins
that share their labels give a warning, since they point to relatedness at fixed offsets. The AUROC criterion of the
motif check fails a task only when `construction.unmatched_site` also reaches its FAIL level. A table that reads a
splice-site consensus from decoy negatives that carry it therefore gives a WARN.

With `--published-best X`, given in the units of `--metric` (which also accepts `auroc`), a motif rule that warns
fails the task when it reaches 0.75 of the best published score above chance. For a length, N or case rule the
fraction is 0.5.

Test-only (zero-shot) inputs without a training split can be audited. The overlap checks and the lookup are then
skipped, the rules are cross-validated within the test items, and the twin and construction checks use all the items
given. The same cross-validation within the test items is used when the training split has a single class.

The membership check needs the reference genome of every organism the data were drawn from: for a task with two
species, both genomes. A membership rule also separates spliced transcripts from genome windows, and it reports this
as a construction difference between the classes.

`--thresholds FILE` (JSON) changes the thresholds, and the thresholds used are copied into every report. With
`--fail-on warn` or `--fail-on fail` the command exits with status 1 at that verdict, for use in continuous
integration.

### Reading the verdict

FAIL is a leak at the level that manual audits of DNA benchmarks count as a defect. WARN is the same signal below
that level, or a composition baseline that a model must beat. WARN is common on tasks without a documented leak, so
read it as a pointer to look. The linter reports only what its checks measure. It sees position confounds only when
distance fields are given (`--distance-cols`, or `--task-dir` for a gene-pair task), copies of another benchmark only
with `--reference`, and membership in a reference genome only with `--genome`. It has no check for transductive
arrangements.

### Cleaning a split

`seqleak clean` writes a version of the split with less leakage, and a manifest of every change:

```
seqleak clean --data task.parquet --out-dir cleaned/ --mode filter    # keep train; drop dev and test items linked to it
seqleak clean --data task.parquet --out-dir cleaned/ --mode resplit   # deal whole groups to train, dev and test
```

Items are joined into groups by exact copies, near-duplicates, overlapping or shifted windows, and composition and
positional twins. When coordinates exist, they can also be grouped by locus (`--group-by locus`, intervals closer than
`--locus-gap`) or by chromosome (`--group-by chromosome`). Filter mode keeps the test set without its leaked items.
Resplit mode keeps every item and the split proportions, and it stratifies the groups by label.

The output directory holds the new split files, with the input columns and the original rows. `manifest.tsv` gives
for every item its original split, its new split or `dropped`, its group, the reason and the linked item.
`clean.json` records the settings, counts and checksums, and `audit_after.json` is the audit of the new split.
Removing overlap does not remove constructed negatives: a cleaned split whose negatives are shuffles of its positives
still warns in `construction.shuffle`.

### Report cards and continuous integration

`report_card/seqleak_card.py` turns a `report.json` into a one-page report card in HTML and Markdown, or into a
"Leakage audit" section for a dataset card. It needs only the Python standard library. `templates/` holds a GitHub
Actions workflow that audits every task folder of a dataset repository and fails on a FAIL, the template of the
dataset-card section, and a guide for benchmark maintainers (`templates/HOWTO.md`). These files are in this
repository but not in the installed package.

### Known limitations

- Without `--published-best`, a fixed-position motif table must reach MCC 0.7 to fail a task (AUROC 0.85 when the
  minority class is below 20%, and then only with an unmatched site). Most motif shortcuts that manual audits count as
  leaks get only a WARN. With `--published-best`, the verdict is only as good as the published score given.
- The splice-site condition costs a FAIL when the positives carry their consensus in fewer than 95% of items.
- Test-only inputs cannot show copies or twins whose partners were in a training split.
- The membership check needs the right reference genome. With a wrong or missing reference it is uninformative: every
  item is a non-member, or the check is skipped. Items shorter than 30 nt are not placed. Spliced transcripts are
  only partly found in a genome, so transcripts against genome windows fail as a construction difference even when no
  negative was generated. The optional minimap2 method was not part of the validation.
- `construction.unmatched_site` needs fixed-length sequences up to 2 kb, and `construction.library` needs a constant
  segment of at least 21 bp in at least 5% of items. Transductive arrangement leaks have no check.
- For regression labels the shortcut rules are skipped. The overlap and library checks run.
- The nearest-training lookup uses exact 21-mers. On short, diverged sequences it is weaker than a lookup by
  alignment.
- In multi-class tasks, a largest feature value held by fewer than 1/64 of the training items still shares the last
  quantile bin of the one-feature rules, as does any value below one quantile step.

With one core and the homology check off, an audit took 1.4 s for 2,382 sequences of 101 bp, and 75 s with 3.9 GB
peak memory for 289,061 sequences (116 million bases). These times were measured with rule set v1.0. On three GUE
tasks, rule set v1.2 used about twice the CPU time of v1.0. The homology check (BLAST on 2,000 test sequences) adds
up to 12 minutes per task. The membership check reads the reference genome once, which took a median of 6.4 minutes
for the human genome.

## Rule versions

The checks and thresholds form a versioned rule set, and every report records the rule set that produced it. A frozen
rule set does not change, and a change makes a new version.

| rules | package | change |
|---|---|---|
| v1.0 | posbench 0.4.0.dev0 | first frozen rule set |
| v1.1 | posbench 0.4.0.dev1 | adds `shortcut.lowinfo`, `construction.synthetic_negatives`, `construction.unmatched_site` and `construction.library`, the criterion relative to a published score, and test-only inputs |
| v1.1.1 | posbench 0.4.0.dev1 | docstrings only; results as for v1.1 |
| v1.2 | posbench 0.4.0.dev2 | adds `construction.genome_membership` and `shortcut.anchored_motif`; scores `construction.synthetic_negatives` on the test split; cross-validates the rules within the test items when the training split has one class |
| v1.2.1 | seqleak 0.4.1 | fixes the multi-class one-feature rule (below); no threshold or check changed |

The linter results of the accompanying paper (see "Citation") were computed with the frozen rule sets v1.0, v1.1 and
v1.2, as the paper states for each result. Rule set v1.2.1 was made after the paper.

In multi-class tasks the one-feature rules of the length, ambiguity, letter-case, composition and genome-membership
checks predict the majority class within quantile bins of the feature. Up to v1.2, the largest value of the feature
always shared the last bin with the values below it. When the feature takes only two values, as genome membership does
when every negative is absent from the genome and every positive is present, the rule had a single bin. Its
prediction was then constant, its MCC was 0 and the check passed. In v1.2.1 a point mass at the largest value gets a
bin of its own. Binary tasks, and features without such a point mass, give the same results as in v1.2. In the
seven-class task of the paper's spike-in simulation, the membership check of v1.2.1 fails all 24 inputs with
synthetic negatives (MCC 0.792 to 0.795), where v1.2 failed 9 of them, and it still passes the unmodified input and
the inputs with genome windows as negatives. `CHANGELOG.md` lists every change of this release.

## Gene-position metrics

Many bacterial benchmarks ask about genes that are near each other (operons, protein interactions, functional
links). Gene position alone (strand, intergenic distance, adjacency) predicts much of what these benchmarks label.
On eukaryotic genomes, distance to the transcription start site plays the same part in enhancer-gene and
variant-effect tasks.
seqleak adds four things to a benchmark table:

1. **Row zero**: a position-only baseline (logistic regression on spline bases of the position features, fitted with
   genome-grouped cross-validation), reported above the models.
2. **Position held fixed**: each model's AUROC computed only between items that share their gene position
   (within strata).
3. **Gain over position**: what a model's score adds to the position baseline.
4. **Position share**: a lower bound, `share`, and an upper bound, `share_upper`, on the share of a model's
   above-chance AUROC that comes from gene position. The lower bound is exact when position is a consequence of the
   labels, the upper bound when position is a cause of them.

Every number comes with a 95% bootstrap interval over genomes (or over blocks of neighbouring items when a task has
few genomes). seqleak is written in pure numpy, pandas, scipy and scikit-learn and is vectorised; a 16-million-pair
task runs on one machine.

These metrics come unchanged from posbench 0.3.2, which is archived with the paper that introduced them
(doi:10.5281/zenodo.23605239).

### Quick start

```python
import seqleak
df = seqleak.row_zero('path/to/task')     # items.parquet + scores/<model>.npy, see "Task format"
print(seqleak.format_table(df))           # row zero, AUROC, within strata, gain, share [lower, upper]
df = seqleak.evaluate(y, genomes, features, {'model': scores})    # the same on arrays in memory
```

The table has one row per scorer: row zero (the position model), the distance rule for pair tasks, then each
model. Its columns:

* `auroc`, `auprc`: per genome, averaged over genomes.
* `within`: the AUROC within position strata.
* `gain`: the out-of-fold AUROC of position + model minus that of position. `gain_p` is its Wilcoxon p-value over
  genomes.
* `share`, `share_upper`: the lower and upper bounds of the position share.

Each of these except `gain_p` has 95% limits in the columns ending in `_lo` and `_hi`. The columns `n_items`, `n_pos`
and `n_genomes` hold the counts, and `df.attrs` holds the settings used.

### Integration examples

* `examples/bacbench_operon.py`: BacBench's long-read operon task, from its gene table and a model's pairwise scores.
* `examples/dgeb_operonic_pair.py`: DGEB's operonic pair tasks, with protein locations from NCBI's `fasta_cds_aa`
  file of each genome record.
* `examples/dnalongbench_distance.py`: distance-to-TSS tasks (DNALongBench enhancer-target genes and eQTL, the
  Long-Range Benchmark's causal eQTL), with distance as row zero.
* `examples/share_bounds.py`: the two share bounds on simulated genomes with a known answer (the table under "Why a
  bound").
* `examples/share_definitions.py`: three alternative share definitions on synthetic data (see "Alternative
  definitions").

### Other calls

Tables with other column names or several sub-tasks:

```python
seqleak.row_zero(task, by=['dataset'])                            # one block of rows per dataset
seqleak.row_zero(task, query="dataset == 'yeast_two_hybrid'")     # a subset
seqleak.row_zero(task, position_columns=['distance_to_tss'])      # any position features (prognostic strata)
```

**Per-gene tasks.** The `'gene'` preset (position model and strata) includes gene length, which is a property of
the gene rather than of its layout. With it, row zero, the gain over position and the position share count gene
length as position. For layout only, pass the layout columns. They must have no missing values, so the list below
leaves out `gap_up` and `gap_down`, which are missing at contig ends:

```python
seqleak.row_zero(task, position_columns=['strand', 'replication_strand', 'dist_origin', 'directon_pos',
                                          'directon_size', 'operon_rule', 'density'])   # gene layout, no length
```

The same on arrays already in memory:

```python
from seqleak import Resample, auroc, within_auc, position_share_bound, gain_over_position, position_model, make_strata
strata = make_strata(features)                          # features: DataFrame of position features
rs = Resample(genomes, method='cluster', n_boot=2000)   # one plan for every statistic -> paired intervals
raw = auroc(y, score, genomes, resample=rs)
within = within_auc(y, score, strata, genomes, resample=rs)
share = position_share_bound(y, score, strata, genomes, resample=rs)   # .estimate, .ci, .info['share_upper']
pos = position_model(features, y, genomes)              # out-of-fold, folds grouped by genome
gain = gain_over_position(y, pos, position_model(features, y, genomes, extra=score), genomes, resample=rs)
print(raw, within, share, gain, sep='\n')               # estimate [95% CI]
```

### Task format

A task directory holds

* `items.parquet`: one row per evaluated item with `genome_id`, `contig`, `i`, `j`, `label` (0/1), optional `split`,
  and the position features. `i` and `j` are 0-based gene indices within the contig, and `j = -1` for per-gene tasks.
* `scores/<model>.npy`: one score per item, aligned with `items.parquet` (zero-shot score or out-of-fold probe
  prediction), and `scores/README` saying how each was made.
* `TASK.md`: source, label construction, the benchmark's own protocol and metric, published numbers.

`seqleak.features.pair_features(coords, i, j)` computes the pair features: `genes_apart`, `same_strand`, `gap` (bp
between the two genes, negative for overlaps), `max_intervening_gap` and `strand_switches`.
`seqleak.features.gene_features(coords, origin=...)` computes the gene features: strand, length, gaps and rule scores
to the neighbours, directon size and position, local gene density, distance to the replication origin, and
leading/lagging strand. `seqleak.features.origin_from_gc_skew(sequence)` locates the origin when it is not known.

If the table has a `split` column with `test` items, those are evaluated. The position models are fitted on `train`
items when every model has scores for them, otherwise by genome-grouped cross-validation within the test genomes.
The folds are 5 genome-grouped folds for more than 10 genomes, leave-one-genome-out for 2-10 genomes, and five
contiguous chunks of the genome for a single genome. In the single-genome case the folds split a genome. Every metric
of out-of-fold predictions (row zero, the gain over position) then compares items only within their fold: AUROC
within folds is weighted by positives x negatives and AUPRC within folds by positives. `df.attrs['fold_stratified']`
is true when metrics are computed within folds. Pooling the predictions of different fold models biases the AUROC
down, because a fold rich in positives was trained on fewer (Forman and Scholz 2010; Airola et al. 2011). The bias is
largest for coarse predictors such as a position model.

### Definitions

Notation: y label (1/0), s model score, g genome, k position stratum, P_g / N_g positives / negatives of genome g.
Metrics are computed per genome and averaged over genomes (the BacBench convention) unless `aggregate='pooled'`.

* **AUROC**: the benchmark's metric on the model's score. **AUPRC**: average precision (step-wise, as in
  scikit-learn).
* **Within-position AUROC**: positives are compared only with negatives of the same genome and stratum,
  `within_g = sum_k U_gk / sum_k P_gk N_gk`, i.e. stratum AUROCs weighted by positives x negatives
  (U = Mann-Whitney count, ties 1/2). A score that is constant within each stratum scores exactly 0.5. A score that
  knows only position is constant within strata only if the strata fix position exactly. With the binned strata used
  here some position signal remains inside each stratum, so a position-only score scores near 0.5 rather than at it.
  Row zero reports, for each table, how near 0.5 the position model itself comes: it scores 0.513 within strata on
  BacBench's long-read operon task and 0.534 on its zero-shot STRING interaction task.
* **Strata** (`seqleak.strata`). Neighbouring pairs are stratified by strand state x 17 gap bins. Any pairs are
  stratified by 11 genes-apart bins x strand switches (0-3+) x 9 bins of the largest intervening gap x strand state.
  Genes (`'gene'`) are stratified by directon position x leading/lagging x distance to origin x gene-length quintile.
  Gene length is a gene-level covariate, not layout (see "Per-gene tasks" above). For any features, `'prognostic'`
  uses 20 quantile bins of the out-of-fold position-model prediction within each genome. A custom scheme is a dict
  of column -> bins.
* **Position model** (row zero): a logistic regression on cubic B-spline bases of the position features, with
  separate splines for same- and opposite-strand pairs, fitted out-of-fold over genome-grouped folds. It is fitted to
  convergence with a second-order solver (see "Validation"). `model='gbm'` uses histogram gradient boosting instead.
* **Gain over position**: the per-genome AUROC of the position + model fit minus that of the position fit, averaged
  over genomes. Both fits are out-of-fold with the same folds. A Wilcoxon signed-rank test over genomes is reported
  with 6 or more genomes.
* **Position share**: see the next section.

### Position share

`share` is a lower bound on the fraction of a model's above-chance AUROC that comes from gene position (0 means none,
1 means position alone), and `share_upper` is the matching upper bound. If position is a consequence of the labels,
as for operons, where co-transcription shapes gene spacing, the true share equals `share`. If position is a cause of
the labels, as for STRING links scored from gene neighbourhood, it equals `share_upper`. If both hold in part, it
lies between them.

Read `share` as "at least this much of the model's performance is gene position". A small `share` with a large
`share_upper` means the answer depends on how the labels were made, so report both.

#### Definition

Let raw be the model's AUROC (per genome, averaged over genomes). Two AUROCs leave out the position information of
the score. Both are computed from the scores and position strata alone:

* **held fixed**: within = the AUROC within position strata (pairs compared only inside a stratum).
* **equalized**: auroc_eq = the AUROC after each score is replaced by its percentile inside its (genome, stratum)
  cell. This makes the score identical in distribution across positions and keeps its ranking inside every stratum
  (`seqleak.metrics.equalized_score`).

Then

    share_within    = 1 - (within   - 1/2) / (raw - 1/2)
    share_equalized = 1 - (auroc_eq - 1/2) / (raw - 1/2)
    position share  = clip(min(share_within, share_equalized), 0, 1)       (column `share`)
    upper bound     = clip(max(share_within, share_equalized), 0, 1)       (column `share_upper`)

`seqleak.metrics.position_share_bound` computes them, and they are the `share` and `share_upper` columns of
`row_zero`.

#### Why a bound

Think of a score as a position part plus a position-free part. Its true position share is the fraction of
raw - 1/2 it would lose without the position part. No single statistic of (score, position, label) recovers this in
every case, because the answer depends on whether position is a consequence or a cause of the label:

* **Position is a consequence of the label** (the operon case above). A model's biological signal is then
  independent of position given the label. Holding position fixed is the right counterfactual, so share_within is
  exact. Equalizing removes label-driven differences between strata, so share_equalized is too high.
* **Position is a cause of the label** (the STRING neighbourhood case above). The biological signal is independent of
  position. Equalizing is the right counterfactual, so share_equalized is exact. Comparisons within strata are less
  diluted than raw comparisons (the AUROC is not collapsible), so share_within is too low.

The true share lies between the two in both cases, so the minimum is a lower bound and the maximum an upper bound.

The table below comes from simulated genomes with a known answer: `seqleak.synthetic.simulate_pairs_consequence`
simulates the first case and `seqleak.synthetic.simulate_pairs` the second. Each mixture is lam x position part +
(1 - lam) x position-free part. Each case has 30 genomes of about 1,500 neighbouring pairs, and the strata are
strand x 17 gap bins. `examples/share_bounds.py` prints the table:

```
position is a CONSEQUENCE of the label (true = within)
score                     AUROC within     eq |   true s_within  s_equal explained |  BOUND  upper
null: oracle (Y + noise)  0.688  0.684  0.579 |  0.000    0.022    0.581     0.579 |  0.022  0.581
null: biology Z = Y+noise 0.761  0.761  0.611 |  0.000    0.002    0.573     0.569 |  0.002  0.573
mixture lam=0.25          0.849  0.761  0.612 |  0.251    0.252    0.680     0.709 |  0.252  0.680
mixture lam=0.5           0.916  0.761  0.612 |  0.373    0.373    0.731     0.813 |  0.373  0.731
mixture lam=0.75          0.943  0.762  0.613 |  0.411    0.409    0.746     0.866 |  0.409  0.746
position only             0.914  0.521  0.513 |  1.000    0.948    0.968     0.997 |  0.948  0.968
position is a CAUSE of the label (true = equalized)
null: z independent       0.650  0.708  0.647 |  0.000   -0.390    0.017     0.009 |  0.000  0.017
mixture lam=0.25          0.742  0.709  0.648 |  0.381    0.136    0.390     0.397 |  0.136  0.390
mixture lam=0.5           0.829  0.710  0.648 |  0.545    0.360    0.548     0.634 |  0.360  0.548
mixture lam=0.75          0.844  0.710  0.648 |  0.565    0.389    0.569     0.799 |  0.389  0.569
position only             0.798  0.527  0.518 |  1.000    0.909    0.940     0.992 |  0.909  0.940
```

#### Properties

1. **Reference values.** The share is 0 for a score that carries no position information. It is 0.002 for a score
   that is independent of position given the label, and 0 (clipped from -0.390) for a score that is independent of
   position when position causes the labels. The oracle's 0.022 is sampling noise (its within-strata AUROC is 0.684
   against a raw AUROC of 0.688). The share is about 1 for a function of position: 0.91 to 0.95 here, the remainder
   being position signal left inside the strata (17 gap bins). Row zero reports the share of the position model's
   own predictions as a check of the strata. The share is monotone in the weight of the position part (0.002, 0.252,
   0.373, 0.409, 0.948 in the first case and 0, 0.136, 0.360, 0.389, 0.909 in the second).
2. **Bounded** in [0, 1] by construction. The flag `clipped_low` means that the minimum was below 0, for example for
   a position-independent score when position causes the labels. The flag `clipped_high` means that it was above 1:
   the part of the score that varies within strata ranks against the labels, for example with a within-strata AUROC
   below 0.5.
3. **Stable near chance.** The share is not reported when raw - 1/2 <= 0.02 (flag `raw_near_chance`). Just above the
   margin the interval is wide: at AUROC 0.535 the share is 0.000 [0.000, 0.093]. If more than 2.5% of bootstrap
   replicates have raw <= 1/2, no interval is given (flag `ci_unbounded`).
4. **Rank-based.** The share depends only on the ranks of the scores, so it is invariant to monotone transforms of the
   score.
5. **Intervals.** They come from the same sufficient statistics as the other columns: per-genome Mann-Whitney counts
   of the raw, within and equalized AUROCs, resampled with the table's shared bootstrap plan. For the genome bootstrap
   the resampled counts give exactly the re-scored values (see "Intervals").
6. **Inputs.** Per-item scores, labels, genomes and position strata only.

#### Edge cases

* AUPRC has a prevalence-dependent chance level that differs between strata, so the share is defined on the AUROC
  scale only. AUPRC is reported raw and as a gain.
* Multi-class labels: compute it for each class against the rest (or from the macro averages of raw, within and
  equalized AUROC). Continuous labels: the same construction with concordance indices,
  `concordance(y, s, groups=g)`, `concordance(y, s, strata, g)` and `concordance(y, equalized_score(s, strata, g), groups=g)`.
* Few genomes: the block bootstrap resamples blocks of neighbouring items, and the percentiles of the equalized score
  are computed once on the full data.

#### Alternative definitions

* **Within share alone**, 1 - (within - 1/2) / (raw - 1/2), is `seqleak.metrics.position_share`, kept for
  compatibility. It is negative, down to -1.1, for a score that ignores position when position causes the labels.
* **Explained share** (`seqleak.metrics.explained_share`) splits raw - 1/2 = sum_k y_k c_k / (P N), with c_k the
  centred rank of the score, into the part carried by the stratum's leave-one-out positive rate and the rest. It is
  0 for a position-independent score when position causes the labels but 0.57 for one when position is a consequence
  of them, and it overstates mixtures in both cases (column `explained` in the table above).
* **Gain share**, 1 - gain / (raw - 1/2), is 0.52-0.82 for a score with no position information, because AUROC gains
  are not additive (`examples/share_definitions.py`).

### Intervals

`seqleak.bootstrap.Resample` draws one resampling plan that every statistic of a table shares, so intervals of
differences and ratios are paired. `method='cluster'` resamples genomes and is the default for >= 10 genomes.
`method='block'` resamples blocks of 50 consecutive items within each genome (at least 20 blocks per genome).
Statistics are computed from sufficient statistics, not by re-scoring items. The cluster bootstrap uses per
(genome, stratum) Mann-Whitney counts, which cost O(genomes) per replicate. The block bootstrap uses the matrices
`M[u, v] = sum_k U(positives of block u, negatives of block v)` and `D[u, v] = sum_k P_uk N_vk`, so that a replicate
with block multiplicities c has AUROC `c'Mc / c'Dc`. Both equal the values obtained by re-scoring the resampled items
(checked in the tests). The intervals are percentile intervals, 95% by default.

### Validation

seqleak reproduces, from their inputs and to the precision printed in the paper, the long-read operon and zero-shot
PPI tables of the final version of *A simple distance rule beats genomic language models at operon prediction*
(doi:10.5281/zenodo.23484461). `posbench/VALIDATION.md` in the archive doi:10.5281/zenodo.23605239 says which values
were compared and how closely they agree. The tests (`tests/test_metrics.py`, `tests/test_features_baseline.py`)
check the metrics against reference implementations and the bootstrap against re-scoring the resampled items. They
run without the paper's data.

For the PPI table, `spec='pair'`, the default, fits the position model to convergence. It gives the same fit on every
machine and thread count, and its position-model AUROC matches the paper's final table. `spec='ppi_v1'` keeps the
solver settings of the paper's original code. That fit stops before convergence and depends on the number of threads,
so it does not reproduce the final table. For the long-read operon table, `spec='operon_v1'` uses the paper's solver
settings and reproduces the table's position model to the printed precision. The 'v1' in both names refers to this
paper.

## Citation

If you use seqleak, please cite the archive of this release, which also holds the scripts and logs behind the paper
(doi:10.5281/zenodo.23605245), and the paper:

Tejasvi Singh Tomar. *Copy and relatedness leaks in genomic language model benchmarks are common and can be found and
largely removed automatically*. Preprint, 2026.

The gene-position metrics were introduced in Tejasvi Singh Tomar, *Gene position and relatedness confound genomic
language model benchmarks*, preprint, 2026. `CITATION.cff` has the citation metadata.

## License

Apache License 2.0 (`LICENSE`). Copyright 2026 Tejasvi S. Tomar.
