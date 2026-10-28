# Leak linter reference

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

## Checks

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

## How the rules are scored

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

## Reading the verdict

FAIL is a leak at the level that manual audits of DNA benchmarks count as a defect. WARN is the same signal below
that level, or a composition baseline that a model must beat. WARN is common on tasks without a documented leak, so
read it as a pointer to look. The linter reports only what its checks measure. It sees position confounds only when
distance fields are given (`--distance-cols`, or `--task-dir` for a gene-pair task), copies of another benchmark only
with `--reference`, and membership in a reference genome only with `--genome`. It has no check for transductive
arrangements.

## Cleaning a split

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

## Report cards and continuous integration

`report_card/seqleak_card.py` turns a `report.json` into a one-page report card in HTML and Markdown, or into a
"Leakage audit" section for a dataset card. It needs only the Python standard library. `templates/` holds a GitHub
Actions workflow that audits every task folder of a dataset repository and fails on a FAIL, the template of the
dataset-card section, and a guide for benchmark maintainers (`templates/HOWTO.md`). These files are in this
repository but not in the installed package.

## Known limitations

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

The accompanying paper (see "Citation" in the README) states for each of its linter results which frozen rule set
produced it.

In multi-class tasks the one-feature rules of the length, ambiguity, letter-case, composition and genome-membership
checks predict the majority class within quantile bins of the feature. Up to v1.2, the largest value of the feature
always shared the last bin with the values below it. When the feature takes only two values, as genome membership does
when every negative is absent from the genome and every positive is present, the rule had a single bin. Its
prediction was then constant, its MCC was 0 and the check passed. In v1.2.1 a point mass at the largest value gets a
bin of its own. Binary tasks, and features without such a point mass, give the same results as in v1.2. In the
seven-class task of the paper's spike-in simulation, the membership check of v1.2.1 fails all 24 inputs with
synthetic negatives (MCC 0.792 to 0.795), where v1.2 failed 9 of them, and it still passes the unmodified input and
the inputs with genome windows as negatives. `CHANGELOG.md` lists every change of this release.
