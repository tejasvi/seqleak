# seqleak

seqleak is a leak linter for DNA sequence-classification benchmarks: it looks for the leaks that let a no-model rule
score high, such as test sequences that are copies or relatives of training sequences, or negatives built from the
positives. `seqleak audit` runs its checks on a task and gives each a verdict. `seqleak clean` writes a version of the
split with less leakage. For gene-pair benchmarks, such as operons and protein interactions, and for tasks on distance
to a transcription start site, the package also computes a position-only baseline and model metrics with position held
fixed.

## Install

```
pip install seqleak                # Python 3.9 or later
```

numpy, pandas, scipy, scikit-learn and pyarrow are installed with it. The optional homology check runs when BLAST+
(`blastn` and `makeblastdb`) or MMseqs2 is on the PATH, and the genome-membership check needs minimap2 when it is run
with `--genome-method minimap2`. In a clone of this repository, `pip install -e .` installs the package for
development and `pytest tests` runs the unit tests.

## Audit a task

```
seqleak audit --train train.csv --dev dev.csv --test test.csv --out report.json
seqleak audit --data task.parquet --out report.json        # one table with sequence, label and split columns
```

With `--fail-on warn` or `--fail-on fail` the command exits with status 1 at that verdict, for use in continuous
integration. Two tiny tasks in `examples/tiny/` show whether an installation works. The second has planted copies of
training items in its test set:

```
cd examples/tiny
seqleak audit --train clean/train.csv --dev clean/dev.csv --test clean/test.csv --out clean.json --fail-on fail   # PASS, exit status 0
seqleak audit --train leaky/train.csv --dev leaky/dev.csv --test leaky/test.csv --out leaky.json --fail-on fail   # FAIL, exit status 1
```

The Markdown report of the leaky task, `leaky.md`, includes the lines below, where `...` marks lines left out. A fit
split is train or dev, and the next section groups the check ids.

```
**Verdict: FAIL**. leak checks failing: overlap.exact, overlap.near, overlap.windows; warnings: construction.segment.
...
Nearest-training lookup: mcc 0.381 [0.307, 0.455] (chance 0.000).
...
| check | status | statistic | value [95% CI] | warn / fail |
|---|---|---|---|---|
| overlap.exact | FAIL | share of test items with an exact or reverse-complement copy in a fit split | 0.150 [0.107, 0.206] | 0.005 / 0.050 |
...
| reference.copies | SKIP | no --reference given |  |  |
...
```

Inputs are FASTA, CSV, TSV or parquet files. Columns are detected by name (`sequence` or `seq`, `label`, and `split`
with the values train, dev and test) or set with `--seq-col`, `--label-col` and `--split-col`. Genomic coordinates
are used when the table has chrom, start and end columns, or ids of the form `chr1:100-600`.

The audit writes the report as JSON and as Markdown, and a table with one row per item (`<out>.items.parquet`). Each
check reports a statistic, its thresholds and a verdict. Every report also includes two baselines. The nearest-training
lookup, which test-only inputs skip, assigns each test item the label of its most similar training sequence. The best
no-model rule is selected on the test split.

## What the checks look for

| checks (report ids) | what they find | input they need |
|---|---|---|
| overlap (`overlap.*`) | exact and reverse-complement copies, near-copies, overlapping or shifted genome windows, copied pairs, homologs | a training split; BLAST+ or MMseqs2 for homologs |
| reference (`reference.copies`) | copies of another benchmark | `--reference` |
| construction (`construction.*`) | negatives built from positives (twins of positives), negatives absent from the organism's genome, unmatched sites (a fixed-position motif that one class carries and another lacks), libraries (constant segments that split the items into groups) | `--genome` for genome membership |
| shortcut (`shortcut.*`) | rules on length, N runs and padding, ambiguity codes, letter case and composition; windows made mostly of N; motifs at fixed positions or at the ends of variable-length items | classification labels for the rules (the shortcut rules are skipped for regression) |
| position (`position.*`) | position confounds, with a position-only model | `--distance-cols`, or `--task-dir` for a gene-pair task |

Composition rules (GC, CpG, k-mers) only warn, because composition can reflect biology. The shortcut rules are fitted
on the training split, chosen by cross-validation on it, and scored on the test split. A shortcut check reaches a
level only when the cross-validation score reaches it too. Test-only (zero-shot) inputs without a training split can
be audited: the overlap checks and the lookup are then skipped, and the rules are cross-validated within the test
items. [docs/checks.md](https://github.com/tejasvi/seqleak/blob/v0.4.1.post1/docs/checks.md) gives the statistic and
thresholds of every check, the definitions of twins, the input formats, the known limitations and run times.

## Reading the verdict

| verdict | meaning |
|---|---|
| FAIL | a leak strong enough that our manual audits of DNA benchmarks count it as a defect (see the accompanying paper) |
| WARN | the same signal at a weaker level, or a composition baseline that a model must beat; common on tasks without a documented leak, so treat it as a reason to look closer |
| PASS | below the WARN level |
| SKIP | the input lacks something the check needs, such as one of the inputs listed under "input they need" above |

The task's verdict is the worst of its check verdicts. A SKIP does not count, so a PASS says nothing about the checks
that were skipped. `--published-best` takes the task's best published score, in the units of `--metric`, which also
accepts `auroc`. Without it, most motif shortcuts that those audits count as leaks get only a WARN. With it, a motif
rule, or a length, N or case rule, that warns can fail the task, by the rule given in `docs/checks.md`, and the verdict
is then only as good as the published score given. The linter has no check for transductive arrangements.

## Clean a split

```
seqleak clean --data task.parquet --out-dir cleaned/ --mode resplit   # assign whole groups of related items to train, dev and test
seqleak clean --data task.parquet --out-dir cleaned/ --mode filter    # keep train; drop the dev and test items linked to it
```

Items are grouped by exact copies, near-copies, overlapping or shifted windows, and composition and positional twins.
When the input has coordinates, items can also be grouped by locus (`--group-by locus`, intervals closer than
`--locus-gap`) or by chromosome (`--group-by chromosome`). Resplit mode keeps every item and the split proportions, and
it stratifies the groups by label. The output contains the new split files, a manifest with the fate of every item and
the audit of the new split. Removing overlap does not remove constructed negatives: a cleaned split whose negatives are
shuffles of its positives still warns.

## Report cards and continuous integration

`report_card/seqleak_card.py` turns a `report.json` into a one-page card or into a "Leakage audit" section for a dataset
card. `templates/` contains a GitHub Actions workflow that audits every task folder of a dataset repository and fails on
a FAIL, and a guide for benchmark maintainers (`templates/HOWTO.md`). These files are in this repository but not in the
installed package.

## Gene-position metrics

On many bacterial gene-pair benchmarks, gene position alone (strand, intergenic distance, adjacency) predicts much of
the label, and on eukaryotic genomes distance to the transcription start site plays the same role in enhancer-gene and
variant-effect tasks. For a task of either kind, seqleak reports a position-only baseline, row zero, above the models.
For each model it reports the AUROC with position held fixed (computed only between items in the same position stratum),
what the model's score adds to the position baseline, and lower and upper bounds on the share of its above-chance AUROC
that comes from position. The lower bound is exact when position is a consequence of the labels, the upper bound when
position is a cause of them. Every number comes with a 95% bootstrap interval.

```python
import seqleak
df = seqleak.row_zero('path/to/task')     # items.parquet + scores/<model>.npy
print(seqleak.format_table(df))
```

For the task format, the definitions and the validation, see
[docs/position_metrics.md](https://github.com/tejasvi/seqleak/blob/v0.4.1.post1/docs/position_metrics.md). These
metrics come unchanged from posbench 0.3.2, which is archived with the paper that introduced them
(doi:10.5281/zenodo.23605239).

## Versions

seqleak was called posbench up to version 0.4.0.dev2. The `posbench` command still works: it prints a one-line notice
and then runs `seqleak`. Every report records the rule set that produced it. seqleak 0.4.1 uses rule set v1.2.1, which
changes some multi-class results from v1.2 (see `CHANGELOG.md` and the section "Rule versions" of `docs/checks.md`). The
accompanying paper (see "Citation") names the rule set behind each of its linter results.

## Citation

If you use seqleak, please cite this release's archive, which also contains the scripts and logs behind the accompanying
paper (doi:10.5281/zenodo.23717097), and that paper:

Tejasvi Singh Tomar. *Copy and relatedness leaks in genomic language model benchmarks are common and can be found and
largely removed automatically*. Preprint, 2026.

The gene-position metrics were introduced in Tejasvi Singh Tomar, *Gene position and relatedness confound genomic
language model benchmarks*, preprint, 2026. `CITATION.cff` has the citation metadata.

## License

Apache License 2.0 (`LICENSE`). Copyright 2026 Tejasvi S. Tomar.
