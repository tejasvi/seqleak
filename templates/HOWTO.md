# How to audit a benchmark for leakage and show the result

A short guide for benchmark authors and maintainers: audit the splits, read the verdict, fix what fails, and put the
result where users see it, in a report card, the dataset card or a continuous-integration check.

## 1. Install

```
pip install seqleak
```

seqleak needs Python 3.9 or later with numpy, pandas, scipy, scikit-learn and pyarrow, which are installed with it.
The optional homology check runs when BLAST+ (`blastn`, `makeblastdb`) or MMseqs2 is on the PATH.

## 2. Audit each task

```
seqleak audit --train train.csv --dev dev.csv --test test.csv --out report.json      # one file per split
seqleak audit --data task.parquet --split-col split --out report.json                # one table with a split column
seqleak audit --train pos.fa@1 neg.fa@0 --test test_pos.fa@1 test_neg.fa@0 --out report.json   # one FASTA per class
```

Columns are found by name (`sequence` or `seq`, `label`, and `split` with train, dev and test) or given with
`--seq-col`, `--label-col` and `--split-col`. Useful options:

* `--seq-col enhancer,promoter` for items with two sequences;
* `--reference other=other_benchmark.parquet` to find copies of another benchmark;
* `--genome genome.fa` to test whether membership in the reference genome of the task's organism gives the label
  (several genomes are given comma-separated);
* `--distance-cols distance_to_tss` to test a position confound;
* `--fail-on fail` to exit with status 1 on a FAIL.

The audit writes `report.json`, `report.md` and a per-item table, `report.items.parquet`, with each item's nearest
training item, near-duplicate group and twin flags.

## 3. Read the verdict

The task verdict is the worst check verdict.

* **FAIL**: a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of test items
  in the training data, negatives constructed from positives, artefact or motif shortcuts, or position confounds.
* **WARN**: the same signals below that level, or a composition baseline that a model must beat. WARN is common on
  tasks without a documented leak, so read it as a pointer to look.
* **PASS**: no check reached its warning threshold. **SKIP**: the input lacks what the check needs.

Every report also gives two baselines. The nearest-training lookup gives each test item the label of its most similar
training sequence. The best rule without a model is chosen on the test split, so it is an upper bound. A model that
does not clearly beat both has not shown that it learned more than the leak.

## 4. Fix what fails

* Copies, near-duplicates, overlapping windows and homologs (`overlap.*`): re-split so that related items stay in one
  split, with `seqleak clean --train train.csv --dev dev.csv --test test.csv --out-dir cleaned --mode resplit`, or
  keep the test set and drop its leaked items with `--mode filter`. `seqleak clean` audits the new split again
  (`audit_after.json`) and writes the fate of every item to `manifest.tsv`.
* Constructed negatives (`construction.shuffle`, `.segment`, `.synthetic_negatives`): re-splitting removes the twins
  across the split but not the construction, and `construction.synthetic_negatives` counts the test items in twin
  pairs anywhere in the data. Use genomic background windows matched on length and GC content as negatives.
* Negatives that are not genome sequence (`construction.genome_membership`, with `--genome`): the remedy is the same,
  negatives drawn from the reference genome. A re-split does not change this verdict either.
* Negatives without the positives' site (`construction.unmatched_site`): draw negatives that carry the site at the
  same position. If the site is what the task tests, report the site rule as a baseline.
* Pooled-library structure (`construction.library`): trim the constant segments (adapters, barcodes, backbones), or
  split and score within one library.
* Shortcuts (`shortcut.length`, `.ambiguous`, `.case`, `.lowinfo`, `.position_motif`, `.anchored_motif`): give both
  classes the same length, padding and letter case, drop windows made mostly of N, and vary where variable-length
  items start and end. If a motif is what the task tests, report the motif table as a baseline.
* Composition (`shortcut.composition`, `.kmer`) can be real biology: report the rule as the baseline to beat.

A new split is a new version of the benchmark. Publish it next to the original, keep the original for comparison with
earlier results, and say which version each score uses.

## 5. Report card

```
python seqleak_card.py report.json --html card.html --md card.md
python seqleak_card.py task_*/report.json --title "My benchmark" --html card.html --md card.md   # all tasks
```

`card.html` is a self-contained page that prints on one A4 sheet per task, and `card.md` fits a pull request or an
issue. `seqleak_card.py` (in `report_card/` of this repository) needs only the Python standard library.

## 6. Dataset card

```
python seqleak_card.py report.json --section leakage_audit.md --date 2026-10-22
```

Paste `leakage_audit.md` into the dataset card (README.md, or the Hugging Face dataset card below its YAML header).
`dataset_card/leakage_audit_section.md` is the blank template for filling by hand.

## 7. Continuous integration

1. Copy `ci/github/workflows/leakage-audit.yml` to `.github/workflows/` of the dataset repository, and copy
   `ci/seqleak_ci.py` and `ci/seqleak_card.py` to `.github/seqleak/`.
2. Set `DATA_ROOT` in the workflow to the folder that holds one sub-folder per task with `train`, `dev` and `test`
   files (CSV, TSV, parquet or FASTA, optionally gzipped; parquet shards such as `train-00000-of-00001.parquet` are
   read together). For other layouts, list the tasks in a file, one line per task with the name, a tab and the input
   arguments of `seqleak audit`, and call `seqleak_ci.py --tasks FILE`.
3. Run it once locally: `python .github/seqleak/seqleak_ci.py --root data --out leakage-audit`.

The check fails when a task's verdict reaches `FAIL_ON` (default `fail`). It writes a summary table to the job summary
and uploads every report and card as the `leakage-audit` artifact. Keep the seqleak version pinned, so that verdicts
stay comparable between runs.

With one thread and the homology check off, the three GUE task folders this workflow was tested on (22,291 to 36,799
sequences each) took 36 s of CPU time together with rule set v1.2, against 16 s with v1.0, and peak memory rose from
0.8 GB to 2.2 GB. Most of the extra time goes to the pooled-library check. With `--genome`, the membership check reads
the reference genome once per task: for the human genome this takes several minutes, for the yeast genome a few
seconds. Wall time on a busy machine varies more than CPU time, so leave headroom in the job's time limit.

## 8. What the linter does not see

It reports only what its checks measure: position confounds only with distance fields, copies of other benchmarks
only with `--reference`, and membership in a reference genome only with `--genome`. The membership check places items
of at least 30 nt, and spliced transcripts against genome windows fail it as a construction difference. There is no
check for transductive arrangements. Test-only inputs cannot show copies or twins whose partners were in a training
split. "Known limitations" in the seqleak README lists the rest.
