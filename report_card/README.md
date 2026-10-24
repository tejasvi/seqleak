# Report cards from seqleak audit reports

`seqleak_card.py` turns the `report.json` written by `seqleak audit` into a one-page report card. It needs only the
Python standard library (3.8 or later) and does not import seqleak, so it also runs where seqleak is not installed,
for example in a continuous integration (CI) step after the audit. It also reads reports written by posbench,
the former name of seqleak.

```
python seqleak_card.py report.json --html card.html --md card.md
python seqleak_card.py report.json --section leakage_audit.md --date 2026-10-22
python seqleak_card.py task_a/report.json task_b/report.json task_c/report.json --title "My benchmark" --html card.html
python seqleak_card.py --section-template > leakage_audit_template.md
```

Outputs:

* `--html`: a self-contained HTML page that prints as one A4 page for a single task. It loads no scripts, fonts,
  images or other files. It shows the verdict with the failing checks and four tiles: test items, the
  nearest-training lookup against chance, the best no-model rule, and check counts. Below them come the
  findings, a table of the other checks that ran, the checks that were skipped and why, and the rule set, inputs and
  command. Each FAIL is shown with its statistic, the linter's explanation and what to do about it, and each WARN
  with its explanation. Inputs are listed by file name and sha256.
* `--md`: the same card in GitHub-flavoured Markdown, with a table of every check, for pull requests, issues and CI
  job summaries.
* `--section`: a "Leakage audit" section for a dataset card (README.md or a Hugging Face dataset card). It has the
  verdict, a table of the checks that ran, the two baselines, how to read the verdict, limitations, and the command
  that reproduces the audit, with file paths reduced to file names.
* With several reports, the card starts with one row per report and lists the failing checks of each.

The card names the rule set, for example "linter rules v1.2.1", when the report records it, or when the thresholds
stored in the report are exactly those of a frozen rule set. For that comparison the card computes a sha256 of the
thresholds. Rule sets v1.2 and v1.2.1 have the same thresholds, so such a report is labelled "v1.2 or v1.2.1".
Reports made with other thresholds are labelled as such. With `--fail-on fail` (or `warn`) the command exits with
status 1 when any report has that verdict, which makes it usable as a CI gate.

## Reading a card

* FAIL: a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of test items in
  the training data, negatives constructed from positives, artefact or motif shortcuts, or position confounds.
* WARN: the same signals below that level, or a composition baseline that a model must beat. WARN is common on tasks
  without a documented leak, so read it as a pointer to look.
* PASS: no check reached its warning threshold. SKIP: the input lacks what the check needs (coordinates, distance
  fields, an aligner, items with several sequences).
* Nearest-training lookup: the score obtained by giving each test item the label of its most similar training
  sequence. A score far above chance means that test items have relatives in the training data.
* Best no-model rule: the best of the linter's rules on the test split. It is chosen on the test split, so it is an
  upper bound for rules without a model, not an estimate of one.

## Examples

`examples/` holds cards for three audited tasks, one each from GUE, the Nucleotide Transformer downstream tasks
(original version) and Genomic Benchmarks. They are made from the reports in `examples/reports/`, which were written
by posbench 0.4.0.dev2 with rule set v1.2 and the reference genome of each task's organism. These three tasks are
binary, so rule set v1.2.1 gives the same results. In the copies here, the command line and input paths are reduced
to file names and nothing else is changed.

| card | task | verdict on the original split |
|---|---|---|
| `gue__prom_core_all.html` / `.md` | GUE core promoter detection, all | FAIL: segment-substitution negatives, with twins across the split and within it, absent from the reference genome |
| `nt__H3K4me3.html` / `.md` | Nucleotide Transformer H3K4me3 (yeast) | FAIL: near-duplicates, overlapping windows and homologs across the split |
| `gb__human_enhancers_cohn.html` / `.md` | Genomic Benchmarks human enhancers (Cohn) | WARN: overlapping windows and homologs below the FAIL level |
| `three_tasks_before_after.html` / `.md` | the three tasks, original split and leak-reduced versions | FAIL to FAIL (re-split, which keeps the constructed negatives) and to WARN (fixneg_resplit: genomic negatives, re-split; a motif table at the start site warns), FAIL to PASS, WARN to PASS |

Each single-task card comes with its dataset-card section (`*_dataset_card_section.md`), and the before/after card
with a section for several tasks (`three_tasks_dataset_card_section.md`).
