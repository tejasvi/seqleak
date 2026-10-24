<!--
"Leakage audit" section for a dataset card (README.md of a dataset repository or a Hugging Face dataset card).
Fill it automatically from the audit report:
    seqleak audit --train train.csv --dev dev.csv --test test.csv --out report.json
    python seqleak_card.py report.json --section leakage_audit.md --date YYYY-MM-DD
(several tasks: python seqleak_card.py task_*/report.json --section leakage_audit.md), then paste leakage_audit.md
into the card. To fill it by hand, replace each ${...} field:
  ${tool}       the tool that wrote the report: seqleak (posbench for reports of its former name)
  ${rules}      linter rules version, e.g. "linter rules v1.2.1"   ${version}   tool and version, e.g. "seqleak 0.4.1"
  ${date_line}  "Audited on YYYY-MM-DD." or nothing                ${verdict}   PASS, WARN or FAIL (worst check)
  ${summary}    the report's verdict summary (failing checks, warnings)
  ${table}      one row per check that ran: check | result | value [95% CI] | warn / fail
  ${baselines}  nearest-training lookup and best no-model rule, each with its 95% interval and chance level
  ${command}    the seqleak audit command that reproduces the report
Delete this comment before publishing.
-->

## Leakage audit

The splits in this repository were checked for data leakage with `${tool} audit` (${rules}; ${version}).
${date_line}
**Verdict: ${verdict}.** ${summary}.

${table}

${baselines}

How to read the verdict. FAIL marks a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of test items in the training data, negatives constructed from positives, artefact or motif shortcuts, or position confounds. WARN marks the same signals below that level, or a composition baseline that a model must beat. It is common on tasks without a documented leak, so treat it as a reason to look closer. PASS means that no check reached its warning threshold, and SKIP that the input lacks what the check needs (coordinates, distance fields, an aligner, or items with several sequences).

The linter reports only what its checks measure. It checks position only when distance fields are given, copies of other benchmarks only with --reference, membership in a reference genome only with --genome, and has no check for transductive arrangements; see "Known limitations" in the seqleak README.

To reproduce:

```
${command}
```
