# Changelog of seqleak (formerly posbench)

seqleak is a leak linter for DNA sequence-classification benchmarks, with gene-position metrics.

## seqleak 0.4.1.post1 (rule set v1.2.1)

Documentation only: the code is the same as in 0.4.1, so reports still show version 0.4.1. The README now starts with
what seqleak does and how to run it. The README's reference sections moved to `docs/checks.md` and
`docs/position_metrics.md`.

## seqleak 0.4.1 (rule set v1.2.1)

Upgrading from posbench: install seqleak and change `import posbench` to `import seqleak`, because seqleak provides no
`posbench` module. The `posbench` command still works, and `POSBENCH_GENOME_REGISTRY` is still read when
`SEQLEAK_GENOME_REGISTRY` is not set. Reports now carry the schema ids `seqleak.audit/1` and `seqleak.clean/1` (reports
written by posbench carry `posbench.audit/1`). Results of multi-class tasks can change (see Fixed); binary tasks give
the same results as with rule set v1.2.

### Changed

- The package is now called seqleak (formerly posbench): the distribution, the import name and the command are
  `seqleak`.
- The `posbench` command prints a one-line notice and then runs `seqleak`.
- The environment variable `SEQLEAK_GENOME_REGISTRY` specifies the genome registry.

### Fixed

- Rule set v1.2.1 fixes the one-feature rule that the length, ambiguity, letter-case, composition and genome-membership
  checks use for multi-class tasks. This rule predicts the majority class within quantile bins of the feature. Up to
  v1.2, the largest value of the feature always shared the last bin with the values below it. A feature with two values
  therefore gave a single bin, a constant prediction and a Matthews correlation coefficient (MCC) of 0, and the check
  passed even when the feature separated the classes. In v1.2.1 a point mass at the largest value gets a bin of its
  own. Binary tasks, and features without such a point mass, give the same results as in v1.2. No threshold or check
  changed.
- Effect on the accompanying paper ("Copy and relatedness leaks in genomic language model benchmarks are common and can
  be found and largely removed automatically"): the bug is present in rule sets v1.0 to v1.2, and the paper names the
  rule set behind each of its linter results. In the paper's spike-in simulation, v1.2 passed a seven-class task whose
  negatives were all synthetic (MCC 0). On the same inputs, v1.2.1 fails all 24 inputs with synthetic negatives (MCC
  0.792 to 0.795), where v1.2 failed 9. Rule set v1.2.1 still passes the unmodified input and the inputs with genome
  windows as negatives.

### Added

- Tests: `tests/test_audit_v121.py` (the fix; three of its four tests fail with v1.2) and `tests/test_rename.py` (the
  `posbench` command and the former registry variable).
- Package metadata now lists the author and the source and archive links.

## Earlier versions (as posbench)

`docs/checks.md` describes the checks named here.

- posbench 0.4.0.dev2 (rule set v1.2): adds the checks `construction.genome_membership` and
  `shortcut.anchored_motif`, scores `construction.synthetic_negatives` on the test split, and cross-validates the rules
  within the test items when the training split has one class.
- posbench 0.4.0.dev1 (rule sets v1.1 and v1.1.1): adds `shortcut.lowinfo`, `construction.synthetic_negatives`,
  `construction.unmatched_site` and `construction.library`, the criterion relative to a published score, and test-only
  inputs; v1.1.1 changes docstrings only.
- posbench 0.4.0.dev0 (rule set v1.0): the first frozen rule set.
- posbench 0.3.2: the gene-position metrics, archived with the paper that introduced them
  (doi:10.5281/zenodo.23605239).

The section "Rule versions" of `docs/checks.md` lists the rule sets with their packages. The frozen copies of rule sets
v1.0 to v1.2 are in the accompanying paper's archive (doi:10.5281/zenodo.23717097).
