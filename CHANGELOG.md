# Changelog

## seqleak 0.4.1 (linter rules v1.2.1)

- The package is now called seqleak (formerly posbench). The distribution, the import name and the command are
  `seqleak`. The `posbench` command still works: it prints a one-line notice and runs `seqleak`. Reports carry the
  schema ids `seqleak.audit/1` and `seqleak.clean/1`. The environment variable `SEQLEAK_GENOME_REGISTRY` names the
  genome registry, and the former `POSBENCH_GENOME_REGISTRY` is still read when it is not set.
- Rule set v1.2.1 fixes the one-feature rule of multi-class tasks, which the length, ambiguity, letter-case,
  composition and genome-membership checks use. This rule predicts the majority class within quantile bins of the
  feature. Up to v1.2, the largest value of the feature always shared the last bin with the values below it, so a
  feature with two values gave a single bin, a constant prediction and MCC 0, and the check passed even when the
  feature separated the classes. In v1.2.1 a point mass at the largest value gets a bin of its own. Binary tasks, and
  features without such a point mass, give the same results as in v1.2. No threshold or check changed.
- The defect is present in rule sets v1.0 to v1.2. The linter results of the accompanying paper were computed with the
  frozen rule sets v1.0, v1.1 and v1.2, as the paper states for each result, and rule set v1.2.1 was not used there.
  In the paper's spike-in simulation, v1.2 passed a seven-class task whose negatives were all synthetic (MCC 0). On the
  same inputs v1.2.1 fails all 24 inputs with synthetic negatives (MCC 0.792 to 0.795), where v1.2 failed 9, and it
  still passes the unmodified input and the inputs with genome windows as negatives.
- New tests: `tests/test_audit_v121.py` (the fix; three of its four tests fail with v1.2) and `tests/test_rename.py`.
- Package metadata now names the author and the source and archive links.
