"""seqleak audit: a leak linter for sequence-classification benchmarks (split overlap, cross-benchmark copies,
label-correlated shortcuts, constructed negatives, position confounds), and the shared loaders used by seqleak clean.

Python use::

    from seqleak.audit import load_task, run_audit
    task = load_task(train=['train.csv'], test=['test.csv'])
    report, items = run_audit(task)
    print(report['verdict'])
"""
from .io import InputError, load_reference, load_task
from .report import to_markdown, write_json
from .run import ALL_CHECKS, SCHEMA, run_audit

__all__ = ['ALL_CHECKS', 'SCHEMA', 'InputError', 'load_reference', 'load_task', 'run_audit', 'to_markdown',
           'write_json']
