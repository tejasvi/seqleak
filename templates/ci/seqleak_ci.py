#!/usr/bin/env python3
"""seqleak_ci: run seqleak audit on every task of a dataset repository and gate on the verdicts.

Tasks are found under --root: every folder holding a train file and a test file (and optionally a dev file), named
train / dev, valid, val, validation / test, with any suffix after "-", "_" or "." (train.csv, test.parquet,
train-00000-of-00001.parquet, dev.tsv.gz, test.fa ...). Shards of one split are audited together. Other layouts are
listed in a --tasks file, one task per line: NAME, a tab, then the input arguments of seqleak audit, for example
    enhancers<TAB>--data data/enhancers.parquet --split-col split
    promoters<TAB>--train data/train_pos.fa@1 data/train_neg.fa@0 --test data/test_pos.fa@1 data/test_neg.fa@0

For each task: OUT/<task>/report.json (+ report.md, report.items.parquet) from seqleak audit, and card.md / card.html
from seqleak_card.py (same folder as this script). For all tasks: OUT/summary.md, OUT/summary.html and
OUT/leakage_audit_section.md (a dataset-card section). When GITHUB_STEP_SUMMARY is set (GitHub Actions), summary.md
is appended to the job summary.

Exit status: 0 when no task reaches --fail-on (default fail), 1 when one does, 2 when an audit could not run.
Standard library only; seqleak must be installed (or give --seqleak "python -m seqleak").

usage: python seqleak_ci.py --root data --out leakage-audit [--fail-on fail|warn|never] [--audit-args "--homology off"]
       python seqleak_ci.py --tasks tasks.tsv --out leakage-audit
"""
import argparse
import os
import re
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seqleak_card as card  # noqa: E402

SPLITS = {'train': 'train', 'dev': 'dev', 'valid': 'dev', 'val': 'dev', 'validation': 'dev', 'test': 'test'}
EXT = re.compile(r'\.(csv|tsv|txt|tab|parquet|pq|fa|fasta|fna|fas)(\.gz)?$', re.I)
STEM = re.compile(r'^(train|dev|valid|validation|val|test)([-_.].*)?$', re.I)


def discover(root, out):
    tasks = []
    out_abs = os.path.abspath(out)
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if not x.startswith('.') and os.path.abspath(os.path.join(d, x)) != out_abs)
        found = {'train': [], 'dev': [], 'test': []}
        for f in sorted(files):
            m = EXT.search(f)
            if not m:
                continue
            s = STEM.match(f[:m.start()])
            if s:
                found[SPLITS[s.group(1).lower()]].append(os.path.join(d, f))
        if found['train'] and found['test']:
            name = os.path.relpath(d, root).replace(os.sep, '/')
            args = ['--train', *found['train']] + (['--dev', *found['dev']] if found['dev'] else []) + ['--test', *found['test']]
            tasks.append((os.path.basename(os.path.abspath(root)) if name == '.' else name, args))
    return tasks


def read_tasks(path):
    tasks = []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, _, rest = line.partition('\t')
        tasks.append((name.strip(), shlex.split(rest)))
    return tasks


def main(argv=None):
    ap = argparse.ArgumentParser(prog='seqleak_ci', description=__doc__.split('\n\n')[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--root', help='folder whose task folders hold train/dev/test files')
    g.add_argument('--tasks', help='file with one task per line: NAME<TAB>seqleak audit input arguments')
    ap.add_argument('--out', required=True, help='output folder (reports, cards, summary)')
    ap.add_argument('--fail-on', default='fail', choices=['fail', 'warn', 'never'])
    ap.add_argument('--audit-args', default='', help='extra seqleak audit arguments, e.g. "--homology off"')
    ap.add_argument('--seqleak', default='seqleak', help='command that runs seqleak (default: seqleak)')
    ap.add_argument('--title', help='title of the summary card')
    a = ap.parse_args(argv)
    tasks = discover(a.root, a.out) if a.root else read_tasks(a.tasks)
    if not tasks:
        print(f'seqleak_ci: no task found under {a.root or a.tasks}', file=sys.stderr)
        return 2
    reps, errors = [], []
    for name, inputs in tasks:
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        cmd = shlex.split(a.seqleak) + ['audit', *inputs, '--task-name', name, '--out', os.path.join(d, 'report.json'),
                                         '--quiet', *shlex.split(a.audit_args)]
        print('seqleak_ci:', ' '.join(shlex.quote(x) for x in cmd), file=sys.stderr, flush=True)
        rc = subprocess.run(cmd).returncode
        path = os.path.join(d, 'report.json')
        if rc not in (0, 1) or not os.path.exists(path):
            errors.append((name, rc))
            continue
        rep = card.load(path)
        reps.append(rep)
        with open(os.path.join(d, 'card.md'), 'w', encoding='utf-8') as f:
            f.write(card.md_single(rep))
        with open(os.path.join(d, 'card.html'), 'w', encoding='utf-8') as f:
            f.write(card.html_single(rep))
        print(f"seqleak_ci: {name}: {rep['verdict']['status']} ({card.cap(rep['verdict']['summary'])})", file=sys.stderr)
    summary = ''
    if reps:
        title = a.title or f'{len(reps)} task(s) under {a.root or a.tasks}'
        summary = card.md_multi(reps, title)
        with open(os.path.join(a.out, 'summary.html'), 'w', encoding='utf-8') as f:
            f.write(card.html_multi(reps, title))
        with open(os.path.join(a.out, 'leakage_audit_section.md'), 'w', encoding='utf-8') as f:
            f.write(card.section(reps))
    if errors:
        summary += '\nAudits that could not run: ' + ', '.join(f'{n} (exit {rc})' for n, rc in errors) + '\n'
    with open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(summary)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write(summary + '\n')
    sys.stdout.write(summary)
    if errors:
        return 2
    lvl = dict(never=99, warn=1, fail=2)[a.fail_on]
    worst = max(dict(PASS=0, WARN=1, FAIL=2).get(r['verdict']['status'], 0) for r in reps)
    return 1 if worst >= lvl else 0


if __name__ == '__main__':
    sys.exit(main())
