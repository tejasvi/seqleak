"""Command line: ``seqleak audit`` (leak linter) and ``seqleak clean`` (reduced-leakage split)."""
import argparse
import json
import os
import sys
import traceback


def _common(p):
    g = p.add_argument_group('input (one table with a split column, or one file per split; FILE = PATH[@LABEL])')
    g.add_argument('--data', nargs='+', action='extend', metavar='FILE', help='table(s) with a split column')
    g.add_argument('--train', nargs='+', action='extend', metavar='FILE')
    g.add_argument('--dev', nargs='+', action='extend', metavar='FILE')
    g.add_argument('--test', nargs='+', action='extend', metavar='FILE')
    g.add_argument('--format', default='auto', choices=['auto', 'csv', 'tsv', 'parquet', 'fasta'])
    g.add_argument('--seq-col', help='sequence column(s), comma-separated for multi-element items')
    g.add_argument('--label-col')
    g.add_argument('--split-col')
    g.add_argument('--id-col')
    g.add_argument('--coords', default='auto', help='auto | none | FILE (id + chrom/start/end)')
    g.add_argument('--distance-cols', help='comma-separated position fields (locus or pair tasks)')
    g.add_argument('--group-col', help='genome or chromosome column (grouped CV / cluster bootstrap)')
    g.add_argument('--task-dir', help='seqleak gene-pair task directory (position check = row zero)')
    g.add_argument('--anchor-col', help='per-item site position (0-based) anchoring the motif table of '
                                        'variable-length items (default: the item ends)')
    g.add_argument('--reference', action='append', default=[], metavar='NAME=FILE[,FILE]',
                   help="another benchmark's table (repeatable)")
    g.add_argument('--task-name')
    g.add_argument('--task-type', default='auto', choices=['auto', 'classification', 'regression'])
    g.add_argument('--metric', default='auto', choices=['auto', 'mcc', 'acc', 'f1', 'auroc'])
    g.add_argument('--published-best', type=float)
    c = p.add_argument_group('checks')
    c.add_argument('--checks', help='comma-separated check ids or families (default all applicable)')
    c.add_argument('--skip', help='comma-separated check ids or families to skip')
    c.add_argument('--homology', default='auto', choices=['auto', 'off', 'blast', 'mmseqs'])
    c.add_argument('--homology-max-test', type=int, default=2000)
    c.add_argument('--blast-bin', help='directory with blastn and makeblastdb')
    c.add_argument('--mmseqs-bin', help='path of the mmseqs binary')
    c.add_argument('--genome', help='reference genome(s) for the membership check: FASTA path(s) or registry names, '
                                    'comma-separated (membership in any)')
    c.add_argument('--genome-registry', help='TSV name<TAB>FASTA path (or set SEQLEAK_GENOME_REGISTRY)')
    c.add_argument('--genome-method', default='kmer', choices=['kmer', 'minimap2'],
                   help='k-mer containment (built in) or full-length minimap2 alignment')
    c.add_argument('--minimap2-bin', help='path of the minimap2 binary (--genome-method minimap2)')
    c.add_argument('--k', type=int, default=21)
    c.add_argument('--sketch-scale', default='auto')
    c.add_argument('--max-df', type=int, default=64)
    c.add_argument('--thresholds', help='JSON file overriding thresholds')
    c.add_argument('--n-boot', type=int, default=1000)
    c.add_argument('--folds', type=int, default=5)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--threads', type=int, default=1)
    c.add_argument('--examples', type=int, default=5)
    c.add_argument('--workdir', help='directory for temporary files (default $TMPDIR, else next to the output)')
    c.add_argument('--quiet', action='store_true')


def build_parser():
    ap = argparse.ArgumentParser(prog='seqleak', description='seqleak command line')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('audit', help='leak linter for a benchmark task')
    _common(a)
    o = a.add_argument_group('output')
    o.add_argument('--out', required=True, help='report.json')
    o.add_argument('--md', help='report.md (default: next to --out)')
    o.add_argument('--items-out', help='per-item table (default: <out>.items.parquet)')
    o.add_argument('--fail-on', default='never', choices=['never', 'warn', 'fail'])
    c = sub.add_parser('clean', help='write a reduced-leakage split and a manifest of every change')
    _common(c)
    o = c.add_argument_group('clean')
    o.add_argument('--out-dir', required=True)
    o.add_argument('--mode', default='filter', choices=['filter', 'resplit'])
    o.add_argument('--group-by', default='auto', choices=['auto', 'cluster', 'locus', 'chromosome'])
    o.add_argument('--links', default='all', help='exact,near,overlap,twins,coords,homology,reference or all')
    o.add_argument('--near-threshold', type=float, default=0.5)
    o.add_argument('--overlap-threshold', type=float, default=0.2)
    o.add_argument('--locus-gap', type=int, default=10000)
    o.add_argument('--fractions', help='TRAIN,DEV,TEST (resplit; default: the input proportions)')
    o.add_argument('--drop-reference', action='store_true')
    o.add_argument('--out-format', default='auto', choices=['auto', 'csv', 'tsv', 'parquet', 'fasta'])
    o.add_argument('--no-reaudit', action='store_true')
    return ap


def _split(s):
    return [x.strip() for x in s.split(',') if x.strip()] if s else None


def load_inputs(args):
    from .audit import load_reference, load_task
    task = load_task(data=args.data, train=args.train, dev=args.dev, test=args.test, fmt=args.format,
                     seq_col=args.seq_col, label_col=args.label_col, split_col=args.split_col, id_col=args.id_col,
                     coords=args.coords, distance_cols=_split(args.distance_cols), group_col=args.group_col,
                     task_name=args.task_name, task_type=args.task_type, anchor_col=args.anchor_col)
    refs = []
    for r in args.reference:
        if '=' not in r:
            raise SystemExit(f'--reference expects NAME=FILE[,FILE], got {r}')
        name, files = r.split('=', 1)
        refs.append(load_reference(name, files.split(','), args.format, args.seq_col, args.split_col))
    return task, refs


def audit_kwargs(args):
    from .audit import thresholds as TH
    return dict(checks=_split(args.checks), skip=_split(args.skip), thresholds=TH.load(args.thresholds), k=args.k,
                sketch_scale=args.sketch_scale, max_df=args.max_df, n_boot=args.n_boot, folds=args.folds,
                seed=args.seed, threads=args.threads, examples=args.examples, homology=args.homology,
                homology_max_test=args.homology_max_test, blast_bin=args.blast_bin, mmseqs_bin=args.mmseqs_bin,
                metric=args.metric, published_best=args.published_best, task_dir=args.task_dir,
                workdir=args.workdir, quiet=args.quiet, genome=_split(args.genome), genome_registry=args.genome_registry,
                genome_method=args.genome_method, minimap2_bin=args.minimap2_bin)


def cmd_audit(args, argv):
    from . import __version__
    from .audit import run_audit, to_markdown, write_json
    task, refs = load_inputs(args)
    rep, items = run_audit(task, refs, argv=argv, version=__version__, **audit_kwargs(args))
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_json(rep, out)
    md = args.md or (os.path.splitext(out)[0] + '.md')
    with open(md, 'w') as f:
        f.write(to_markdown(rep))
    io_ = args.items_out or (os.path.splitext(out)[0] + '.items.parquet')
    items.to_parquet(io_, index=False)
    if not args.quiet:
        print(f"[seqleak audit] {task.name}: {rep['verdict']['status']} ({rep['verdict']['summary']}); "
              f"{rep['runtime']['seconds']} s; report {out}", file=sys.stderr)
    lvl = dict(never=99, warn=1, fail=2)[args.fail_on]
    rank = dict(PASS=0, WARN=1, FAIL=2)[rep['verdict']['status']]
    return 1 if rank >= lvl else 0


def cmd_clean(args, argv):
    from . import __version__
    from .clean import run_clean
    task, refs = load_inputs(args)
    run_clean(task, refs, args, argv=argv, version=__version__, audit_kw=audit_kwargs(args))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    args = ap.parse_args(argv)
    from .audit import InputError
    try:
        if args.cmd == 'audit':
            return cmd_audit(args, ['seqleak'] + argv)
        return cmd_clean(args, ['seqleak'] + argv)
    except InputError as e:
        print(f'seqleak {args.cmd}: input error: {e}', file=sys.stderr)
        return 2
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 3


def main_posbench(argv=None):
    """The former command name (the package was called posbench up to rules v1.2): the same command line, after a
    one-line notice of the rename."""
    print('posbench has been renamed seqleak; running seqleak', file=sys.stderr)
    return main(argv)


if __name__ == '__main__':
    sys.exit(main())
