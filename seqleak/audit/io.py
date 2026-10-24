"""Reading a benchmark task's splits: FASTA, CSV/TSV or parquet, one table with a split column or one file per split."""
import gzip
import hashlib
import io as _io
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEQ_COLS = ['sequence', 'seq', 'Sequence', 'SEQ', 'sequences', 'dna', 'DNA']
LABEL_COLS = ['label', 'labels', 'Label', 'y', 'target', 'class', 'Class']
SPLIT_COLS = ['split', 'set', 'Split', 'fold']
ID_COLS = ['id', 'ID', 'name', 'Name', 'seq_id', 'item_id']
CHROM_COLS = ['chrom', 'chr', 'chromosome', 'Chromosome', 'seqname', 'contig', 'seqnames']
START_COLS = ['start', 'chromStart', 'Start', 'begin']
END_COLS = ['end', 'chromEnd', 'End', 'stop']
STRAND_COLS = ['strand', 'Strand']
SPLIT_MAP = {'train': 'train', 'training': 'train', 'tr': 'train', 'trn': 'train',
             'dev': 'dev', 'valid': 'dev', 'validation': 'dev', 'val': 'dev', 'development': 'dev',
             'test': 'test', 'testing': 'test', 'te': 'test', 'tst': 'test'}
COORD_RE = re.compile(r'^(?P<chrom>[^:\s|]+):(?P<start>\d+)-(?P<end>\d+)(?:\((?P<strand>[+-])\))?')


class InputError(ValueError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def detect_format(path, fmt='auto'):
    if fmt != 'auto':
        return fmt
    p = path.lower()
    if p.endswith('.gz'):
        p = p[:-3]
    if p.endswith(('.fa', '.fasta', '.fna', '.fas')):
        return 'fasta'
    if p.endswith('.csv'):
        return 'csv'
    if p.endswith(('.tsv', '.txt', '.tab')):
        return 'tsv'
    if p.endswith(('.parquet', '.pq')):
        return 'parquet'
    raise InputError(f'cannot tell the format of {path}; use --format')


def read_fasta(path):
    op = gzip.open if path.endswith('.gz') else open
    recs = []
    with op(path, 'rt') as f:
        head, chunks = None, []
        for line in f:
            line = line.rstrip('\n\r')
            if line.startswith('>'):
                if head is not None:
                    recs.append((head, ''.join(chunks)))
                head, chunks = line[1:], []
            elif head is not None:
                chunks.append(line.strip())
        if head is not None:
            recs.append((head, ''.join(chunks)))
    rows = []
    for head, s in recs:
        toks = head.split()
        r = dict(id=toks[0] if toks else '', sequence=s)
        for t in toks[1:]:
            if '=' in t:
                k, v = t.split('=', 1)
                r[k] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ('id', 'sequence'):
            df[c] = _maybe_numeric(df[c])
    return df


def _maybe_numeric(s):
    try:
        return pd.to_numeric(s)
    except (ValueError, TypeError):
        return s


def read_table(path, fmt='auto'):
    fmt = detect_format(path, fmt)
    if fmt == 'fasta':
        return read_fasta(path)
    if fmt == 'csv':
        return pd.read_csv(path)
    if fmt == 'tsv':
        return pd.read_csv(path, sep='\t')
    if fmt == 'parquet':
        return pd.read_parquet(path)
    raise InputError(f'unknown format {fmt}')


def parse_spec(spec):
    """'PATH[@LABEL]' -> (path, label or None)."""
    if os.path.exists(spec) or '@' not in spec:
        return spec, None
    p, lab = spec.rsplit('@', 1)
    return p, lab


def _pick(cols, cands, what, required=True):
    for c in cands:
        if c in cols:
            return c
    if required:
        raise InputError(f'no {what} column among {list(cols)[:20]}; name it with the matching option')
    return None


@dataclass
class Task:
    name: str
    items: pd.DataFrame
    elements: list
    inputs: list = field(default_factory=list)
    has_coords: bool = False
    distance_cols: list = field(default_factory=list)
    group_col: str = None
    task_type: str = 'classification'
    label_values: list = field(default_factory=list)
    raw: pd.DataFrame = None
    columns: dict = field(default_factory=dict)
    formats: list = field(default_factory=list)
    mode: str = 'standard'


def _norm_label(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            f = float(v)
            return int(f) if f.is_integer() else v
        except ValueError:
            return v
    return v.item() if hasattr(v, 'item') else v


def load_files(specs, fmt, split=None):
    frames, inputs = [], []
    for spec in specs:
        path, lab = parse_spec(spec)
        if not os.path.exists(path):
            raise InputError(f'file not found: {path}')
        df = read_table(path, fmt)
        if lab is not None:
            df['label'] = lab
        if split is not None:
            df['__split'] = split
        df['__file'] = os.path.basename(path)
        frames.append(df)
        inputs.append(dict(path=os.path.abspath(path), sha256=sha256(path), split=split, label=lab))
    return frames, inputs


def load_task(data=None, train=None, dev=None, test=None, fmt='auto', seq_col=None, label_col=None, split_col=None,
              id_col=None, coords='auto', distance_cols=None, group_col=None, task_name=None, task_type='auto',
              anchor_col=None):
    frames, inputs = [], []
    if data:
        f, i = load_files(data, fmt)
        frames += f
        inputs += i
    else:
        if not test:
            raise InputError('give --data FILE (with a split column) or --test FILE (with --train FILE unless the '
                             'task is test-only)')
        for sp, specs in (('train', train), ('dev', dev or []), ('test', test)):
            if specs:
                f, i = load_files(specs, fmt, sp)
                frames += f
                inputs += i
    df = pd.concat(frames, ignore_index=True, sort=False)
    cols = list(df.columns)
    seqs = [c.strip() for c in seq_col.split(',')] if seq_col else [_pick(cols, SEQ_COLS, 'sequence')]
    for c in seqs:
        if c not in cols:
            raise InputError(f'sequence column {c} not found')
    lab = label_col or _pick(cols, LABEL_COLS, 'label')
    if lab not in cols:
        raise InputError(f'label column {lab} not found')
    if '__split' in cols and df['__split'].notna().all():
        sp = df['__split'].astype(str)
    else:
        sc = split_col or _pick(cols, SPLIT_COLS, 'split')
        if sc not in cols:
            raise InputError(f'split column {sc} not found')
        raw = df[sc].astype(str).str.strip().str.lower()
        sp = raw.map(SPLIT_MAP)
        bad = sorted(set(raw[sp.isna()]))
        if bad:
            raise InputError(f'unknown split values {bad[:10]} (expected train/dev/test and synonyms)')
    idc = id_col or _pick(cols, ID_COLS, 'id', required=False)
    items = pd.DataFrame(dict(split=sp.to_numpy()))
    items['item_id'] = df[idc].astype(str).to_numpy() if idc else np.arange(len(df)).astype(str)
    if items['item_id'].duplicated().any():
        items['item_id'] = items['item_id'] + '#' + np.arange(len(df)).astype(str)
    for i, c in enumerate(seqs):
        items[f'seq{i}'] = df[c].fillna('').astype(str).str.replace(r'\s+', '', regex=True).to_numpy()
    yl = df[lab].map(_norm_label)
    if yl.isna().any():
        raise InputError(f'{int(yl.isna().sum())} items have no label')
    items['label'] = yl.to_numpy()
    nun = items['label'].nunique()
    if task_type == 'auto':
        numeric = pd.api.types.is_numeric_dtype(pd.Series(items['label'].tolist()))
        task_type = 'regression' if numeric and nun > 20 and not all(float(v).is_integer() for v in items['label']) \
            else 'classification'
    if task_type == 'classification' and nun < 2:
        raise InputError('fewer than two label values')
    items['__file'] = df['__file'].to_numpy()
    has_coords = False
    if coords != 'none':
        cdf = None
        if coords not in (None, 'auto'):
            cd = read_table(coords)
            cid = _pick(cd.columns, ID_COLS, 'id')
            cd = cd.set_index(cd[cid].astype(str))
            cdf = cd.reindex(items['item_id'])
            cdf.index = items.index
        else:
            cdf = df
        cc = _pick(cdf.columns, CHROM_COLS, 'chrom', required=False)
        s0 = _pick(cdf.columns, START_COLS, 'start', required=False)
        e0 = _pick(cdf.columns, END_COLS, 'end', required=False)
        if cc and s0 and e0:
            items['chrom'] = cdf[cc].astype(str).to_numpy()
            items['start'] = pd.to_numeric(cdf[s0], errors='coerce').to_numpy()
            items['end'] = pd.to_numeric(cdf[e0], errors='coerce').to_numpy()
            st = _pick(cdf.columns, STRAND_COLS, 'strand', required=False)
            items['strand'] = cdf[st].astype(str).to_numpy() if st else '.'
            has_coords = bool(np.isfinite(items['start']).mean() > 0.95)
        elif coords in (None, 'auto'):
            names = df[idc].astype(str) if idc else None
            if names is not None:
                m = names.str.extract(COORD_RE)
                if m['chrom'].notna().mean() >= 0.95:
                    items['chrom'] = m['chrom'].to_numpy()
                    items['start'] = pd.to_numeric(m['start']).to_numpy()
                    items['end'] = pd.to_numeric(m['end']).to_numpy()
                    items['strand'] = m['strand'].fillna('.').to_numpy()
                    has_coords = True
    dcols = []
    for c in (distance_cols or []):
        if c not in cols:
            raise InputError(f'distance column {c} not found')
        items['dist__' + c] = pd.to_numeric(df[c], errors='coerce').to_numpy()
        dcols.append(c)
    if group_col:
        if group_col not in cols:
            raise InputError(f'group column {group_col} not found')
        items['group'] = df[group_col].astype(str).to_numpy()
    if anchor_col:
        if anchor_col not in cols:
            raise InputError(f'anchor column {anchor_col} not found')
        a = pd.to_numeric(df[anchor_col], errors='coerce')
        if a.isna().any():
            raise InputError(f'anchor column {anchor_col} has missing or non-numeric values')
        items['anchor'] = a.astype(np.int64).to_numpy()
    name = task_name or (os.path.splitext(os.path.basename(inputs[0]['path']))[0] if inputs else 'task')
    if not (items['split'] == 'test').any():
        raise InputError('no test items')
    mode = 'standard' if (items['split'] == 'train').any() else 'test_only'   # test-only (zero-shot) tasks: rules v1.1
    raw = df.drop(columns=[c for c in ('__split', '__file') if c in df.columns])
    split_name = None if ('__split' in cols and df['__split'].notna().all()) else (split_col or _pick(cols, SPLIT_COLS, 'split'))
    fmts = sorted({detect_format(parse_spec(x)[0], fmt) for x in (data or (list(train or []) + list(dev or []) + list(test or [])))})
    return Task(name=name, items=items, elements=seqs, inputs=inputs, has_coords=has_coords, distance_cols=dcols,
                group_col=group_col, task_type=task_type, label_values=sorted(items['label'].unique(), key=str),
                raw=raw, columns=dict(seq=seqs, label=lab, split=split_name, id=idc), formats=fmts, mode=mode)


def load_reference(name, specs, fmt='auto', seq_col=None, split_col=None):
    frames, inputs = load_files(specs, fmt)
    df = pd.concat(frames, ignore_index=True, sort=False)
    cols = list(df.columns)
    seqs = [c.strip() for c in seq_col.split(',')] if seq_col and all(
        c.strip() in cols for c in seq_col.split(',')) else [_pick(cols, SEQ_COLS, 'sequence')]
    sc = split_col if split_col in cols else _pick(cols, SPLIT_COLS, 'split', required=False)
    sp = df[sc].astype(str).str.strip().str.lower().map(SPLIT_MAP).fillna('train') if sc else pd.Series(['all'] * len(df))
    lab = _pick(cols, LABEL_COLS, 'label', required=False)
    out = pd.DataFrame(dict(split=sp.to_numpy()))
    for i, c in enumerate(seqs):
        out[f'seq{i}'] = df[c].fillna('').astype(str).to_numpy()
    out['label'] = df[lab].map(_norm_label).to_numpy() if lab else None
    return dict(name=name, items=out, elements=seqs, inputs=inputs)
