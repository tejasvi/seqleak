"""Task tables in the shared format (one directory per task).

``<task_dir>/items.parquet``: one row per evaluated item with ``genome_id``, ``contig``, ``i``, ``j`` (gene indices;
``j = -1`` for per-gene tasks), ``label``, optional ``split``, and the position features
(pairs: ``genes_apart``, ``same_strand``, ``gap``, ``max_intervening_gap``, ``strand_switches``; genes: the columns
of :func:`seqleak.features.gene_features`).
``<task_dir>/scores/<model>.npy``: one score per item (zero-shot score or out-of-fold probe prediction).
``<task_dir>/scores/README``: how each score was made. ``<task_dir>/TASK.md``: source, labels, protocol, owner.
"""
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED = ('genome_id', 'label')


@dataclass
class Task:
    path: str
    items: pd.DataFrame

    @property
    def task_id(self):
        return os.path.basename(os.path.normpath(self.path))

    @property
    def kind(self):
        if 'j' in self.items and (self.items['j'].to_numpy() >= 0).any():
            return 'pair'
        return 'gene'

    @property
    def models(self):
        return sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(self.path, 'scores', '*.npy')))

    def scores(self, model, mmap=False):
        s = np.load(os.path.join(self.path, 'scores', f'{model}.npy'), mmap_mode='r' if mmap else None)
        if s.ndim != 1 and not (s.ndim == 2 and s.shape[1] > 1):
            s = s.reshape(-1)
        if len(s) != len(self.items):
            raise ValueError(f'{model}: {len(s)} scores for {len(self.items)} items')
        return s

    def order(self):
        """Genome order of the items (for block bootstrap): genome, contig, min(i, j), max(i, j)."""
        it = self.items
        cols = [c for c in ('genome_id', 'contig') if c in it]
        keys = [it[c].to_numpy() for c in cols]
        if 'i' in it:
            i = it['i'].to_numpy()
            j = it['j'].to_numpy() if 'j' in it else i
            j = np.where(j < 0, i, j)
            keys += [np.minimum(i, j), np.maximum(i, j)]
        if not keys:
            return np.arange(len(it))
        return np.lexsort(tuple(reversed([pd.factorize(k, sort=True)[0] if k.dtype.kind in 'OUS' else k for k in keys])))


def load_task(task_dir, columns=None):
    items = pd.read_parquet(os.path.join(task_dir, 'items.parquet'), columns=columns)
    missing = [c for c in REQUIRED if c not in items]
    if missing:
        raise ValueError(f'items.parquet lacks columns {missing}')
    return Task(task_dir, items)


def write_task(task_dir, items, scores=None, task_md=None, scores_readme=None):
    """Write a task directory (items.parquet, scores/<model>.npy, TASK.md, scores/README)."""
    os.makedirs(os.path.join(task_dir, 'scores'), exist_ok=True)
    missing = [c for c in REQUIRED if c not in items]
    if missing:
        raise ValueError(f'items lack columns {missing}')
    items.to_parquet(os.path.join(task_dir, 'items.parquet'), index=False)
    for m, s in (scores or {}).items():
        s = np.asarray(s)
        if len(s) != len(items):
            raise ValueError(f'{m}: {len(s)} scores for {len(items)} items')
        np.save(os.path.join(task_dir, 'scores', f'{m}.npy'), s)
    if task_md is not None:
        with open(os.path.join(task_dir, 'TASK.md'), 'w') as f:
            f.write(task_md)
    if scores_readme is not None:
        with open(os.path.join(task_dir, 'scores', 'README'), 'w') as f:
            f.write(scores_readme)
