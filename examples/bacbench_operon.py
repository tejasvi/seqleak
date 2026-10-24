"""BacBench long-read operon task: row zero, within-position AUROC, gain over position and position share.

Inputs: the task's gene table (one row per strain; per-contig lists in the columns ``start``, ``end``, ``strand`` and
``operon_prot_indices``, as in BacBench's long-read operon dataset) and one or more model score files. Each score
file is a .npy array with one score per pair of neighbouring genes, in the order BacBench scores them (strains, then
contigs, then genes), for example the concatenated ``operon_pairwise_scores`` of
``bacbench/tasks/operon/run_evaluation_long_read_rna_seq.py``.

    python bacbench_operon.py genes.parquet bacformer.npy esm2_35m.npy
"""
import os
import sys

import numpy as np
import pandas as pd

import seqleak

genes = pd.read_parquet(sys.argv[1])
coords = pd.concat([pd.DataFrame(dict(genome_id=r.strain_name, contig=c, start=r.start[c], end=r.end[c], strand=r.strand[c]))
                    for r in genes.itertuples() for c in range(len(r.start))], ignore_index=True)
pairs = seqleak.features.adjacent_pairs(coords)          # genome_id, contig, i, j and the position features

# a pair (i, i + 1) is labelled 1 when both genes lie inside one operon (BacBench's pairwise labels)
labels = []
for r in genes.itertuples():
    for c in range(len(r.start)):
        y = np.zeros(max(len(r.start[c]) - 1, 0), int)
        for operon in r.operon_prot_indices[c]:
            if len(operon):
                y[min(operon):max(operon)] = 1
        labels.append(y)
y = np.concatenate(labels)
assert len(y) == len(pairs)

scores = {os.path.basename(f)[:-4]: np.load(f) for f in sys.argv[2:]}
df = seqleak.evaluate(y, pairs.genome_id.to_numpy(), pairs, scores, order=np.arange(len(y)))
print(seqleak.format_table(df))
