"""DGEB operonic pair tasks (E. coli K-12, S. elongatus PCC 7942, V. cholerae N16961): position baseline and
position-held-fixed metrics for a model's score of consecutive entries.

DGEB's datasets list the proteins of one genome in genome order (columns ``Entry``, ``Sequence``, ``Label``) and
score consecutive entries; ``Label`` of entry k is the label of the pair (k, k + 1). The ``Entry`` ids are the FASTA
ids of NCBI's ``fasta_cds_aa`` file of the genome record (U00096.3, CP000100.1, NZ_LT906614.1), whose headers give
each protein's location. The score file is a .npy array with one score per consecutive pair (n_entries - 1), for
example the cosine of the two entries' embeddings at the layer the leaderboard number is taken from.

    python dgeb_operonic_pair.py train.parquet U00096.3.cds_aa.fasta esm2_35m_cos.npy
"""
import collections
import os
import re
import sys

import numpy as np
import pandas as pd

import seqleak


def protein_id(fasta_id):
    m = re.search(r'_prot_(.+)_\d+$', fasta_id)
    return m.group(1) if m else None


def cds_locations(fasta):
    """(start, end, strand) by FASTA id and by protein id (unique ids only) from NCBI fasta_cds_aa headers. NCBI
    numbers the CDS in each id, so ids shift when genes are added to the record after a dataset was built; the
    protein id still matches then."""
    by_id, by_protein, count = {}, {}, collections.Counter()
    for line in open(fasta):
        if line.startswith('>'):
            fid = line[1:].split()[0].replace('lcl|', '')
            where = re.search(r'\[location=([^\]]+)\]', line).group(1)
            bp = [int(x) for x in re.findall(r'\d+', where)]
            by_id[fid] = (min(bp), max(bp), -1 if 'complement' in where else 1)
            count[protein_id(fid)] += 1
            by_protein[protein_id(fid)] = by_id[fid]
    return by_id, {p: v for p, v in by_protein.items() if count[p] == 1}


ds = pd.read_parquet(sys.argv[1])
by_id, by_protein = cds_locations(sys.argv[2])
nan3 = (np.nan, np.nan, np.nan)
xyz = np.array([by_id.get(e.replace('lcl|', ''), by_protein.get(protein_id(e.replace('lcl|', '')), nan3)) for e in ds.Entry], float)
start, end, strand = xyz.T
ok = np.isfinite(start[:-1]) & np.isfinite(start[1:])      # pairs whose two entries were found in the record
print(f'{len(ds)} entries, {int(np.isfinite(start).sum())} with coordinates; {int(ok.sum())} of {len(ds) - 1} pairs usable')

gap = (start[1:] - end[:-1])[ok].astype(int)
same = (strand[1:] == strand[:-1])[ok]
features = pd.DataFrame(dict(genes_apart=1, same_strand=same, gap=gap, max_intervening_gap=np.maximum(gap, 0),
                             strand_switches=(~same).astype(int)))
y = ds.Label.to_numpy()[:-1][ok]
scores = {os.path.basename(f)[:-4]: np.load(f)[ok] for f in sys.argv[3:]}
genome = np.repeat(os.path.basename(sys.argv[2]).split('.cds')[0], len(y))
df = seqleak.evaluate(y, genome, features, scores, order=np.arange(len(y)))     # one genome: 5 contiguous folds
print(seqleak.format_table(df))
