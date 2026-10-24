"""Synthetic genomes and labels with a known answer, for tests and examples.

``simulate_pairs``: position is a CAUSE of the label. Labels of neighbouring gene pairs depend on position (strand and
gap) and on a latent variable ``z`` that is independent of position. Scores:

* ``position_only``: a function of strand and gap only (no ``z``);
* ``independent``: ``z`` plus noise (no position);
* ``mixed``: both.

``simulate_pairs_consequence``: position is a CONSEQUENCE of the label (labels first, then strand and gap).
"""
import numpy as np
import pandas as pd

from .features import adjacent_pairs


def simulate_genomes(n_genomes=20, n_genes=2000, seed=0):
    """Gene coordinates: directons with geometric lengths, gene lengths ~ 300-3000 bp, gaps that are short inside
    directons and longer between them."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_genomes):
        strand = []
        while len(strand) < n_genes:
            strand += [rng.choice([1, -1])] * int(rng.geometric(0.3))
        strand = np.array(strand[:n_genes])
        same_next = np.r_[strand[1:] == strand[:-1], True]
        gap = np.where(same_next, rng.normal(40, 60, n_genes), rng.lognormal(5.3, 0.7, n_genes)).round().astype(int)
        length = rng.integers(300, 3000, n_genes)
        start = np.zeros(n_genes, np.int64)
        end = np.zeros(n_genes, np.int64)
        pos = 1000
        for t in range(n_genes):
            start[t] = pos
            end[t] = pos + length[t] - 1
            pos = end[t] + gap[t]
        rows.append(pd.DataFrame(dict(genome_id=f'g{g:03d}', contig=0, start=start, end=end, strand=strand)))
    return pd.concat(rows, ignore_index=True)


def simulate_pairs_consequence(n_genomes=20, n_pairs=1500, seed=0):
    """Neighbouring pairs where position is a CONSEQUENCE of the label (as for operons): the label comes first, strand
    and gap depend on it, and a biology variable ``z = label + noise`` is independent of position given the label.
    Columns: genome_id, label, the pair position features, ``z``, ``oracle`` (label + noise) and ``position_only``
    (the distance rule of the pair's strand and gap)."""
    from .features import rule_score
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_genomes):
        y = (rng.random(n_pairs) < 0.4).astype(np.int8)
        same = np.where(y == 1, True, rng.random(n_pairs) < 0.5)
        gap = np.where(y == 1, rng.normal(25, 35, n_pairs),
                       np.where(same, rng.lognormal(5.0, 0.8, n_pairs) - 40, rng.lognormal(5.5, 0.7, n_pairs))).round()
        rows.append(pd.DataFrame(dict(genome_id=f'a{g:03d}', label=y, genes_apart=1, same_strand=same, gap=gap,
                                      max_intervening_gap=np.maximum(gap, 0), strand_switches=(~same).astype(int),
                                      z=y + rng.normal(size=n_pairs), oracle=y + 1.5 * rng.normal(size=n_pairs))))
    P = pd.concat(rows, ignore_index=True)
    P['position_only'] = rule_score(P.same_strand.to_numpy(), P.gap.to_numpy())
    return P


def simulate_pairs(n_genomes=20, n_genes=2000, beta_position=1.0, beta_latent=1.0, seed=0):
    """Neighbouring pairs with labels and three scores (see module docstring). Returns (coords, pairs) where pairs
    has the position features, ``label``, ``z`` and the score columns."""
    rng = np.random.default_rng(seed + 1)
    coords = simulate_genomes(n_genomes, n_genes, seed)
    P = adjacent_pairs(coords)
    same = P['same_strand'].to_numpy()
    gap = P['gap'].to_numpy().astype(float)
    f_pos = np.where(same, 2.0 - gap / 60.0, -3.0)
    z = rng.normal(size=len(P))
    logit = beta_position * f_pos + beta_latent * z - 0.5
    P['label'] = (rng.random(len(P)) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    P['z'] = z
    P['position_only'] = f_pos + 0.05 * rng.normal(size=len(P))
    P['independent'] = z + 0.5 * rng.normal(size=len(P))
    P['mixed'] = f_pos + z + 0.5 * rng.normal(size=len(P))
    return coords, P
