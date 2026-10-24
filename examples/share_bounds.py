"""The two position-share bounds on simulated genomes with a known answer (the table under "Why a bound" in the README).

Two cases, 30 genomes each, neighbouring gene pairs, strata strand x 17 gap bins:

  position is a consequence of the label (seqleak.synthetic.simulate_pairs_consequence): the label comes first and
      strand and gap depend on it; the biology variable Z = label + noise is independent of position given the label.
  position is a cause of the label (seqleak.synthetic.simulate_pairs): logit P(label) = f(strand, gap) + z, with z
      independent of position.

Scores: nulls with no position information, a position-only score, and mixtures lam x position part + (1 - lam) x
position-free part (both standardised). True share of a mixture: 1 - (AUROC of the position-free part - 1/2) / (AUROC
of the mixture - 1/2); 0 for the nulls, 1 for the position-only score. Columns: AUROC, within-strata AUROC, equalized
AUROC, the true share, share_within, share_equalized, the explained share, and the two bounds (BOUND = share).

Run: python examples/share_bounds.py
"""
import numpy as np

from seqleak import metrics as M
from seqleak.strata import gap_strata
from seqleak.synthetic import simulate_pairs, simulate_pairs_consequence


def std(x):
    return (x - x.mean()) / x.std()


def table(title, P, pos_part, free_part, nulls, header=False):
    y = P.label.to_numpy()
    g = P.genome_id.to_numpy()
    k = gap_strata(P.same_strand.to_numpy(), P.gap.to_numpy())
    a_free = M.auroc(y, free_part, g).estimate
    scores = dict(nulls)
    for lam in (0.25, 0.5, 0.75):
        scores[f'mixture lam={lam}'] = lam * std(pos_part) + (1 - lam) * std(free_part)
    scores['position only'] = pos_part + 0.01 * np.random.default_rng(9).normal(size=len(y))
    print(title)
    if header:
        print(f'{"score":<25s}{"AUROC":>6s} {"within":>6s} {"eq":>6s} | {"true":>6s} {"s_within":>8s} '
              f'{"s_equal":>8s} {"explained":>9s} | {"BOUND":>6s} {"upper":>6s}')
    for name, s in scores.items():
        raw = M.auroc(y, s, g).estimate
        true = 1 - (a_free - 0.5) / (raw - 0.5) if name.startswith('mixture') else float(name == 'position only')
        b = M.position_share_bound(y, s, k, g)
        i = b.info
        ex = M.explained_share(y, s, k, g).estimate
        print(f'{name:<25s}{raw:6.3f} {i["within"]:6.3f} {i["auroc_eq"]:6.3f} | {true:6.3f} {i["share_within"]:8.3f} '
              f'{i["share_equalized"]:8.3f} {ex:9.3f} | {b.estimate:6.3f} {i["share_upper"]:6.3f}')


PA = simulate_pairs_consequence(n_genomes=30, n_pairs=1500, seed=0)
table('position is a CONSEQUENCE of the label (true = within)', PA, PA.position_only.to_numpy(), PA.z.to_numpy(),
      {'null: oracle (Y + noise)': PA.oracle.to_numpy(), 'null: biology Z = Y+noise': PA.z.to_numpy()}, header=True)
_, PB = simulate_pairs(n_genomes=30, n_genes=1500, beta_position=1.0, beta_latent=1.0, seed=0)
table('position is a CAUSE of the label (true = equalized)', PB,
      np.where(PB.same_strand, 2.0 - PB.gap / 60.0, -3.0), PB.independent.to_numpy(),
      {'null: z independent': PB.independent.to_numpy()})
