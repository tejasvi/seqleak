"""Three alternative definitions of a "position share" on synthetic genomes with a known answer.

Labels of neighbouring gene pairs depend on position (strand, gap) with strength ``beta_position`` and on a latent
variable independent of position. Scores: ``position_only`` (knows position only, answer 1), ``independent`` (knows
the latent only, answer 0), ``mixed`` (both). Candidates:

  within share     1 - (within - 1/2) / (AUROC - 1/2)
  gain share       1 - gain over position / (AUROC - 1/2)
  explained share  seqleak.metrics.explained_share

seqleak reports instead the lower and upper bounds of seqleak.metrics.position_share_bound (share, share_upper).

Run: python examples/share_definitions.py
"""
import numpy as np

from seqleak import metrics as M
from seqleak.baseline import position_model
from seqleak.strata import gap_strata
from seqleak.synthetic import simulate_pairs

print(f'{"beta_pos":>8s} {"score":14s} {"AUROC":>6s} {"within":>6s} {"gain":>7s} | {"within share":>12s} {"gain share":>10s} {"explained":>9s}')
for bp in (0.5, 1.0, 2.0):
    coords, P = simulate_pairs(n_genomes=30, n_genes=1500, beta_position=bp, beta_latent=1.0, seed=0)
    y = P.label.to_numpy()
    g = P.genome_id.to_numpy()
    k = gap_strata(P.same_strand.to_numpy(), P.gap.to_numpy())
    pos = position_model(P, y, g, folds=5)
    for name in ('position_only', 'independent', 'mixed'):
        s = P[name].to_numpy()
        raw = M.auroc(y, s, g).estimate
        wit = M.within_auc(y, s, k, g).estimate
        gain = M.gain_over_position(y, pos, position_model(P, y, g, folds=5, extra=s), g).estimate
        ex = M.explained_share(y, s, k, g).estimate
        print(f'{bp:8.1f} {name:14s} {raw:6.3f} {wit:6.3f} {gain:+7.4f} | {1 - (wit - .5) / (raw - .5):12.3f} '
              f'{1 - gain / (raw - .5):10.3f} {ex:9.3f}', flush=True)
