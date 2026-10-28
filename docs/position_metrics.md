# Gene-position metrics

Many bacterial benchmarks ask about genes that are near each other (operons, protein interactions, functional
links). Gene position alone (strand, intergenic distance, adjacency) predicts much of what these benchmarks label.
On eukaryotic genomes, distance to the transcription start site plays the same part in enhancer-gene and
variant-effect tasks.
seqleak adds four things to a benchmark table:

1. **Row zero**: a position-only baseline (logistic regression on spline bases of the position features, fitted with
   genome-grouped cross-validation), reported above the models.
2. **Position held fixed**: each model's AUROC computed only between items that share their gene position
   (within strata).
3. **Gain over position**: what a model's score adds to the position baseline.
4. **Position share**: a lower bound, `share`, and an upper bound, `share_upper`, on the share of a model's
   above-chance AUROC that comes from gene position. The lower bound is exact when position is a consequence of the
   labels, the upper bound when position is a cause of them.

Every number comes with a 95% bootstrap interval over genomes (or over blocks of neighbouring items when a task has
few genomes). seqleak is written in pure numpy, pandas, scipy and scikit-learn and is vectorised; a 16-million-pair
task runs on one machine.

These metrics come unchanged from posbench 0.3.2, which is archived with the paper that introduced them
(doi:10.5281/zenodo.23605239).

## Quick start

```python
import seqleak
df = seqleak.row_zero('path/to/task')     # items.parquet + scores/<model>.npy, see "Task format"
print(seqleak.format_table(df))           # row zero, AUROC, within strata, gain, share [lower, upper]
df = seqleak.evaluate(y, genomes, features, {'model': scores})    # the same on arrays in memory
```

The table has one row per scorer: row zero (the position model), the distance rule for pair tasks, then each
model. Its columns:

* `auroc`, `auprc`: per genome, averaged over genomes.
* `within`: the AUROC within position strata.
* `gain`: the out-of-fold AUROC of position + model minus that of position. `gain_p` is its Wilcoxon p-value over
  genomes.
* `share`, `share_upper`: the lower and upper bounds of the position share.

Each of these except `gain_p` has 95% limits in the columns ending in `_lo` and `_hi`. The columns `n_items`, `n_pos`
and `n_genomes` hold the counts, and `df.attrs` holds the settings used.

## Integration examples

* `examples/bacbench_operon.py`: BacBench's long-read operon task, from its gene table and a model's pairwise scores.
* `examples/dgeb_operonic_pair.py`: DGEB's operonic pair tasks, with protein locations from NCBI's `fasta_cds_aa`
  file of each genome record.
* `examples/dnalongbench_distance.py`: distance-to-TSS tasks (DNALongBench enhancer-target genes and eQTL, the
  Long-Range Benchmark's causal eQTL), with distance as row zero.
* `examples/share_bounds.py`: the two share bounds on simulated genomes with a known answer (the table under "Why a
  bound").
* `examples/share_definitions.py`: three alternative share definitions on synthetic data (see "Alternative
  definitions").

## Other calls

Tables with other column names or several sub-tasks:

```python
seqleak.row_zero(task, by=['dataset'])                            # one block of rows per dataset
seqleak.row_zero(task, query="dataset == 'yeast_two_hybrid'")     # a subset
seqleak.row_zero(task, position_columns=['distance_to_tss'])      # any position features (prognostic strata)
```

**Per-gene tasks.** The `'gene'` preset (position model and strata) includes gene length, which is a property of
the gene rather than of its layout. With it, row zero, the gain over position and the position share count gene
length as position. For layout only, pass the layout columns. They must have no missing values, so the list below
leaves out `gap_up` and `gap_down`, which are missing at contig ends:

```python
seqleak.row_zero(task, position_columns=['strand', 'replication_strand', 'dist_origin', 'directon_pos',
                                          'directon_size', 'operon_rule', 'density'])   # gene layout, no length
```

The same on arrays already in memory:

```python
from seqleak import Resample, auroc, within_auc, position_share_bound, gain_over_position, position_model, make_strata
strata = make_strata(features)                          # features: DataFrame of position features
rs = Resample(genomes, method='cluster', n_boot=2000)   # one plan for every statistic -> paired intervals
raw = auroc(y, score, genomes, resample=rs)
within = within_auc(y, score, strata, genomes, resample=rs)
share = position_share_bound(y, score, strata, genomes, resample=rs)   # .estimate, .ci, .info['share_upper']
pos = position_model(features, y, genomes)              # out-of-fold, folds grouped by genome
gain = gain_over_position(y, pos, position_model(features, y, genomes, extra=score), genomes, resample=rs)
print(raw, within, share, gain, sep='\n')               # estimate [95% CI]
```

## Task format

A task directory holds

* `items.parquet`: one row per evaluated item with `genome_id`, `contig`, `i`, `j`, `label` (0/1), optional `split`,
  and the position features. `i` and `j` are 0-based gene indices within the contig, and `j = -1` for per-gene tasks.
* `scores/<model>.npy`: one score per item, aligned with `items.parquet` (zero-shot score or out-of-fold probe
  prediction), and `scores/README` saying how each was made.
* `TASK.md`: source, label construction, the benchmark's own protocol and metric, published numbers.

`seqleak.features.pair_features(coords, i, j)` computes the pair features: `genes_apart`, `same_strand`, `gap` (bp
between the two genes, negative for overlaps), `max_intervening_gap` and `strand_switches`.
`seqleak.features.gene_features(coords, origin=...)` computes the gene features: strand, length, gaps and rule scores
to the neighbours, directon size and position, local gene density, distance to the replication origin, and
leading/lagging strand. `seqleak.features.origin_from_gc_skew(sequence)` locates the origin when it is not known.

If the table has a `split` column with `test` items, those are evaluated. The position models are fitted on `train`
items when every model has scores for them, otherwise by genome-grouped cross-validation within the test genomes.
The folds are 5 genome-grouped folds for more than 10 genomes, leave-one-genome-out for 2-10 genomes, and five
contiguous chunks of the genome for a single genome. In the single-genome case the folds split a genome. Every metric
of out-of-fold predictions (row zero, the gain over position) then compares items only within their fold: AUROC
within folds is weighted by positives x negatives and AUPRC within folds by positives. `df.attrs['fold_stratified']`
is true when metrics are computed within folds. Pooling the predictions of different fold models biases the AUROC
down, because a fold rich in positives was trained on fewer (Forman and Scholz 2010; Airola et al. 2011). The bias is
largest for coarse predictors such as a position model.

## Definitions

Notation: y label (1/0), s model score, g genome, k position stratum, P_g / N_g positives / negatives of genome g.
Metrics are computed per genome and averaged over genomes (the BacBench convention) unless `aggregate='pooled'`.

* **AUROC**: the benchmark's metric on the model's score. **AUPRC**: average precision (step-wise, as in
  scikit-learn).
* **Within-position AUROC**: positives are compared only with negatives of the same genome and stratum,
  `within_g = sum_k U_gk / sum_k P_gk N_gk`, i.e. stratum AUROCs weighted by positives x negatives
  (U = Mann-Whitney count, ties 1/2). A score that is constant within each stratum scores exactly 0.5. A score that
  knows only position is constant within strata only if the strata fix position exactly. With the binned strata used
  here some position signal remains inside each stratum, so a position-only score scores near 0.5 rather than at it.
  Row zero reports, for each table, how near 0.5 the position model itself comes: it scores 0.513 within strata on
  BacBench's long-read operon task and 0.534 on its zero-shot STRING interaction task.
* **Strata** (`seqleak.strata`). Neighbouring pairs are stratified by strand state x 17 gap bins. Any pairs are
  stratified by 11 genes-apart bins x strand switches (0-3+) x 9 bins of the largest intervening gap x strand state.
  Genes (`'gene'`) are stratified by directon position x leading/lagging x distance to origin x gene-length quintile.
  Gene length is a gene-level covariate, not layout (see "Per-gene tasks" above). For any features, `'prognostic'`
  uses 20 quantile bins of the out-of-fold position-model prediction within each genome. A custom scheme is a dict
  of column -> bins.
* **Position model** (row zero): a logistic regression on cubic B-spline bases of the position features, with
  separate splines for same- and opposite-strand pairs, fitted out-of-fold over genome-grouped folds. It is fitted to
  convergence with a second-order solver (see "Validation"). `model='gbm'` uses histogram gradient boosting instead.
* **Gain over position**: the per-genome AUROC of the position + model fit minus that of the position fit, averaged
  over genomes. Both fits are out-of-fold with the same folds. A Wilcoxon signed-rank test over genomes is reported
  with 6 or more genomes.
* **Position share**: see the next section.

## Position share

`share` is a lower bound on the fraction of a model's above-chance AUROC that comes from gene position (0 means none,
1 means position alone), and `share_upper` is the matching upper bound. If position is a consequence of the labels,
as for operons, where co-transcription shapes gene spacing, the true share equals `share`. If position is a cause of
the labels, as for STRING links scored from gene neighbourhood, it equals `share_upper`. If both hold in part, it
lies between them.

Read `share` as "at least this much of the model's performance is gene position". A small `share` with a large
`share_upper` means the answer depends on how the labels were made, so report both.

### Definition

Let raw be the model's AUROC (per genome, averaged over genomes). Two AUROCs leave out the position information of
the score. Both are computed from the scores and position strata alone:

* **held fixed**: within = the AUROC within position strata (pairs compared only inside a stratum).
* **equalized**: auroc_eq = the AUROC after each score is replaced by its percentile inside its (genome, stratum)
  cell. This makes the score identical in distribution across positions and keeps its ranking inside every stratum
  (`seqleak.metrics.equalized_score`).

Then

    share_within    = 1 - (within   - 1/2) / (raw - 1/2)
    share_equalized = 1 - (auroc_eq - 1/2) / (raw - 1/2)
    position share  = clip(min(share_within, share_equalized), 0, 1)       (column `share`)
    upper bound     = clip(max(share_within, share_equalized), 0, 1)       (column `share_upper`)

`seqleak.metrics.position_share_bound` computes them, and they are the `share` and `share_upper` columns of
`row_zero`.

### Why a bound

Think of a score as a position part plus a position-free part. Its true position share is the fraction of
raw - 1/2 it would lose without the position part. No single statistic of (score, position, label) recovers this in
every case, because the answer depends on whether position is a consequence or a cause of the label:

* **Position is a consequence of the label** (the operon case above). A model's biological signal is then
  independent of position given the label. Holding position fixed is the right counterfactual, so share_within is
  exact. Equalizing removes label-driven differences between strata, so share_equalized is too high.
* **Position is a cause of the label** (the STRING neighbourhood case above). The biological signal is independent of
  position. Equalizing is the right counterfactual, so share_equalized is exact. Comparisons within strata are less
  diluted than raw comparisons (the AUROC is not collapsible), so share_within is too low.

The true share lies between the two in both cases, so the minimum is a lower bound and the maximum an upper bound.

The table below comes from simulated genomes with a known answer: `seqleak.synthetic.simulate_pairs_consequence`
simulates the first case and `seqleak.synthetic.simulate_pairs` the second. Each mixture is lam x position part +
(1 - lam) x position-free part. Each case has 30 genomes of about 1,500 neighbouring pairs, and the strata are
strand x 17 gap bins. `examples/share_bounds.py` prints the table:

```
position is a CONSEQUENCE of the label (true = within)
score                     AUROC within     eq |   true s_within  s_equal explained |  BOUND  upper
null: oracle (Y + noise)  0.688  0.684  0.579 |  0.000    0.022    0.581     0.579 |  0.022  0.581
null: biology Z = Y+noise 0.761  0.761  0.611 |  0.000    0.002    0.573     0.569 |  0.002  0.573
mixture lam=0.25          0.849  0.761  0.612 |  0.251    0.252    0.680     0.709 |  0.252  0.680
mixture lam=0.5           0.916  0.761  0.612 |  0.373    0.373    0.731     0.813 |  0.373  0.731
mixture lam=0.75          0.943  0.762  0.613 |  0.411    0.409    0.746     0.866 |  0.409  0.746
position only             0.914  0.521  0.513 |  1.000    0.948    0.968     0.997 |  0.948  0.968
position is a CAUSE of the label (true = equalized)
null: z independent       0.650  0.708  0.647 |  0.000   -0.390    0.017     0.009 |  0.000  0.017
mixture lam=0.25          0.742  0.709  0.648 |  0.381    0.136    0.390     0.397 |  0.136  0.390
mixture lam=0.5           0.829  0.710  0.648 |  0.545    0.360    0.548     0.634 |  0.360  0.548
mixture lam=0.75          0.844  0.710  0.648 |  0.565    0.389    0.569     0.799 |  0.389  0.569
position only             0.798  0.527  0.518 |  1.000    0.909    0.940     0.992 |  0.909  0.940
```

### Properties

1. **Reference values.** The share is 0 for a score that carries no position information. It is 0.002 for a score
   that is independent of position given the label, and 0 (clipped from -0.390) for a score that is independent of
   position when position causes the labels. The oracle's 0.022 is sampling noise (its within-strata AUROC is 0.684
   against a raw AUROC of 0.688). The share is about 1 for a function of position: 0.91 to 0.95 here, the remainder
   being position signal left inside the strata (17 gap bins). Row zero reports the share of the position model's
   own predictions as a check of the strata. The share is monotone in the weight of the position part (0.002, 0.252,
   0.373, 0.409, 0.948 in the first case and 0, 0.136, 0.360, 0.389, 0.909 in the second).
2. **Bounded** in [0, 1] by construction. The flag `clipped_low` means that the minimum was below 0, for example for
   a position-independent score when position causes the labels. The flag `clipped_high` means that it was above 1:
   the part of the score that varies within strata ranks against the labels, for example with a within-strata AUROC
   below 0.5.
3. **Stable near chance.** The share is not reported when raw - 1/2 <= 0.02 (flag `raw_near_chance`). Just above the
   margin the interval is wide: at AUROC 0.535 the share is 0.000 [0.000, 0.093]. If more than 2.5% of bootstrap
   replicates have raw <= 1/2, no interval is given (flag `ci_unbounded`).
4. **Rank-based.** The share depends only on the ranks of the scores, so it is invariant to monotone transforms of the
   score.
5. **Intervals.** They come from the same sufficient statistics as the other columns: per-genome Mann-Whitney counts
   of the raw, within and equalized AUROCs, resampled with the table's shared bootstrap plan. For the genome bootstrap
   the resampled counts give exactly the re-scored values (see "Intervals").
6. **Inputs.** Per-item scores, labels, genomes and position strata only.

### Edge cases

* AUPRC has a prevalence-dependent chance level that differs between strata, so the share is defined on the AUROC
  scale only. AUPRC is reported raw and as a gain.
* Multi-class labels: compute it for each class against the rest (or from the macro averages of raw, within and
  equalized AUROC). Continuous labels: the same construction with concordance indices,
  `concordance(y, s, groups=g)`, `concordance(y, s, strata, g)` and `concordance(y, equalized_score(s, strata, g), groups=g)`.
* Few genomes: the block bootstrap resamples blocks of neighbouring items, and the percentiles of the equalized score
  are computed once on the full data.

### Alternative definitions

* **Within share alone**, 1 - (within - 1/2) / (raw - 1/2), is `seqleak.metrics.position_share`, kept for
  compatibility. It is negative, down to -1.1, for a score that ignores position when position causes the labels.
* **Explained share** (`seqleak.metrics.explained_share`) splits raw - 1/2 = sum_k y_k c_k / (P N), with c_k the
  centred rank of the score, into the part carried by the stratum's leave-one-out positive rate and the rest. It is
  0 for a position-independent score when position causes the labels but 0.57 for one when position is a consequence
  of them, and it overstates mixtures in both cases (column `explained` in the table above).
* **Gain share**, 1 - gain / (raw - 1/2), is 0.52-0.82 for a score with no position information, because AUROC gains
  are not additive (`examples/share_definitions.py`).

## Intervals

`seqleak.bootstrap.Resample` draws one resampling plan that every statistic of a table shares, so intervals of
differences and ratios are paired. `method='cluster'` resamples genomes and is the default for >= 10 genomes.
`method='block'` resamples blocks of 50 consecutive items within each genome (at least 20 blocks per genome).
Statistics are computed from sufficient statistics, not by re-scoring items. The cluster bootstrap uses per
(genome, stratum) Mann-Whitney counts, which cost O(genomes) per replicate. The block bootstrap uses the matrices
`M[u, v] = sum_k U(positives of block u, negatives of block v)` and `D[u, v] = sum_k P_uk N_vk`, so that a replicate
with block multiplicities c has AUROC `c'Mc / c'Dc`. Both equal the values obtained by re-scoring the resampled items
(checked in the tests). The intervals are percentile intervals, 95% by default.

## Validation

seqleak reproduces, from their inputs and to the precision printed in the paper, the long-read operon and zero-shot
PPI tables of the final version of *A simple distance rule beats genomic language models at operon prediction*
(doi:10.5281/zenodo.23484461). `posbench/VALIDATION.md` in the archive doi:10.5281/zenodo.23605239 says which values
were compared and how closely they agree. The tests (`tests/test_metrics.py`, `tests/test_features_baseline.py`)
check the metrics against reference implementations and the bootstrap against re-scoring the resampled items. They
run without the paper's data.

For the PPI table, `spec='pair'`, the default, fits the position model to convergence. It gives the same fit on every
machine and thread count, and its position-model AUROC matches the paper's final table. `spec='ppi_v1'` keeps the
solver settings of the paper's original code. That fit stops before convergence and depends on the number of threads,
so it does not reproduce the final table. For the long-read operon table, `spec='operon_v1'` uses the paper's solver
settings and reproduces the table's position model to the printed precision. The 'v1' in both names refers to this
paper.
