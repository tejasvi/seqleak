"""Distance-to-TSS tasks (DNALongBench enhancer-target genes and eQTL, the Long-Range Benchmark's causal eQTL):
distance as row zero, AUROC within distance strata, gain over distance and position share for each model.

Input: a table (CSV or parquet) with one row per evaluated item and the columns ``chrom``, ``pos`` (position of the
element or variant), ``label`` (0/1), ``distance`` (bp to the TSS) and one column per model score. Distance enters
the position model as log10(distance + 1); strata are quantile bins of the out-of-fold position-model prediction.
One genome: the cross-validation folds are five contiguous chunks of it, and every metric of out-of-fold predictions
compares items only within their fold.

    python dnalongbench_distance.py items.csv model_a model_b

To check a padded loader for a leak, add the count of N characters in each model input as a column and score it as
a model: if it matches distance, the inputs encode distance.
"""
import sys

import numpy as np
import pandas as pd

import seqleak

f = sys.argv[1]
items = pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
items['log10_distance'] = np.log10(items['distance'].astype(float) + 1)
order = np.lexsort((items['pos'].to_numpy(), pd.factorize(items['chrom'], sort=True)[0]))     # genome order
scores = {m: items[m].to_numpy(float) for m in sys.argv[2:]}
df = seqleak.evaluate(items['label'].to_numpy(), np.repeat('genome', len(items)), items, scores,
                       position_columns=['log10_distance'], order=order)
print(seqleak.format_table(df))
