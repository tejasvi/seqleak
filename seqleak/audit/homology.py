"""Optional homology check with BLASTN (dc-megablast, or blastn for median length <= 150; both strands; e < 1e-5) or
MMseqs2 (nucleotide search), on a seeded random subsample of test items against the unique fit-split sequences."""
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from . import stats as S
from . import thresholds as TH


def _find(name, given):
    if given:
        p = os.path.join(given, name) if os.path.isdir(given) else given
        return p if os.path.exists(p) else None
    return shutil.which(name)


def check_homology(ctx, th, mode, max_test, threads, blast_bin, mmseqs_bin, workdir, _check, _skip):
    t0 = time.time()
    if mode == 'off':
        return _skip('overlap.homology', '--homology off')
    blastn = _find('blastn', blast_bin)
    makedb = _find('makeblastdb', blast_bin)
    mmseqs = _find('mmseqs', mmseqs_bin)
    tool = None
    if mode in ('auto', 'blast') and blastn and makedb:
        tool = 'blast'
    elif mode in ('auto', 'mmseqs') and mmseqs:
        tool = 'mmseqs'
    if tool is None:
        return _skip('overlap.homology', f'no {"BLAST" if mode == "blast" else "MMseqs2" if mode == "mmseqs" else "BLAST or MMseqs2"} binary found')
    if ctx.E != 1:
        return _skip('overlap.homology', 'multi-element items')
    rng = np.random.default_rng(ctx.seed)
    te = np.flatnonzero(ctx.te)
    sub = np.sort(rng.choice(te, max_test, replace=False)) if len(te) > max_test else te
    uid = ctx.uid[0]
    ref_u = np.unique(uid[ctx.fit])
    seqs = ctx.useq[0]
    base = workdir or os.environ.get('TMPDIR') or os.getcwd()
    os.makedirs(base, exist_ok=True)
    W = tempfile.mkdtemp(prefix='seqleak_hom_', dir=base)
    try:
        with open(os.path.join(W, 'ref.fa'), 'w') as f:
            for u in ref_u:
                f.write(f'>{u}\n{seqs[u]}\n')
        with open(os.path.join(W, 'query.fa'), 'w') as f:
            for i in sub:
                f.write(f'>{i}\n{seqs[uid[i]]}\n')
        env = dict(os.environ, TMPDIR=W)
        cols = ['qseqid', 'sseqid', 'pident', 'length', 'qlen', 'slen', 'qstart', 'qend', 'sstart', 'send', 'evalue',
                'bitscore']
        out = os.path.join(W, 'hits.tsv')
        med = float(np.median([len(seqs[uid[i]]) for i in sub])) if len(sub) else 0
        if tool == 'blast':
            subprocess.run([makedb, '-in', os.path.join(W, 'ref.fa'), '-dbtype', 'nucl', '-out', os.path.join(W, 'ref')],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            task, extra = ('blastn', ['-word_size', '11']) if med <= 150 else ('dc-megablast', [])
            subprocess.run([blastn, '-task', task, '-query', os.path.join(W, 'query.fa'), '-db', os.path.join(W, 'ref'),
                            '-outfmt', '6 ' + ' '.join(cols), '-evalue', '1e-5', '-max_target_seqs', '50',
                            '-num_threads', str(threads), '-dust', 'no', '-out', out] + extra,
                           check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            detail = f'BLASTN {task}'
        else:
            subprocess.run([mmseqs, 'easy-search', os.path.join(W, 'query.fa'), os.path.join(W, 'ref.fa'), out,
                            os.path.join(W, 'tmp'), '--search-type', '3', '--threads', str(threads), '-e', '1e-5',
                            '--format-output', 'query,target,pident,alnlen,qlen,tlen,qstart,qend,tstart,tend,evalue,bits'],
                           check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            detail = 'MMseqs2 easy-search (nucleotide)'
        h = pd.read_csv(out, sep='\t', names=cols) if os.path.getsize(out) > 0 else pd.DataFrame(columns=cols)
    finally:
        shutil.rmtree(W, ignore_errors=True)
    if tool == 'mmseqs' and len(h) and h['pident'].max() <= 1.0:
        h['pident'] = h['pident'] * 100
    h['qcov'] = (h['qend'] - h['qstart']).abs().add(1) / h['qlen']
    h = h.sort_values(['qseqid', 'bitscore'], ascending=[True, False])
    best = h.drop_duplicates('qseqid').set_index('qseqid')
    pid = best['pident'].reindex(sub).fillna(0).to_numpy()
    qc = best['qcov'].reindex(sub).fillna(0).to_numpy()
    homo = (pid >= th['homolog_id']) & (qc >= th['homolog_cov'])
    near = (pid >= th['near_id']) & (qc >= th['near_cov'])
    sh = S.share(homo)
    ev = dict(tool=detail, n_subsample=int(len(sub)), near_duplicate_share=float(np.mean(near)) if len(sub) else None,
              any_hit=float(np.mean(pid > 0)) if len(sub) else None)
    if ctx.classification and len(best):
        fit_first = pd.Series(np.flatnonzero(ctx.fit), index=uid[ctx.fit])
        fit_first = fit_first[~fit_first.index.duplicated()]
        s_item = fit_first.reindex(best['sseqid'].astype(np.int64).to_numpy()).to_numpy()
        ok = np.isfinite(s_item.astype(float))
        blab = pd.Series(np.where(ok, ctx.yi[np.nan_to_num(s_item.astype(float), nan=0).astype(np.int64)], -1),
                         index=best.index)
        hit_rows = sub[homo]
        if len(hit_rows):
            ev['homolog_label_agreement'] = float(np.mean(blab.reindex(hit_rows).to_numpy() == ctx.yi[hit_rows]))
    st = TH.status(sh['value'], th)
    return _check('overlap.homology', st, 'share of test items with a fit-split hit of >= 80% identity over >= 50% of '
                  'the query', sh['value'], sh['ci'], sh['n'], th,
                  f"{sh['k']} of {sh['n']} sampled test items have a homologous training/dev sequence ({detail})", ev, [],
                  time.time() - t0)
