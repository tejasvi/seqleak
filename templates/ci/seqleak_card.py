#!/usr/bin/env python3
"""seqleak_card: one-page report cards from seqleak audit reports.

Reads one or more report.json files written by ``seqleak audit`` (schema seqleak.audit/1, or posbench.audit/1 from posbench, its former name) and writes

  --html FILE     a self-contained one-page HTML card (A4; no scripts, fonts or images to load; prints as one page)
  --md FILE       the same card in GitHub-flavoured Markdown (pull requests, issues, CI job summaries)
  --section FILE  a "Leakage audit" section for a dataset card (README.md or a Hugging Face dataset card)

With several reports the card summarises all of them (one row per report), followed by the findings of each.
Standard library only (Python >= 3.8); seqleak itself is not imported.

usage:
  python seqleak_card.py report.json --html card.html --md card.md
  python seqleak_card.py gue/*/report.json --title "GUE, 28 tasks" --html gue.html --md gue.md
  python seqleak_card.py report.json --section leakage_audit.md --date 2026-10-22
  python seqleak_card.py --section-template            # the section template (fields to fill), on stdout
  python seqleak_card.py reports/*.json --md summary.md --fail-on fail   # exit 1 if any report is FAIL
"""
import argparse
import hashlib
import html
import json
import os
import shlex
import string
import sys

SCHEMA_PREFIXES = ('seqleak.audit/', 'posbench.audit/')   # posbench.audit/: reports of posbench, the former name
# rule set of each frozen threshold set: sha256 of the thresholds as canonical JSON (keys sorted, separators ',' and ':')
KNOWN_RULES = {
    'f6d21ddf703698b98a3190e921c5f38b41f72bfd9d70eee468e93ce501e09063': 'v1.0',
    '28415c8a1e8e65a4b896c543710fcdc10ab96d43540800d9bbb9a5a5f5785884': 'v1.1 or v1.1.1',
    '73e21c0c82a5fe7bb87921f4f2f1c135cc3e16eb484ccfcaf7a4c8ca88127244': 'v1.2 or v1.2.1',
}
STATUSES = ('FAIL', 'WARN', 'PASS', 'SKIP', 'INFO')
ICON = dict(FAIL='\u2715', WARN='!', PASS='\u2713', SKIP='\u2013', INFO='i')
PATH_OPTS = ('--blast-bin', '--mmseqs-bin', '--workdir', '--genome-registry', '--minimap2-bin')   # machine-specific: dropped from the portable command
OUT_OPTS = ('--out', '--md', '--items-out')

# What to do when a check fails (or warns). Fixed text, one entry per check id.
ADVICE = {
    'overlap.exact': 'Test items have exact or reverse-complement copies in the training data. Remove them from the '
                     'test set, or re-split so that identical sequences stay in one split (seqleak clean).',
    'overlap.near': 'Test items share most of their k-mers with a training sequence. Keep each group of near-duplicates '
                    'in one split (seqleak clean --mode resplit groups them).',
    'overlap.windows': 'Test windows overlap, or are shifted copies of, training windows. Split by genomic locus or '
                       'chromosome so that overlapping windows stay together (seqleak clean --group-by locus).',
    'overlap.pair': 'Every element of these multi-sequence items is copied from one training item. Split by element, '
                    'so that no element recurs across splits.',
    'overlap.homology': 'Test items have close homologs in the training data. Cluster by sequence identity and keep '
                        'each cluster in one split.',
    'reference.copies': "Items are copies of another benchmark's items, which models trained on that benchmark have "
                        'seen. Remove them, or report scores with and without them.',
    'shortcut.length': 'Sequence length or padding predicts the label. Give both classes the same length and padding.',
    'shortcut.ambiguous': 'N runs or ambiguity codes predict the label. Treat them the same way in every class, or '
                          'drop the windows that contain them.',
    'shortcut.case': 'Lowercase (soft-masked) letters predict the label. Uppercase every sequence.',
    'shortcut.lowinfo': 'Test items made mostly of N carry no sequence to learn from, and their label is often one '
                        'class. Drop windows with less than half A/C/G/T from every split, or report scores with and '
                        'without them.',
    'shortcut.composition': 'A GC or CpG rule separates the classes. This can be real biology: report the rule as the '
                            'baseline a model must beat.',
    'shortcut.kmer': 'A k-mer composition rule separates the classes. This can be real biology: report the rule as the '
                     'baseline a model must beat.',
    'shortcut.position_motif': 'A few k-mers at fixed positions predict the label (for example a consensus at a fixed '
                               'offset). If the motif is what the task tests, report the table as a baseline; '
                               'otherwise vary where the windows are placed.',
    'construction.shuffle': 'Negatives are dinucleotide shuffles of positives, and twin pairs cross the split. Keeping '
                            'twins in one split (seqleak clean) removes the cross-split leak, but the negatives stay '
                            'constructed: genomic background windows matched on length and GC remove the shortcut.',
    'construction.segment': 'Negatives were made by substituting segments of positives, and twin pairs cross the split. '
                            'Keeping twins in one split (seqleak clean) removes the cross-split leak, but the negatives '
                            'stay constructed: genomic background windows matched on length and GC remove the shortcut.',
    'construction.synthetic_negatives': 'Negatives are dinucleotide shuffles or segment substitutions of positives, '
                                        'anywhere in the data (within a split too). A re-split does not remove them: '
                                        'replace them with genomic background windows matched on length and GC.',
    'construction.unmatched_site': 'One class carries a site at a fixed position (a motif or consensus) that the other '
                                   'class lacks, so the site alone gives the label. Draw negatives that carry the site '
                                   'at the same position, or report the site rule as the baseline a model must beat.',
    'construction.genome_membership': 'Membership in the reference genome alone gives the label: the negatives are not genome '
                                      'sequence (shuffled, substituted or otherwise generated). Draw negatives from the genome '
                                      '(windows matched on length and GC), or report the membership rule as a baseline.',
    'shortcut.anchored_motif': 'A few k-mers at fixed positions from the item ends predict the label (for example a '
                               'consensus at the start of variable-length items). Vary where the items start and end, or '
                               'report the table as a baseline.',
    'construction.library': 'Constant segments shared by groups of items (adapters, barcodes, vector backbones) predict '
                            'the label: the groups come from different libraries. Trim the constant segments, or split '
                            'and score within one library.',
    'position.distance': 'Position alone (the distance fields) predicts the label. Report the position-only model as a '
                         'baseline and score models with position held fixed.',
    'position.row_zero': 'Gene position alone predicts the label. Report the position-only baseline (row zero) above '
                         'the models and score them with position held fixed.',
}
# checks whose remedy is the same: when several of them FAIL in one card, their advice is given once, for the group
ADVICE_GROUPS = (
    (('construction.shuffle', 'construction.segment', 'construction.synthetic_negatives', 'construction.genome_membership'),
     'The negatives were constructed from the positives (shuffled or substituted) rather than drawn from the genome. A '
     're-split removes the twins across the split, not the construction: genomic background windows matched on length '
     'and GC remove the shortcut.'),
)


def advice_items(cids):
    """(check ids, advice) for the failing check ids, in order: one item per group with two or more members failing,
    otherwise one per check."""
    out, seen = [], set()
    for cid in cids:
        if cid in seen or cid not in ADVICE:
            continue
        grp = next((g for g in ADVICE_GROUPS if cid in g[0]), None)
        members = [c for c in cids if grp and c in grp[0]]
        if grp and len(members) > 1:
            out.append((members, grp[1]))
            seen.update(members)
        else:
            out.append(([cid], ADVICE[cid]))
            seen.add(cid)
    return out


READING = ('FAIL marks a leak at the level that manual audits of DNA benchmarks count as a defect, such as copies of '
           'test items in the training data, negatives constructed from positives, artefact or motif shortcuts, or '
           'position confounds. WARN marks the same signals below that level, or a composition baseline that a model '
           'must beat. It is common on tasks without a documented leak, so treat it as a reason to look closer. PASS '
           'means that no check reached its warning threshold, and SKIP that the input lacks what the check needs '
           '(coordinates, distance fields, an aligner, or items with several sequences).')
# check ids of the rule set this card was released with (the blank section template describes that rule set)
KIT_CHECKS = (
    'baseline.lookup', 'construction.genome_membership', 'construction.library', 'construction.segment',
    'construction.shuffle', 'construction.synthetic_negatives', 'construction.unmatched_site', 'overlap.exact',
    'overlap.homology', 'overlap.near', 'overlap.pair', 'overlap.windows', 'position.distance', 'position.row_zero',
    'reference.copies', 'shortcut.ambiguous', 'shortcut.anchored_motif', 'shortcut.case', 'shortcut.composition',
    'shortcut.kmer', 'shortcut.length', 'shortcut.lowinfo', 'shortcut.position_motif',
)


def limits(reps=None):
    """What the linter does not see. Pooled-library structure has a check (construction.library) from rules v1.1 on,
    genome membership (construction.genome_membership, with --genome) from v1.2 on."""
    lib = [('construction.library' in r['checks']) for r in reps] if reps else ['construction.library' in KIT_CHECKS]
    mem = [('construction.genome_membership' in r['checks']) for r in reps] if reps else ['construction.genome_membership' in KIT_CHECKS]
    if all(mem) and all(lib):      # rules v1.2 on: membership is checked when a reference genome is given
        return ('The linter reports only what its checks measure. It checks position only when distance fields are given, '
                'copies of other benchmarks only with --reference, membership in a reference genome only with --genome, and '
                'has no check for transductive arrangements; see "Known limitations" in the seqleak README.')
    if all(lib):
        miss = 'has no check of membership in a reference genome or for transductive arrangements'
    elif any(lib):
        miss = ('has no check of membership in a reference genome or for transductive arrangements (nor for '
                'pooled-library structure, in the reports without construction.library)')
    else:
        miss = 'has no check for pooled-library structure or transductive arrangements'
    return ('The linter reports only what its checks measure. It checks position only when distance fields are given, '
            f'copies of other benchmarks only with --reference, and {miss}; see "Known limitations" in the seqleak README.')


SECTION_TEMPLATE = """## Leakage audit

The splits in this repository were checked for data leakage with `${tool} audit` (${rules}; ${version}).
${date_line}
**Verdict: ${verdict}.** ${summary}.

${table}

${baselines}

How to read the verdict. ${reading}

${limits}

To reproduce:

```
${command}
```
"""


# ------------------------------------------------------------------------------------------------ helpers
def load(path):
    with open(path, encoding='utf-8') as f:
        rep = json.load(f)
    if not str(rep.get('schema', '')).startswith(SCHEMA_PREFIXES):
        raise SystemExit(f'{path}: not a seqleak audit report (schema {rep.get("schema")!r})')
    return rep


def rules_label(rep):
    for part in ('tool', 'settings'):          # a report may record its rule set
        rules = (rep.get(part) or {}).get('rules')
        if isinstance(rules, str) and rules:
            return 'linter rules ' + rules
    th = (rep.get('settings') or {}).get('thresholds')
    if th is None:
        return 'thresholds not recorded'
    h = hashlib.sha256(json.dumps(th, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return 'linter rules ' + KNOWN_RULES[h] if h in KNOWN_RULES else 'custom thresholds (not a frozen rule set)'


def tool_label(rep):
    t = rep.get('tool') or {}
    return f"{t.get('name') or 'seqleak'} {t.get('version')}"


def num(v, d=3):
    if v is None:
        return ''
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f'{v:,}'
    if isinstance(v, float):
        return f'{v:.{d}f}'
    return str(v)


def ci(c):
    if not c or c[0] is None or c[1] is None:
        return ''
    return f' [{c[0]:.3f}, {c[1]:.3f}]'


def thr(t):
    if not t:
        return ''
    return f"{num(t.get('warn')) or '-'} / {num(t.get('fail')) or '-'}"


METRIC = dict(mcc='MCC', acc='accuracy', f1='F1', auroc='AUROC')


def metric(m):
    return METRIC.get(str(m), str(m))


def cap(text):
    text = str(text or '')
    return text[:1].upper() + text[1:]


def lc_title(t):
    """Lowercase the first letter of a check title unless it starts an acronym (GC, N)."""
    t = str(t)
    return t[:1].lower() + t[1:] if len(t) > 1 and t[1:2].islower() else t


def counts(rep):
    c = {s: 0 for s in STATUSES}
    for r in rep['checks'].values():
        c[r['status']] = c.get(r['status'], 0) + 1
    return c


TWIN = {'construction.segment': 'positional (segment-substitution) twin',
        'construction.shuffle': 'composition (dinucleotide-shuffle) twin'}


def rule_text(rep, family):
    """Plain description of a best-rule family id (seqleak.audit.run.best_rule)."""
    titles = {cid: r.get('title', cid) for cid, r in rep['checks'].items()}

    def one(f):
        if f == 'lookup':
            return 'label of the nearest training sequence'
        cid, _, nm = f.partition(':')
        if cid in titles:
            t = lc_title(titles[cid])
            word = nm.replace('_', ' ').rstrip('0123456789').strip().lower()
            if nm and not (word and word in t.lower() and '/' not in t):
                t += f', feature {nm}'
            return t
        return f

    if family.startswith('lookup_if_near_else:'):
        return 'label of the nearest training sequence when it is a near copy, otherwise the ' + one(family.split(':', 1)[1])
    if family.startswith('twin:'):
        cid, _, rest = family[5:].partition('_else')
        tail = 'the majority class' if rest == '_majority' else 'the ' + one(rest.lstrip(':'))
        return f'label implied by a training {TWIN.get(cid, "twin (" + cid + ")")}, otherwise {tail}'
    return one(family)


def portable_command(rep):
    """The audit command with file paths reduced to file names and machine-specific options dropped."""
    argv = list((rep.get('tool') or {}).get('argv') or [])
    if not argv:
        return 'seqleak audit --train train.csv --dev dev.csv --test test.csv --out report.json'
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a in PATH_OPTS or a == '--quiet':
            skip = a in PATH_OPTS
            continue
        prev = argv[i - 1] if i else ''
        if prev in OUT_OPTS:
            a = os.path.basename(a)
        elif '/' in a and not a.startswith('-'):
            a = (a.split('=', 1)[0] + '=' + os.path.basename(a.split('=', 1)[1])) if '=' in a else os.path.basename(a)
        out.append(a)
    return ' '.join(shlex.quote(a) for a in out)


def input_names(rep):
    ins = (rep.get('task') or {}).get('inputs') or []
    return [(os.path.basename(str(x.get('path', ''))), (x.get('sha256') or '')[:12]) for x in ins]


def findings(rep, statuses=('FAIL', 'WARN')):
    return [(cid, r) for s in statuses for cid, r in rep['checks'].items() if r['status'] == s]


def homology_note(rep):
    r = rep['checks'].get('overlap.homology')
    if not r:
        return ''
    if r['status'] == 'SKIP':
        return 'homology check skipped (' + str(r.get('reason', '')).rstrip('.') + ')'
    return 'homology check run'


# ------------------------------------------------------------------------------------------------ HTML
CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--base:#c3c2b7;--border:rgba(11,11,11,.10);--good:#0ca30c;--warning:#fab219;--critical:#d03b3b}
*{box-sizing:border-box}
html{background:var(--page)}
body{margin:0 auto;max-width:210mm;padding:4mm 11mm 3mm;background:var(--surface);color:var(--ink);
font:9.2pt/1.36 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-print-color-adjust:exact;print-color-adjust:exact}
h1{font-size:17pt;line-height:1.15;margin:2px 0 3px;word-break:break-word}
h2{font-size:10.5pt;margin:5px 0 3px;padding-bottom:2px;border-bottom:1px solid var(--grid)}
.eyebrow{color:var(--ink2);font-size:8pt;letter-spacing:.06em;text-transform:uppercase}
.sub{color:var(--ink2);font-size:8.5pt}
.head{display:flex;gap:14px;align-items:flex-start;justify-content:space-between}
.verdict{min-width:58mm;max-width:74mm;border:1px solid var(--border);border-left:6px solid var(--muted);border-radius:6px;
padding:6px 9px;background:#fff}
.verdict .word{font-size:19pt;font-weight:700;display:flex;flex-wrap:wrap;align-items:center;gap:8px;line-height:1.1}
.verdict .note{color:var(--ink2);font-size:8pt;margin-top:2px}
.v-FAIL{border-left-color:var(--critical)}.v-WARN{border-left-color:var(--warning)}.v-PASS{border-left-color:var(--good)}
.dot{display:inline-flex;align-items:center;justify-content:center;width:1.35em;height:1.35em;border-radius:50%;
font-weight:700;font-size:.82em;line-height:1;color:#fff;background:var(--muted);flex:none}
.verdict .dot{font-size:12pt}
.d-FAIL{background:var(--critical)}.d-WARN{background:var(--warning);color:var(--ink)}.d-PASS{background:var(--good)}
.d-SKIP{background:var(--muted)}.d-INFO{background:#fff;color:var(--ink2);box-shadow:inset 0 0 0 1.5px var(--ink2)}
.chip{display:inline-flex;align-items:center;gap:4px;font-weight:600;font-size:8.5pt;white-space:nowrap}
.tiles{display:grid;grid-template-columns:.85fr 1fr 1.45fr 1fr;gap:6px;margin:8px 0 2px}
.tile{border:1px solid var(--border);border-radius:6px;padding:5px 8px;background:#fff}
.tile .lab{color:var(--ink2);font-size:7.5pt;letter-spacing:.05em;text-transform:uppercase}
.tile .val{font-size:15pt;font-weight:600;line-height:1.2}
.tile .val.small{font-size:10.5pt;line-height:1.5}
.tile .s{color:var(--ink2);font-size:7.5pt;line-height:1.3}
.finding{border-left:3px solid var(--muted);padding:1px 0 1px 8px;margin:2px 0;break-inside:avoid}
.f-FAIL{border-left-color:var(--critical)}.f-WARN{border-left-color:var(--warning)}
.finding .t{font-weight:600}.finding .v{font-variant-numeric:tabular-nums;color:var(--ink)}
.finding .th{color:var(--ink2);font-size:8pt}
.finding .r{color:var(--ink2);font-size:8pt}
.finding .a{font-size:8pt}
table{border-collapse:collapse;width:100%;font-size:8.1pt}
th{color:var(--ink2);font-weight:600;text-align:left;border-bottom:1px solid var(--base);padding:2px 5px}
td{border-bottom:1px solid var(--grid);padding:1px 5px;vertical-align:top;line-height:1.24}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td code{font-size:7.5pt;color:var(--ink2)}
.muted{color:var(--muted)}
td.task{min-width:42mm;overflow-wrap:anywhere}
td.ids{font-size:7.6pt}
.skip{font-size:7.8pt;color:var(--ink2);margin:4px 0 0}
.foot{margin-top:4px;color:var(--ink2);font-size:7pt;line-height:1.26}
.foot p{margin:2px 0}
@page{size:A4;margin:0}
@media print{html{background:none}body{max-width:none}}
@media (forced-colors:active){.dot{border:1px solid CanvasText}}
"""


def e(s):
    return html.escape(str(s), quote=True)


def chip(status):
    return f'<span class="chip"><span class="dot d-{e(status)}" aria-hidden="true">{ICON.get(status, "?")}</span>{e(status)}</span>'


def html_single(rep, title=None):
    t, v = rep['task'], rep['verdict']
    n = t.get('n') or {}
    c = counts(rep)
    lb = rep.get('lookup_baseline') or {}
    br = rep.get('best_rule')
    H = []
    w = H.append
    w('<header class="head"><div>')
    w('<div class="eyebrow">seqleak leakage audit &middot; report card</div>')
    w(f'<h1>{e(title or t.get("name", "task"))}</h1>')
    classes = ', '.join(str(x) for x in (t.get('classes') or [])) or '-'
    ln = t.get('length') or {}
    w(f'<div class="sub">{e(t.get("task_type", ""))} &middot; classes {e(classes)} &middot; length {e(ln.get("min"))}-{e(ln.get("max"))} '
      f'(median {e(num(ln.get("median"), 0))}) &middot; elements {e(", ".join(t.get("elements") or []))} &middot; coordinates '
      f'{"yes" if t.get("has_coords") else "no"} &middot; distance fields {"yes" if t.get("has_distance") else "no"}</div>')
    w('</div>')
    note = v.get('summary', '')
    if len(note) > 90 and v.get('warn'):
        nw = len(v['warn'])
        note = (f'leak checks failing: {", ".join(v.get("fail") or [])}; ' if v.get('fail') else '') + \
            f'{nw} warning{"s" if nw > 1 else ""}, listed below'
    w(f'<div class="verdict v-{e(v["status"])}"><div class="word"><span class="dot d-{e(v["status"])}" aria-hidden="true">'
      f'{ICON.get(v["status"], "?")}</span>{e(v["status"])}</div><div class="note">{e(note)}</div></div>')
    w('</header>')
    # tiles
    w('<section class="tiles">')
    w(f'<div class="tile"><div class="lab">Items</div><div class="val">{e(num(n.get("test")))}</div>'
      f'<div class="s">test items; train {e(num(n.get("train")))}, dev {e(num(n.get("dev")))}</div></div>')
    if lb.get('value') is not None:
        w(f'<div class="tile"><div class="lab">Nearest-training lookup</div><div class="val">{e(num(lb["value"]))}</div>'
          f'<div class="s">{e(metric(lb.get("metric", "")))}{e(ci(lb.get("ci")))}; chance {e(num(lb.get("chance")))}. Label of the most '
          f'similar training sequence</div></div>')
    else:
        w('<div class="tile"><div class="lab">Nearest-training lookup</div><div class="val muted">-</div><div class="s">not computed</div></div>')
    if br:
        extra = ''
        if br.get('published_best') is not None:
            extra = f'; published best {num(br["published_best"])}'
        w(f'<div class="tile"><div class="lab">Best no-model rule</div><div class="val">{e(num(br.get("value")))}</div>'
          f'<div class="s">{e(metric(br.get("metric", "")))}{e(ci(br.get("ci")))}; chance {e(num(br.get("chance")))}{e(extra)}. '
          f'Chosen on test (an upper bound): {e(rule_text(rep, br.get("family", "")))}</div></div>')
    else:
        w('<div class="tile"><div class="lab">Best no-model rule</div><div class="val muted">-</div><div class="s">no rule (regression labels)</div></div>')
    w('<div class="tile"><div class="lab">Checks</div><div class="val small">'
      + ' &nbsp;'.join(f'{chip(s)}&nbsp;{c[s]}' for s in ('FAIL', 'WARN', 'PASS') if c.get(s)) +
      f'</div><div class="s">{c.get("SKIP", 0)} skipped (input lacks what they need); {e(homology_note(rep))}</div></div>')
    w('</section>')
    # findings
    fs = findings(rep)
    w('<h2>What the audit found</h2>')
    if not fs:
        w('<p class="sub">No check reached its warning threshold.</p>')
    adv = {tuple(c)[0]: (c, a) for c, a in advice_items([cid for cid, r in fs if r['status'] == 'FAIL'])}
    for cid, r in fs:
        head = (f'{chip(r["status"])} <span class="t">{e(r.get("title", cid))}</span> <code class="muted">{e(cid)}</code> '
                f'&nbsp;<span class="v">{e(num(r.get("value")))}{e(ci(r.get("ci")))}</span> '
                f'<span class="th">&nbsp;warn / fail {e(thr(r.get("threshold")))}</span>')
        if r['status'] == 'FAIL':
            w(f'<div class="finding f-FAIL"><div>{head}</div>')
            w(f'<div class="r">{e(cap(r.get("statistic", "")))}: {e(r.get("reason", ""))}</div>')
            if cid in adv:
                cs, a = adv[cid]
                also = f' (also for {", ".join(cs[1:])})' if len(cs) > 1 else ''
                w(f'<div class="a"><b>What to do{e(also)}:</b> {e(a)}</div>')
        else:
            w(f'<div class="finding f-{e(r["status"])}"><div>{head} <span class="r">&mdash; {e(r.get("reason", ""))}</span></div>')
        w('</div>')
    # the other checks that ran (every FAIL and WARN is a finding above; skipped checks are listed below)
    ok = [(cid, r) for cid, r in rep['checks'].items() if r['status'] not in ('FAIL', 'WARN', 'SKIP', 'INFO')]
    if ok:
        w('<h2>Checks that passed</h2><table><thead><tr><th>check</th><th>status</th><th class="n">value [95% CI]</th>'
          '<th class="n">warn / fail</th></tr></thead><tbody>')
        for cid, r in ok:
            val = (num(r.get('value')) + ci(r.get('ci'))) if r.get('value') is not None else ''
            w(f'<tr><td>{e(r.get("title", cid))} <code>{e(cid)}</code></td><td>{chip(r["status"])}</td><td class="n">{e(val)}</td>'
              f'<td class="n">{e(thr(r.get("threshold")))}</td></tr>')
        w('</tbody></table>')
    sk = [(cid, r) for cid, r in rep['checks'].items() if r['status'] == 'SKIP']
    if sk:
        w('<p class="skip">' + chip('SKIP') + ' ' + '; '.join(
            f'<code>{e(cid)}</code> ({e(str(r.get("reason", "")).rstrip("."))})' for cid, r in sk) + '.</p>')
    w(footer_html(rep))
    return page(title or t.get('name', 'task'), H)


def footer_html(rep, reps=None):
    reps = reps or [rep]
    rl = sorted({rules_label(r) for r in reps})
    ver = sorted({tool_label(r) for r in reps})
    line = f'Rules: {", ".join(rl)}; {", ".join(ver)}; report schema {rep.get("schema")}.'
    if len(reps) > 1:              # several reports: the rule-set line ends the reading paragraph
        return '\n'.join(['<div class="foot">', f'<p><b>How to read the verdict.</b> {e(READING)} {e(limits(reps))} '
                                                 f'{e(line)} Card: seqleak_card.py.</p>', '</div>'])
    F = ['<div class="foot">', f'<p><b>How to read the verdict.</b> {e(READING)} {e(limits(reps))}</p>']
    if len(reps) == 1:
        ins = '; '.join(f'{n} (sha256 {h})' for n, h in input_names(rep))
        rt = rep.get('runtime') or {}
        line += f' Inputs: {ins or "-"}. Runtime {rt.get("seconds")} s, peak memory {rt.get("peak_rss_mb")} MB.'
        line = e(line) + f' Command: <code>{e(portable_command(rep))}</code>. Card: seqleak_card.py.'
        F.append(f'<p>{line}</p>')
    else:
        F.append(f'<p>{e(line)} Card: seqleak_card.py.</p>')
    F.append('</div>')
    return '\n'.join(F)


def page(title, body):
    return ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>seqleak audit: {e(title)}</title><style>{CSS}</style></head><body>\n' + '\n'.join(body) +
            '\n</body></html>\n')


def html_multi(reps, title=None):
    H = []
    w = H.append
    tot = {s: sum(1 for r in reps if r['verdict']['status'] == s) for s in ('FAIL', 'WARN', 'PASS')}
    worst = 'FAIL' if tot['FAIL'] else ('WARN' if tot['WARN'] else 'PASS')
    w('<header class="head"><div><div class="eyebrow">seqleak leakage audit &middot; report card</div>')
    w(f'<h1>{e(title or f"{len(reps)} audited tasks")}</h1>')
    w(f'<div class="sub">{len(reps)} reports; task verdict = the worst check verdict</div></div>')
    w(f'<div class="verdict v-{worst}"><div class="word">' + ' &nbsp;'.join(
        f'<span class="chip" style="font-size:11pt">{chip(s)}&nbsp;{tot[s]}</span>' for s in ('FAIL', 'WARN', 'PASS')) +
      '</div><div class="note">tasks per verdict</div></div></header>')
    w('<h2>Tasks</h2><table><thead><tr><th>task</th><th>verdict</th><th>failing checks</th><th>warnings</th>'
      '<th class="n">lookup (chance)<br>[95% CI]</th><th class="n">best rule</th><th class="n">test items</th></tr></thead><tbody>')
    for r in reps:
        lb = r.get('lookup_baseline') or {}
        br = r.get('best_rule') or {}
        lk = (f'{e(num(lb.get("value")))} ({e(num(lb.get("chance")))})<br><span class="muted">{e(ci(lb.get("ci")).strip())}</span>'
              if lb.get('value') is not None else '-')
        name = e(r['task'].get('name')).replace('@', '@<wbr>').replace('__', '__<wbr>')
        w(f'<tr><td class="task">{name}</td><td>{chip(r["verdict"]["status"])}</td>'
          f'<td class="ids">{e(", ".join(r["verdict"]["fail"]) or "-")}</td><td class="ids">{e(", ".join(r["verdict"]["warn"]) or "-")}</td>'
          f'<td class="n">{lk}</td><td class="n">{e(num(br.get("value")) or "-")}</td>'
          f'<td class="n">{e(num((r["task"].get("n") or {}).get("test")))}</td></tr>')
    w('</tbody></table>')
    w('<h2>Failing checks</h2>')
    any_fail = False
    for r in reps:
        for cid, x in findings(r, ('FAIL',)):
            any_fail = True
            w(f'<div class="finding f-FAIL"><div>{chip("FAIL")} <span class="t">{e(r["task"].get("name"))}</span>: '
              f'<code>{e(cid)}</code> &nbsp;<span class="v">{e(num(x.get("value")))}'
              f'{e(ci(x.get("ci")))}</span></div>'      # thresholds: on the single-task cards
              f'<div class="r">{e(x.get("reason", ""))}</div></div>')
    if not any_fail:
        w('<p class="sub">No task has a failing check.</p>')
    advice = advice_items(sorted({cid for r in reps for cid, _ in findings(r, ('FAIL',)) if cid in ADVICE}))
    if advice:
        w('<h2>What to do</h2>')
        for cs, a in advice:
            w(f'<div class="finding"><span class="t">{e(", ".join(cs))}</span>: <span class="a">{e(a)}</span></div>')
    w(footer_html(reps[0], reps))
    return page(title or f'{len(reps)} tasks', H)


# ------------------------------------------------------------------------------------------------ Markdown
def md_cell(s):
    return str(s).replace('|', '\\|').replace('\n', ' ')


def md_single(rep, title=None):
    t, v = rep['task'], rep['verdict']
    n = t.get('n') or {}
    lb = rep.get('lookup_baseline') or {}
    br = rep.get('best_rule')
    c = counts(rep)
    L = [f'### seqleak audit: {title or t.get("name")}', '',
         f'**Verdict: {v["status"]}.** {cap(v.get("summary", ""))}.', '']
    ln = t.get('length') or {}
    L.append(f'Items: train {num(n.get("train"))}, dev {num(n.get("dev"))}, test {num(n.get("test"))}; '
             f'{t.get("task_type", "")}, classes {", ".join(str(x) for x in (t.get("classes") or [])) or "-"}; length '
             f'{ln.get("min")}-{ln.get("max")}. Checks: ' + ', '.join(f'{c[s]} {s}' for s in STATUSES if c.get(s)) + '.')
    L.append('')
    if lb.get('value') is not None:
        L.append(f'- Nearest-training lookup (label of the most similar training sequence): {metric(lb.get("metric"))} '
                 f'{num(lb["value"])}{ci(lb.get("ci"))}; chance {num(lb.get("chance"))}.')
    if br:
        s = (f'- Best no-model rule, chosen on test (an upper bound): {metric(br.get("metric"))} {num(br.get("value"))}'
             f'{ci(br.get("ci"))}; {rule_text(rep, br.get("family", ""))}')
        if br.get('published_best') is not None:
            s += f'; published best {num(br["published_best"])}'
        L.append(s + '.')
    L.append('')
    fs = findings(rep)
    if fs:
        L += ['| | check | value [95% CI] | warn / fail | finding |', '|---|---|---|---|---|']
        for cid, r in fs:
            L.append(f'| **{r["status"]}** | {md_cell(r.get("title", cid))} (`{cid}`) | {num(r.get("value"))}{ci(r.get("ci"))} | '
                     f'{thr(r.get("threshold"))} | {md_cell(r.get("reason", ""))} |')
        L.append('')
        adv = [cid for cid, r in fs if r['status'] == 'FAIL' and cid in ADVICE]
        if adv:
            L.append('What to do:')
            L += [f'- `{cid}`: {ADVICE[cid]}' for cid in adv]
            L.append('')
    else:
        L += ['No check reached its warning threshold.', '']
    L += ['<details><summary>All checks</summary>', '', '| check | status | value [95% CI] | warn / fail |', '|---|---|---|---|']
    for cid, r in rep['checks'].items():
        val = (num(r.get('value')) + ci(r.get('ci'))) if r.get('value') is not None else (
            md_cell(r.get('reason', '')) if r['status'] == 'SKIP' else '')
        L.append(f'| {md_cell(r.get("title", cid))} (`{cid}`) | {r["status"]} | {val} | {thr(r.get("threshold"))} |')
    L += ['', '</details>', '']
    L.append(f'{rules_label(rep).capitalize()}; {tool_label(rep)}; schema {rep.get("schema")}. '
             f'Reading: {READING}')
    return '\n'.join(L) + '\n'


def md_multi(reps, title=None):
    tot = {s: sum(1 for r in reps if r['verdict']['status'] == s) for s in ('FAIL', 'WARN', 'PASS')}
    L = [f'### seqleak audit: {title or f"{len(reps)} tasks"}', '',
         f'Task verdicts: {tot["FAIL"]} FAIL, {tot["WARN"]} WARN, {tot["PASS"]} PASS.', '',
         '| task | verdict | failing checks | warnings | lookup [95% CI] (chance) | best rule | test items |',
         '|---|---|---|---|---|---|---|']
    for r in reps:
        lb = r.get('lookup_baseline') or {}
        br = r.get('best_rule') or {}
        lk = f'{num(lb.get("value"))}{ci(lb.get("ci"))} ({num(lb.get("chance"))})' if lb.get('value') is not None else '-'
        L.append(f'| {md_cell(r["task"].get("name"))} | **{r["verdict"]["status"]}** | {", ".join(r["verdict"]["fail"]) or "-"} | '
                 f'{", ".join(r["verdict"]["warn"]) or "-"} | {lk} | {num(br.get("value")) or "-"} | '
                 f'{num((r["task"].get("n") or {}).get("test"))} |')
    L.append('')
    fl = [(r, cid, x) for r in reps for cid, x in findings(r, ('FAIL',))]
    if fl:
        L.append('Failing checks:')
        L += [f'- {r["task"].get("name")}: {x.get("title", cid)} (`{cid}`) {num(x.get("value"))}{ci(x.get("ci"))}: '
              f'{x.get("reason", "")}' for r, cid, x in fl]
        L.append('')
    rl = sorted({rules_label(r) for r in reps})
    L.append(f'{", ".join(rl).capitalize()}. Reading: {READING}')
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------------------------------------ dataset card
def section(reps, date=None):
    rep = reps[0]
    if len(reps) == 1:
        v = rep['verdict']
        rows = ['| check | id | result | value [95% CI] | warn / fail |', '|---|---|---|---|---|']
        for cid, r in rep['checks'].items():
            if r['status'] in ('SKIP', 'INFO'):
                continue
            rows.append(f'| {md_cell(r.get("title", cid))} | `{cid}` | {r["status"]} | '
                        f'{num(r.get("value"))}{ci(r.get("ci"))} | {thr(r.get("threshold"))} |')
        sk = [(cid, r) for cid, r in rep['checks'].items() if r['status'] == 'SKIP']
        if sk:
            rows += ['', 'Skipped (SKIP): ' + '; '.join(f'`{c}` ({str(r.get("reason", "")).rstrip(".")})' for c, r in sk) + '.']
        lb = rep.get('lookup_baseline') or {}
        br = rep.get('best_rule')
        base = []
        if lb.get('value') is not None:
            base.append(f'- Nearest-training lookup (label of the most similar training sequence): {metric(lb.get("metric"))} '
                        f'{num(lb["value"])}{ci(lb.get("ci"))}; chance {num(lb.get("chance"))}.')
        if br and br.get('family') == 'lookup':
            base.append('- Best no-model rule: the nearest-training lookup above. It is chosen on the test split, so it is '
                        'an upper bound.')
        elif br:
            base.append(f'- Best no-model rule ({rule_text(rep, br.get("family", ""))}): {metric(br.get("metric"))} '
                        f'{num(br.get("value"))}{ci(br.get("ci"))}. It is chosen on the test split, so it is an upper bound.')
        verdict, summary, cmd = v['status'], cap(v.get('summary', '')), portable_command(rep)
    else:
        tot = {s: sum(1 for r in reps if r['verdict']['status'] == s) for s in ('FAIL', 'WARN', 'PASS')}
        verdict = 'FAIL' if tot['FAIL'] else ('WARN' if tot['WARN'] else 'PASS')
        summary = f'{len(reps)} tasks: {tot["FAIL"]} FAIL, {tot["WARN"]} WARN, {tot["PASS"]} PASS'
        rows = ['| task | verdict | failing checks | nearest-training lookup (chance) | best no-model rule |',
                '|---|---|---|---|---|']
        for r in reps:
            lb = r.get('lookup_baseline') or {}
            br = r.get('best_rule') or {}
            rows.append(f'| {md_cell(r["task"].get("name"))} | {r["verdict"]["status"]} | {", ".join(r["verdict"]["fail"]) or "-"} | '
                        f'{num(lb.get("value")) or "-"} ({num(lb.get("chance")) or "-"}) | {num(br.get("value")) or "-"} |')
        base = ['Lookup: label of the most similar training sequence. Best no-model rule: chosen on the test split, an '
                'upper bound for rules without a model. Metric: ' + ', '.join(sorted({metric((r.get('lookup_baseline') or {}).get('metric')) for r in reps})) + '.']
        cmd = '\n'.join(portable_command(r) for r in reps[:3]) + ('\n...' if len(reps) > 3 else '')
    vals = dict(rules=', '.join(sorted({rules_label(r) for r in reps})),
                tool=', '.join(sorted({(r.get('tool') or {}).get('name') or 'seqleak' for r in reps})),
                version=', '.join(sorted({tool_label(r) for r in reps})),
                date_line=(f'Audited on {date}.\n' if date else ''), verdict=verdict, summary=summary,
                table='\n'.join(rows), baselines='\n'.join(base), reading=READING, limits=limits(reps), command=cmd)
    return string.Template(SECTION_TEMPLATE).substitute(vals)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(prog='seqleak_card', description=__doc__.split('\n\n')[0])
    ap.add_argument('reports', nargs='*', help='report.json files of seqleak audit')
    ap.add_argument('--html', help='write the HTML card here')
    ap.add_argument('--md', help='write the Markdown card here')
    ap.add_argument('--section', help='write a "Leakage audit" dataset-card section here')
    ap.add_argument('--title', help='card title (default: the task name, or "N audited tasks")')
    ap.add_argument('--date', help='audit date printed in the dataset-card section (e.g. 2026-10-22)')
    ap.add_argument('--section-template', action='store_true', help='print the blank section template and exit')
    ap.add_argument('--fail-on', default='never', choices=['never', 'warn', 'fail'],
                    help='exit with status 1 if any report has this verdict or worse')
    a = ap.parse_args(argv)
    if a.section_template:
        sys.stdout.write(string.Template(SECTION_TEMPLATE).safe_substitute(reading=READING, limits=limits()))
        return 0
    if not a.reports:
        ap.error('give at least one report.json')
    reps = [load(p) for p in a.reports]
    if not (a.html or a.md or a.section):
        sys.stdout.write(md_single(reps[0], a.title) if len(reps) == 1 else md_multi(reps, a.title))
    for path, text in ((a.html, lambda: html_single(reps[0], a.title) if len(reps) == 1 else html_multi(reps, a.title)),
                       (a.md, lambda: md_single(reps[0], a.title) if len(reps) == 1 else md_multi(reps, a.title)),
                       (a.section, lambda: section(reps, a.date))):
        if path:
            d = os.path.dirname(os.path.abspath(path))
            os.makedirs(d, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text())
    lvl = dict(never=99, warn=1, fail=2)[a.fail_on]
    rank = max(dict(PASS=0, WARN=1, FAIL=2).get(r['verdict']['status'], 0) for r in reps)
    return 1 if rank >= lvl else 0


if __name__ == '__main__':
    sys.exit(main())
