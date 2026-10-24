"""Writing the audit report: JSON (schema seqleak.audit/1) and a Markdown summary."""
import json
import math

import numpy as np


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(rep, path):
    with open(path, 'w') as f:
        json.dump(_clean(rep), f, indent=1)


def _f(v, d=3):
    if v is None:
        return ''
    if isinstance(v, float):
        return f'{v:.{d}f}'
    return str(v)


def _ci(ci):
    if not ci or ci[0] is None:
        return ''
    return f' [{ci[0]:.3f}, {ci[1]:.3f}]'


def to_markdown(rep):
    t = rep['task']
    L = []
    w = L.append
    w(f"# seqleak audit: {t['name']}")
    w('')
    v = rep['verdict']
    w(f"**Verdict: {v['status']}**. {v['summary']}.")
    w('')
    n = t['n']
    w(f"Items: train {n['train']}, dev {n['dev']}, test {n['test']}; {t['task_type']}, classes {t['classes']}; "
      f"length {t['length']['min']}-{t['length']['max']} (median {t['length']['median']:.0f}); elements "
      f"{', '.join(t['elements'])}.")
    if t.get('single_class_train'):
        w('The training split has one class: rules are cross-validated within the test items; the overlap checks use '
          'the training split.')
    if t.get('mode') == 'test_only':
        w('Test-only task (no training split): overlap checks and the lookup are skipped; rules are cross-validated '
          'within the test items; twin and construction checks use all items given.')
    lb = rep.get('lookup_baseline') or {}
    if lb.get('value') is not None:
        w(f"Nearest-training lookup: {lb['metric']} {lb['value']:.3f}{_ci(lb.get('ci'))} (chance {_f(lb.get('chance'))}).")
    br = rep.get('best_rule')
    if br:
        s = f"Best no-model rule on test ({br['family']}): {br['metric']} {br['value']:.3f}{_ci(br.get('ci'))}"
        if br.get('published_best') is not None:
            s += f"; published best {br['published_best']:.3f}, ratio of above-chance scores {_f(br.get('ratio'))}"
        w(s + ' (selected on test: an upper bound for no-model rules).')
    w('')
    w('| check | status | statistic | value [95% CI] | warn / fail |')
    w('|---|---|---|---|---|')
    for cid, r in rep['checks'].items():
        thr = r.get('threshold') or {}
        tt = f"{_f(thr.get('warn'))} / {_f(thr.get('fail')) or '-'}" if thr else ''
        val = (_f(r['value']) + _ci(r.get('ci'))) if r.get('value') is not None else ''
        w(f"| {cid} | {r['status']} | {r.get('statistic') or r.get('reason', '')} | {val} | {tt} |")
    w('')
    for cid, r in rep['checks'].items():
        if r['status'] in ('FAIL', 'WARN', 'INFO'):
            w(f"## {cid} ({r['status']}): {r['title']}")
            w(r.get('reason', ''))
            ev = r.get('evidence') or {}
            if ev:
                w('')
                w('Evidence: `' + json.dumps(_clean(ev), sort_keys=True)[:1500] + '`')
            if r.get('examples'):
                w('')
                w('Examples:')
                for e in r['examples']:
                    w('- ' + ', '.join(f'{k}={v}' for k, v in e.items()))
            w('')
    rt = rep.get('runtime', {})
    w(f"Runtime {rt.get('seconds')} s, peak memory {rt.get('peak_rss_mb')} MB. seqleak {rep['tool'].get('version')}, "
      f"schema {rep['schema']}.")
    return '\n'.join(L) + '\n'
