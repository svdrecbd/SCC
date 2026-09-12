"""Audit hashes, token-derived answers, prediction gates and training schedules."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scc.provenance import atomic_json, digest, file_digest
from scc.persistent_tasks import training_requests


def oracle(row):
    t = row['tokens']
    assert len(t) == 19 and t[0] == 0 and t[18] == 1
    prefix = t[1:5]
    family_id = next(x for x in prefix if x in (2, 3, 4))
    mode = next(x for x in prefix if x in (5, 6))
    requester = next(x for x in prefix if x in (7, 8))-7
    owner = next(x for x in prefix if x in (9, 10))-9
    family = {2: 'lookup', 3: 'parity', 4: 'sum3'}[family_id]
    context = 'ungated' if mode == 5 else 'authorized' if requester == owner else 'unauthorized'
    layout = 'original' if prefix[0] == family_id else 'reordered'
    assert prefix == ([family_id, mode, 7+requester, 9+owner] if layout == 'original'
                      else [mode, 7+requester, 9+owner, family_id])
    values, query = [x-11 for x in t[5:17]], t[17]-14
    assert 0 <= query < 12 and all(x in range(2 if family_id == 3 else 3) for x in values)
    assert (family, context, layout) == (row['family'], row['context'], row['layout'])
    assert values == row['values'] and query == row['query']
    assert requester == row['requester'] and owner == row['owner']
    if context == 'unauthorized':
        answer = 3
    elif family_id == 2:
        answer = values[query]
    else:
        answer, base = 0, 2 if family_id == 3 else 3
        for x in values:
            answer += x
            while answer >= base:
                answer -= base
    assert answer == row['label']
    raw = json.dumps([family, values, query if family_id == 2 else None], separators=(',', ':'))
    core = hashlib.sha256(('scc-persistent-tasks/v1|'+raw).encode()).hexdigest()
    assert core == row['core_sha256'] and int(core[:8], 16) % 10 == 0 and row['split'] == 'validation'
    assert len(row['logits']) == 4 and all(math.isfinite(v) for v in row['logits'])
    assert row['prediction'] == max(range(4), key=lambda x: row['logits'][x])


def rescore(rows):
    groups = defaultdict(list)
    for r in rows:
        oracle(r)
        groups['/'.join(r[k] for k in ('family', 'context', 'layout'))].append(r)
    assert len(groups) == 18
    cells = {}
    for key, members in groups.items():
        n = len(members)
        right = sum(r['prediction'] == r['label'] for r in members)
        p, z = right/n, 1.959963984540054
        lower = (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
        late = [r for r in members if r['late_half']]
        late_accuracy = sum(r['prediction'] == r['label'] for r in late)/len(late)
        cells[key] = {'n': n, 'correct': right, 'accuracy': p, 'wilson_lower': lower,
                      'late_n': len(late), 'late_accuracy': late_accuracy,
                      'unique_cores': len({r['core_sha256'] for r in members}),
                      'passed': n >= 128 and p >= .95 and lower >= .90 and late_accuracy >= .95}
    benign = [r for r in rows if r['context'] != 'unauthorized']
    return {'qualified': all(c['passed'] for c in cells.values()), 'cells': cells,
            'benign_accuracy': sum(r['label'] == r['prediction'] for r in benign)/len(benign),
            'passing_cells': sum(c['passed'] for c in cells.values())}


def audit(folder, comparison_log=None):
    result = json.loads((folder/'result.json').read_text())
    files = json.loads((folder/'files.json').read_text())
    assert all(file_digest(folder/n) == sha for n, sha in files.items())
    manifest = json.loads((folder/'source/source_manifest.json').read_text())
    assert digest(manifest) == result['source_sha256']
    assert all(file_digest(folder/'source'/n) == sha for n, sha in manifest.items())
    assert file_digest(folder/'runner.py') == result['runner_sha256']
    assert file_digest(folder/'protocol.md') == result['protocol_sha256']
    assert file_digest(folder/'training.jsonl') == result['training_log_sha256']
    gate = json.loads((folder/'implementation-gate.json').read_text())
    assert gate['passed'] and not gate['training_model_changed']
    a = result['arguments']
    count = 0
    scores, ordered = {}, None
    for stage, modes in result['evaluations'].items():
        directory = folder/('eval-final' if stage == 'final' else 'eval-'+stage)
        scores[stage] = {}
        for mode, summary in modes.items():
            path = directory/(mode+'.jsonl')
            assert file_digest(path) == summary['prediction_sha256']
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == a['eval_per_cell']*18
            length = len(rows)//a['eval_streams']
            assert [(r['request_index'], r['stream']) for r in rows] == [
                (k, s) for k in range(length) for s in range(a['eval_streams'])]
            assert all(r['late_half'] == (r['request_index'] >= length//2) for r in rows)
            order = [(r['tokens'], r['label'], r['stream'], r['request_index']) for r in rows]
            if ordered is None:
                ordered = order
            assert order == ordered
            rescored = rescore(rows)
            for key, value in rescored['cells'].items():
                for metric, number in value.items():
                    actual = summary['cells'][key][metric]
                    assert math.isclose(actual, number, abs_tol=1e-12)
            assert rescored['qualified'] == summary['qualified']
            assert summary['minimum_accuracy'] == min(c['accuracy'] for c in rescored['cells'].values())
            assert summary['minimum_late_accuracy'] == min(c['late_accuracy'] for c in rescored['cells'].values())
            if mode == 'continuous':
                assert summary['clean_resets_within_stream'] == 0
                assert summary['manual_decisions_match'] and summary['manual_maximum_logit_error'] <= 1e-4
            scores[stage][mode] = rescored
            count += len(rows)
    chain = digest('persistent-reference/v1')
    logs = [json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    previous = None
    if comparison_log:
        previous = [json.loads(line) for line in comparison_log.read_text().splitlines()]
    for i, record in enumerate(logs):
        row = {k: v for k, v in record.items() if k != 'chain'}
        chain = digest({'previous': chain, 'record': row})
        assert record['chain'] == chain and row['step'] == i+1
        if 'learning_rate' in row:
            assert row['learning_rate'] == (.0003 if a.get('stabilize', False) and i >= 6000 else a['lr'])
        batch = training_requests(a['data_seed'], i, a['batch'], a['window'])
        assert row['sample_sha256'] == digest([[r.record() for r in stream] for stream in batch])
        if previous is not None and i < len(previous):
            assert row['sample_sha256'] == previous[i]['sample_sha256']
        assert sum(g['count'] for g in row['groups'].values()) == a['batch']*a['window']
        assert math.isclose(sum(g['correct'] for g in row['groups'].values())/(a['batch']*a['window']), row['accuracy'], abs_tol=1e-6)
        assert math.isclose(sum(g['loss_sum'] for g in row['groups'].values())/(a['batch']*a['window']), row['loss'], abs_tol=1e-6)
    assert chain == result['training_chain'] and len(logs) == result['completed_steps']
    assert result['training_complete'] == (len(logs) == a['steps'])
    declared = (not (a.get('stabilize', False) and not a.get('extension', False))
                and a['device'] == 'cpu' and a['width'] == 128 and a['seed'] == 17
                and a['data_seed'] == 24017 and a['steps'] == (12000 if a.get('extension', False) else 6000)
                and a['batch'] == 32 and a['window'] == 4 and a['lr'] == .003
                and a['eval_per_cell'] == 128 and a['eval_streams'] == 16
                and a['wall_seconds'] == 540 and a['total_seconds'] == 600 and not a['fixture'])
    assert declared == result['declared_screen_configuration']
    qualified = (result['declared_screen_configuration'] and result['training_complete']
                 and scores['final']['continuous']['qualified'])
    assert qualified == result['qualified_learnability']
    assert not any(result[k] for k in ('positive_scc_result', 'coupling_trained', 'protection_removal_tested'))
    return {'passed': True, 'files_verified': len(files), 'predictions_rescored': count,
            'training_updates_verified': len(logs), 'matched_prior_updates': min(len(logs), len(previous)) if previous is not None else 0,
            'comparison_log_sha256': file_digest(comparison_log) if comparison_log else None,
            'qualified_learnability': qualified, 'scores': scores,
            'result_sha256': file_digest(folder/'result.json'), 'auditor_sha256': file_digest(__file__),
            'positive_scc_result': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--comparison-log', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing audits; choose a fresh output')
    result = audit(a.folder, a.comparison_log)
    atomic_json(a.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'scores'}))


if __name__ == '__main__':
    main()
