"""Rescore localization evidence, verify training chains and preserve failed gates."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.localize_persistent_learning import scores, independent_check
from scc.provenance import atomic_json, digest, file_digest


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit_predictions(path, summary, expected=None):
    rows = read_rows(path)
    assert len(rows) == summary['predictions'] and file_digest(path) == summary['sha256']
    assert digest(scores(rows)) == digest(summary['cells'])
    if expected is not None:
        assert [{k: v for k, v in r.items() if k not in ('logits', 'prediction')} for r in rows] == expected
    return rows


def audit_persistence(folder, result):
    assert json.loads((folder/'result.json').read_text()) == result
    assert set(result) == {'continuous', 'batch1', 'float64', 'fresh'}
    reference, metadata, count = None, None, 0
    for mode in ('continuous', 'batch1', 'float64', 'fresh'):
        summary = result[mode]
        rows = audit_predictions(folder/(mode+'.jsonl'), summary, metadata)
        if metadata is None:
            metadata = [{k: v for k, v in r.items() if k not in ('logits', 'prediction')} for r in rows]
            reference = rows
        assert [(r['stream'], r['request_index']) for r in rows] == [(s, i) for s in range(2) for i in range(144)]
        assert sum(r['prediction'] == r['label'] for r in rows)/len(rows) == summary['accuracy']
        late = [r for r in rows if r['request_index'] >= 72]
        assert sum(r['prediction'] == r['label'] for r in late)/len(late) == summary['late_half_accuracy']
        if mode in ('batch1', 'float64'):
            maximum = max(abs(a-b) for r, q in zip(rows, reference) for a, b in zip(r['logits'], q['logits']))
            mismatches = sum(r['prediction'] != q['prediction'] for r, q in zip(rows, reference))
            assert math.isclose(maximum, summary['maximum_logit_error'], abs_tol=1e-7, rel_tol=1e-6)
            assert mismatches == summary['decision_mismatches']
            assert summary['numerical_gate_passed'] == (summary['maximum_logit_error'] <= 1e-4 and mismatches == 0)
        count += len(rows)
    return count


def audit(folder):
    result = json.loads((folder/'result.json').read_text())
    assert result['schema'] == 'persistent-localization/v1' and result['status'] == 'complete'
    source = json.loads((folder/'source/source_manifest.json').read_text())
    assert all(file_digest(folder/'source'/n) == h for n, h in source.items())
    assert file_digest(folder/'runner.py') == result['runner_sha256']
    parents = json.loads((folder/'parents.json').read_text())
    assert all(file_digest(n) == h for n, h in parents.items()) and result['parents_unchanged']
    data = json.loads((folder/'data.json').read_text())
    assert file_digest(folder/'data.json') == result['data_sha256']
    for row in data['validation'] + [r for rows in data['fit'].values() for r in rows]:
        independent_check(row)
    assert not ({r['core_sha256'] for r in data['validation']} &
                {r['core_sha256'] for rows in data['fit'].values() for r in rows})
    assert json.loads((folder/'implementation-gate.json').read_text())['passed']
    count, updates = 0, 0
    expected_states = 3 if result['fixture'] else 18
    ordinary = {k: v for k, v in result['saved_states'].items() if not k.endswith('-persistence')}
    assert len(ordinary) == expected_states
    for name, summary in ordinary.items():
        count += len(audit_predictions(folder/(name+'.jsonl'), summary, data['validation']))
    for name, summary in result['saved_states'].items():
        if name.endswith('-persistence'):
            count += audit_persistence(folder/name, summary)
    assert len(result['fits']) == (4 if result['fixture'] else 12)
    failed, incomplete, passed = [], [], []
    for name, summary in result['fits'].items():
        path = folder/name
        if summary['status'] == 'condition_failed':
            assert (path/'failure.json').is_file() and not summary['fit_passed']
            failed.append(name)
            continue
        family = name.split('-')[-1]
        chain = digest('persistent-localization-fit/v1')
        training = read_rows(path/'training.jsonl')
        for i, row in enumerate(training):
            chain = digest({'previous': chain, 'record': {k: v for k, v in row.items() if k != 'chain'}})
            assert row['chain'] == chain and row['step'] == i+1
            assert row['lr'] == (.003 if i < 1000 else .0003)
            assert row['batch_sha256'] == digest(data['fit'][family])
            assert all(math.isfinite(v) for v in [row['loss'], row['gradient_norm_before_clip'], *row['gradient_blocks'].values()])
        assert len(training) == summary['completed_steps'] and chain == summary['chain']
        saved = torch.load(path/'fitted.pt', weights_only=True, map_location='cpu')
        assert saved['steps'] == len(training)
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        assert summary['requested_steps'] == (2 if result['fixture'] else 1500)
        complete = len(training) == summary['requested_steps']
        assert summary['status'] == ('complete' if complete else 'wall_limit')
        fit_rows = audit_predictions(path/'fit.jsonl', summary['fit'], data['fit'][family])
        accuracy = sum(r['prediction'] == r['label'] for r in fit_rows)/len(fit_rows)
        loss = torch.nn.functional.cross_entropy(torch.tensor([r['logits'] for r in fit_rows]),
                                                 torch.tensor([r['label'] for r in fit_rows])).item()
        assert math.isclose(accuracy, summary['fit']['accuracy'], abs_tol=5e-8)
        assert math.isclose(loss, summary['fit']['loss'], abs_tol=1e-7)
        assert summary['fit_passed'] == (complete and accuracy == 1. and loss <= .05)
        if summary['fit_passed']: passed.append(name)
        if not complete: incomplete.append(name)
        assert not summary['qualified_learnability'] and not summary['positive_scc_result']
        validation = [r for r in data['validation'] if r['family'] == family]
        count += len(fit_rows)+len(audit_predictions(path/'validation.jsonl', summary['transfer'], validation))
        count += audit_persistence(path/'persistence', summary['persistence'])
        updates += len(training)
    assert not result['qualified_learnability'] and not result['positive_scc_result']
    return {'passed': True, 'scope': 'Artifact integrity and independent token-oracle rescoring, not scientific qualification',
        'predictions_rescored': count, 'training_updates_verified': updates, 'source_files_verified': len(source),
        'parent_files_unchanged': len(parents), 'fit_passed_conditions': passed,
        'failed_conditions': failed, 'incomplete_conditions': incomplete, 'positive_scc_result': False,
        'fixture': result['fixture'], 'result_sha256': file_digest(folder/'result.json'),
        'auditor_sha256': file_digest(__file__)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Use a fresh audit path')
    torch.set_num_threads(2)
    result = audit(a.folder); atomic_json(a.output, result)
    print(json.dumps(result))


if __name__ == '__main__': main()
