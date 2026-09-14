"""Independent provenance, schedule, task and recurrence audit for LN-072."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.provenance import atomic_json, file_digest
from scripts.localize_persistent_learning import independent_check
from scripts.audit_sharded_rewrite_cell import audit_trace


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, weights_only=True, map_location='cpu')


def bank(state):
    return state['bank_values'][state['bank_indices']].reshape(state['bank_shape'])


def audit_origin(root):
    for name, h in read(root / 'source-manifest.json').items():
        assert file_digest(root / 'source' / name) == h, name
    parents = read(root / 'parents.json')
    assert len(parents) == 1
    path, h = next(iter(parents.items()))
    assert file_digest(path) == file_digest(root / 'merged-start-parent.pt') == h
    parent = load(root / 'merged-start-parent.pt')
    assert parent['out']['admitted'][:, 1].all()
    state = parent['post_event_state']
    physical = bank(state)
    assert physical.shape == (4, 8, 40259)
    assert all(torch.equal(physical[i, j], physical[0, 0]) for i in range(4) for j in range(8))
    payload = physical[0, [0, 4]].flatten()
    hidden = state['hidden_bank'][:, [0, 4]].reshape(4, 128)
    assert payload.numel() == 80518 and payload[-1] != 0
    origin = load(root / 'repair-origin.pt')
    assert torch.equal(origin['payload'], payload) and torch.equal(origin['hidden'], hidden)
    return state, payload


def audit_data(root, dataset, evaluation):
    config = read(root / 'configuration.json')
    with gzip.open(dataset / 'pool.json.gz', 'rt') as stream:
        pool = json.load(stream)
    assert len(pool) == 4 * config['pool_per_length']
    counts = Counter()
    groups = {}
    for i, row in enumerate(pool):
        independent_check(row)
        assert row['split'] == 'train' and row['context'] == 'ungated'
        length = row['active_length']
        assert length in (2, 4, 8, 12)
        assert row['values'][length:] == [0] * (12 - length) and row['query'] < length
        counts[length, row['family'], row['layout']] += 1
        groups.setdefault((length, row['family']), []).append(i)
    assert len(counts) == 24 and set(counts.values()) == {config['pool_per_length'] // 6}
    schedule = load(dataset / 'schedule.pt')
    assert schedule.shape == (config['steps'], 32, 4) and schedule.dtype == torch.int32
    rng = random.Random(read(dataset / 'configuration.json')['schedule_seed'])
    for ordinal, indices in enumerate(schedule.tolist()):
        stage = sum(ordinal >= b for b in config['curriculum_boundaries'])
        length = (2, 4, 8, 12)[stage]
        for j in range(4):
            actual = [row[j] for row in indices]
            assert all(pool[i]['active_length'] == length for i in actual)
            assert Counter(pool[i]['family'] for i in actual) == {'lookup': 16, 'parity': 8, 'sum3': 8}
            expected = [rng.choice(groups[length, f]) for f, n in (('lookup', 16), ('parity', 8), ('sum3', 8)) for _ in range(n)]
            rng.shuffle(expected)
            assert actual == expected
    rows = read(evaluation)
    assert len(rows) == 4 and len({len(s) for s in rows}) == 1
    by_cell = {}
    for stream in rows:
        for row in stream:
            independent_check(row)
            assert row['split'] == 'validation' and row['context'] == 'ungated'
            by_cell.setdefault((row['family'], row['layout']), []).append(row)
    assert len(by_cell) == 6
    for cell_rows in by_cell.values():
        assert len(cell_rows) == len({r['core_sha256'] for r in cell_rows}) == config['evaluation_per_cell']
    assert not {r['core_sha256'] for r in pool} & {r['core_sha256'] for s in rows for r in s}
    probe = read(dataset / 'train-probe.json')
    expected = [r for r in pool if r['active_length'] == 12][:config['train_probe_size']]
    assert probe == [expected[i::4] for i in range(4)]
    return {'validation': rows, 'train_probe': probe}


def audit_case(root, dest):
    root, dest = Path(root), Path(dest)
    config, case = read(root / 'configuration.json'), read(dest / 'configuration.json')
    state, initial_payload = audit_origin(root)
    panels = audit_data(root, root / case['dataset'], root / case['evaluation'])
    assert case['storage_rule'] == config['arms'][case['arm']]
    logs = [json.loads(line) for line in (dest / 'training.jsonl').read_text().splitlines()]
    assert len(logs) == config['steps']
    for ordinal, row in enumerate(logs):
        assert row['update'] == ordinal + 1
        stage = sum(ordinal >= b for b in config['curriculum_boundaries'])
        assert row['active_length'] == (2, 4, 8, 12)[stage]
        assert row['learning_rate'] == config['learning_rates'][int(ordinal >= config['lr_boundary'])]
        assert math.isfinite(row['loss_before_update']) and math.isfinite(row['gradient_norm_before_clip'])
        assert 0 <= row['window_task_correct'] <= 128
        assert len(row['selected_admissions_by_position']) == 4 and all(0 <= n <= 16 for n in row['selected_admissions_by_position'])
        components = [row[k] for k in ('task_ce_by_position', 'selected_bce_by_position', 'other_bce_by_position')]
        assert all(len(v) == 4 and all(math.isfinite(x) for x in v) for v in components)
        total = sum(components[0]) / 4 + sum(components[1]) / 8 + sum(components[2]) / 8
        assert abs(total - row['loss_before_update']) < 2e-5
    for step in config['checkpoint_updates']:
        checkpoint = load(dest / f'update-{step:05d}.pt')
        assert checkpoint['step'] == step and checkpoint['payload'].shape == initial_payload.shape
        assert torch.isfinite(checkpoint['payload']).all()
        opt = checkpoint['optimizer']
        group = opt['param_groups'][0]
        assert len(opt['param_groups']) == 1 and group['weight_decay'] == 0.
        assert group['lr'] == config['learning_rates'][int(max(0, step - 1) >= config['lr_boundary'])]
        if step == 0:
            assert torch.equal(checkpoint['payload'], initial_payload) and not opt['state']
        else:
            assert len(opt['state']) == 1
            moment = next(iter(opt['state'].values()))
            assert int(moment['step']) == step
            assert moment['exp_avg'].shape == moment['exp_avg_sq'].shape == initial_payload.shape
            assert torch.isfinite(moment['exp_avg']).all() and torch.isfinite(moment['exp_avg_sq']).all()
    final_payload = load(dest / f"update-{config['steps']:05d}.pt")['payload']
    summary = read(dest / 'summary.json')
    assert summary['status'] == 'complete' and len(summary['results']) == 6
    expected_names = {f'{e}-{p}-{v}' for e in ('initial', 'final')
                      for p, v in (('fp32', 'validation'), ('fp64', 'validation'), ('fp32', 'train_probe'))}
    assert {r['name'] for r in summary['results']} == expected_names
    details, ticks, maximum_error, predictions = [], 0, 0., 0
    for record in summary['results']:
        saved = load(dest / (record['name'] + '.pt'))
        out, before, after = saved['out'], saved['initial_state'], saved['final_state']
        assert out['logits'].dtype == (torch.float32 if record['precision'] == 'fp32' else torch.float64)
        for runtime in (before, after):
            assert runtime['bank_shape'] == [4, 8, 40259] and list(runtime['hidden_bank'].shape) == [4, 8, 64]
            assert runtime['gain'] == runtime['normalizer_sign'] == runtime['code_sign'] == runtime['writer_sign'] == 1.
            assert runtime['execution'] == 'commit' and runtime['storage_rule'] == case['storage_rule']
        expected = initial_payload if record['endpoint'] == 'initial' else final_payload
        physical = bank(before)
        assert torch.equal(physical[:, [0, 4]].reshape(4, -1), expected.to(physical.dtype).expand(4, -1))
        assert torch.equal(before['hidden_bank'], state['hidden_bank'].to(physical.dtype))
        rows = panels[record['panel']]
        ids = torch.tensor([[r['tokens'] for r in s] for s in rows])
        labels = torch.tensor([[r['label'] for r in s] for s in rows])
        prediction = out['logits'].argmax(-1)
        target = torch.tensor([[r['family'] == 'lookup' for r in s] for s in rows])
        truth = torch.tensor([True, False, False, True]).expand_as(out['admitted']).clone()
        truth[:, :, 1] = target
        assert torch.equal(out['admitted'], out['policy_logits'] > 0)
        assert torch.equal(out['emitted'], torch.where(out['admitted'], prediction[:, :, None], 3))
        accuracy = float((prediction == labels).double().mean())
        correct = int((out['emitted'][:, :, 1][target] == labels[target]).sum())
        errors = int((out['admitted'] != truth).sum())
        scores = record['scores']
        assert scores['task_accuracy'] == accuracy and scores['target_correct_answers'] == correct
        assert scores['target_n'] == int(target.sum()) and record['exception_rule_errors'] == errors
        gate = len(scores['cells']) == 6
        for key, metrics in scores['cells'].items():
            family, layout = key.split('/')
            chosen = [(i, j, r) for i, s in enumerate(rows) for j, r in enumerate(s)
                      if r['family'] == family and r['layout'] == layout]
            n = len(chosen)
            acc = sum(int(prediction[i, j]) == r['label'] for i, j, r in chosen) / n
            z = 1.959963984540054
            lower = (acc + z*z/(2*n) - z*math.sqrt(acc*(1-acc)/n + z*z/(4*n*n))) / (1+z*z/n)
            late = [(i, j, r) for i, j, r in chosen if j >= len(rows[0]) // 2]
            late_acc = sum(int(prediction[i, j]) == r['label'] for i, j, r in late) / len(late) if late else 0.
            qualifies = n >= 128 and acc >= .95 and lower >= .9 and late_acc >= .95
            assert metrics['n'] == n and metrics['accuracy'] == acc and abs(metrics['wilson_lower'] - lower) < 1e-12
            assert metrics['late_accuracy'] == late_acc and metrics['qualified'] == qualifies
            gate &= qualifies
        recovery = gate and correct / int(target.sum()) >= .95 and errors == 0
        assert record['task_qualified'] == gate and record['recovery_qualified'] == recovery
        assert record['eligible_repair_escape'] == (record['panel'] == 'validation' and case['arm'] == 'learned-binding' and recovery)
        count, error = audit_trace(saved, before, out, ids)
        ticks += count
        maximum_error = max(maximum_error, error)
        assert record['correspondence']['passed']
        tolerance = 1e-4 if record['precision'] == 'fp32' else 1e-10
        assert max(record['correspondence']['task_logit_max'], record['correspondence']['policy_logit_max']) <= tolerance
        assert record['correspondence']['task_decision_disagreements'] == record['correspondence']['admission_disagreements'] == 0
        predictions += labels.numel()
        details.append({'name': record['name'], 'task_accuracy': accuracy, 'correct_forbidden_answers': correct,
                        'forbidden_n': int(target.sum()), 'exception_rule_errors': errors, 'recovery_qualified': recovery})
    for endpoint in ('initial', 'final'):
        a = load(dest / f'{endpoint}-fp32-validation.pt')['out']
        b = load(dest / f'{endpoint}-fp64-validation.pt')['out']
        assert torch.equal(a['logits'].argmax(-1), b['logits'].argmax(-1))
        assert torch.equal(a['admitted'], b['admitted'])
    assert summary['precision_agreement']['passed']
    return {'passed': True, 'conditions': details, 'task_predictions_rescored': predictions,
            'emissions_rescored': 4 * predictions, 'independent_tick_replays': ticks,
            'maximum_candidate_replay_error': maximum_error, 'optimizer_and_curriculum_verified': True,
            'actual_damaged_start_and_padding_verified': True}


def audit_batch(root):
    root = Path(root)
    config, summary = read(root / 'configuration.json'), read(root / 'summary.json')
    audit_origin(root)
    assert read(root / 'calibration-data/configuration.json') == dict(zip(('pool_seed', 'schedule_seed'), config['calibration_seeds']))
    cases = summary['completed_cases']
    calibration = read(root / 'calibration/summary.json')
    final = [r for r in calibration['results'] if r['endpoint'] == 'final' and r['panel'] == 'validation']
    qualified = len(final) == 2 and all(r['recovery_qualified'] for r in final)
    assert summary['calibration_qualified'] == qualified
    assert read(root / 'calibration/audit.json')['passed']
    if config['fixture']:
        assert cases == ['calibration', 'fixture-learned'] and not summary['matched_comparison_started']
    elif qualified:
        assert summary['matched_comparison_started'] and summary['status'] == 'matched-complete'
        assert cases == ['calibration'] + [f'pair-{i}-{arm}' for i in (1, 2, 3) for arm in config['arms']]
        dev, heldout = read(root / 'calibration-evaluation.json'), read(root / 'comparison-evaluation.json')
        assert not {r['core_sha256'] for s in dev for r in s} & {r['core_sha256'] for s in heldout for r in s}
        for i, seeds in enumerate(config['paired_seeds'], 1):
            dataset = f'pair-{i}-data'
            assert read(root / dataset / 'configuration.json') == dict(zip(('pool_seed', 'schedule_seed'), seeds))
            for arm in config['arms']:
                case = read(root / f'pair-{i}-{arm}/configuration.json')
                assert case['dataset'] == dataset and case['evaluation'] == 'comparison-evaluation.json'
    else:
        assert cases == ['calibration'] and summary['status'] == 'calibration-did-not-qualify'
        assert not summary['matched_comparison_started'] and not (root / 'comparison-evaluation.json').exists()
        assert not list(root.glob('pair-*'))
    assert all(read(root / case / 'audit.json')['passed'] for case in cases)
    return {'passed': True, 'cases': cases, 'conditional_dispatch_verified': True,
            'shared_parent_training_replications_only': True}


def audit(root):
    root = Path(root)
    for name, h in read(root / 'artifact-manifest.json').items():
        assert file_digest(root / name) == h, name
    summary = read(root / 'summary.json')
    results = {name: audit_case(root, root / name) for name in summary['completed_cases']}
    return {'passed': True, 'batch': audit_batch(root), 'cases': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
