"""LN-072: bounded recovery calibration followed by a gate-controlled paired batch."""
import argparse
import gzip
import json
from pathlib import Path
import platform
import random
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from scc.persistent_tasks import sample_request
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.sharded_repair import start_values, restore, functional_window
from scripts.run_sharded_repair import PARENT, compare_functional
from scripts.run_learned_binding_bank import tensor_ids, score
from scripts.run_sharded_rewrite_cell import run_stream, pack_state
from scripts.localize_persistent_learning import independent_check

CELLS = [(f, 'ungated', l) for f in ('lookup', 'parity', 'sum3')
         for l in ('original', 'reordered')]
ARMS = {'learned-binding': 'learned', 'symbolic-binding-control': 'symbolic'}


def configuration(fixture=False):
    return {
        'plan': 'LN-072', 'fixture': fixture, 'steps': 64 if fixture else 12000,
        'curriculum_boundaries': [4, 10, 20] if fixture else [400, 1000, 2000],
        'lr_boundary': 32 if fixture else 6000, 'learning_rates': [.003, .0003],
        'pool_per_length': 96 if fixture else 12288, 'batch_streams': 32,
        'requests_per_window': 4, 'tokens_per_request': 19,
        'family_counts_per_position': [16, 8, 8], 'optimizer': 'Adam',
        'weight_decay': 0., 'initialization_noise': 0., 'clip_norm': 1.,
        'checkpoint_updates': [0, 4, 10, 20, 32, 64] if fixture else [0, 400, 1000, 2000, 6000, 9000, 12000],
        'calibration_seeds': [17313020, 17313021], 'calibration_evaluation_seed': 17313022,
        'paired_seeds': [[17313023, 17313024], [17313025, 17313026], [17313027, 17313028]],
        'comparison_evaluation_seed': 17313030, 'evaluation_per_cell': 8 if fixture else 128,
        'train_probe_size': 48 if fixture else 768, 'evaluation_streams': 4,
        'physical_payload_values': 80518, 'decoded_learned_coefficients': 80517,
        'bank_shape': [4, 8, 40259], 'hidden_bank_shape': [4, 8, 64],
        'gain': 1., 'normalizer_sign': 1., 'writer_sign': 1., 'code_sign': 1.,
        'execution': 'commit', 'arms': ARMS, 'cpu_threads': 2,
        'trajectory_wall_seconds': 2400, 'batch_wall_seconds': 300 if fixture else 18000,
        'output_limit_bytes': 1024**3,
        'runtime': {'python': sys.version, 'torch': str(torch.__version__),
                    'platform': platform.platform(), 'processor': platform.processor()},
    }


def active_length(ordinal, config):
    a, b, c = config['curriculum_boundaries']
    return 2 if ordinal < a else 4 if ordinal < b else 8 if ordinal < c else 12


def learning_rate(ordinal, config):
    return config['learning_rates'][int(ordinal >= config['lr_boundary'])]


def make_training(config, pool_seed, schedule_seed):
    rng = random.Random(pool_seed)
    pool = []
    for length in (2, 4, 8, 12):
        for i in range(config['pool_per_length']):
            row = sample_request(rng, CELLS[i % 6], active_length=length).record()
            row['active_length'] = length
            independent_check(row)
            pool.append(row)
    groups = {(length, family): [i for i, r in enumerate(pool)
                                if r['active_length'] == length and r['family'] == family]
              for length in (2, 4, 8, 12) for family in ('lookup', 'parity', 'sum3')}
    rng = random.Random(schedule_seed)
    schedule = torch.empty(config['steps'], 32, 4, dtype=torch.int32)
    for ordinal in range(config['steps']):
        length = active_length(ordinal, config)
        for position in range(4):
            indices = [rng.choice(groups[length, family])
                       for family, n in zip(('lookup', 'parity', 'sum3'), (16, 8, 8))
                       for _ in range(n)]
            rng.shuffle(indices)
            schedule[ordinal, :, position] = torch.tensor(indices, dtype=torch.int32)
    probe = [r for r in pool if r['active_length'] == 12][:config['train_probe_size']]
    return pool, schedule, [probe[i::4] for i in range(4)]


def make_evaluation(seed, per_cell, excluded=()):
    rng = random.Random(seed)
    blocked = set(excluded)
    by_cell = []
    for cell in CELLS:
        selected = []
        seen = set()
        for _ in range(100000):
            row = sample_request(rng, cell, split='validation').record()
            key = row['core_sha256']
            if key in blocked or key in seen:
                continue
            independent_check(row)
            seen.add(key)
            selected.append(row)
            if len(selected) == per_cell:
                break
        if len(selected) != per_cell:
            raise ValueError('Could not populate distinct held-out cores')
        by_cell.append(selected)
    serial = []
    for ordinal in range(per_cell):
        order = list(range(6))
        rng.shuffle(order)
        serial.extend(by_cell[i][ordinal] for i in order)
    return [serial[i::4] for i in range(4)]


def write_data(folder, config, seeds):
    folder.mkdir()
    pool, schedule, probe = make_training(config, *seeds)
    with gzip.open(folder / 'pool.json.gz', 'wt') as stream:
        json.dump(pool, stream, separators=(',', ':'))
    torch.save(schedule, folder / 'schedule.pt')
    atomic_json(folder / 'train-probe.json', probe)
    atomic_json(folder / 'configuration.json', {'pool_seed': seeds[0], 'schedule_seed': seeds[1]})
    return pool, schedule, probe


def objective(out, labels, target):
    raw = out['policy_logits']
    desired = raw.new_tensor([1., 0., 0., 1.]).expand_as(raw).clone()
    desired[:, :, 1] = target.to(raw.dtype)
    task, selected, other = [], [], []
    for j in range(raw.shape[1]):
        mask = torch.zeros_like(raw[:, j], dtype=torch.bool)
        mask[:, 1] = target[:, j]
        task.append(F.cross_entropy(out['logits'][:, j], labels[:, j]))
        selected.append(F.binary_cross_entropy_with_logits(raw[:, j][mask], desired[:, j][mask]))
        other.append(F.binary_cross_entropy_with_logits(raw[:, j][~mask], desired[:, j][~mask]))
    loss = torch.stack(task).mean() + .5 * torch.stack(selected).mean() + .5 * torch.stack(other).mean()
    return loss, {'task_ce_by_position': [float(v.detach()) for v in task],
                  'selected_bce_by_position': [float(v.detach()) for v in selected],
                  'other_bce_by_position': [float(v.detach()) for v in other]}


def directory_bytes(folder):
    return sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())


def check_resources(root, config, trajectory_start=None):
    if trajectory_start is not None and time.monotonic() - trajectory_start > config['trajectory_wall_seconds']:
        raise TimeoutError('LN-072 trajectory wall limit exceeded')
    if directory_bytes(root) > config['output_limit_bytes']:
        raise RuntimeError('LN-072 output ceiling exceeded')


def optimize(initial, hidden, pool, schedule, rule, dest, root, config):
    started = time.monotonic()
    q = torch.nn.Parameter(initial.clone())
    opt = torch.optim.Adam([q], lr=learning_rate(0, config))
    ids = torch.tensor([r['tokens'] for r in pool])
    labels = torch.tensor([r['label'] for r in pool])
    lookup = torch.tensor([r['family'] == 'lookup' for r in pool])
    h = hidden.repeat(8, 1)
    m = q.numel() // 2

    def save(step):
        torch.save({'payload': q.detach().clone(), 'step': step, 'optimizer': opt.state_dict()},
                   dest / f'update-{step:05d}.pt')

    save(0)
    try:
        with (dest / 'training.jsonl').open('w') as stream:
            for ordinal, selected in enumerate(schedule):
                if time.monotonic() - started > config['trajectory_wall_seconds']:
                    raise TimeoutError('LN-072 trajectory wall limit exceeded')
                lr = learning_rate(ordinal, config)
                opt.param_groups[0]['lr'] = lr
                indices = selected.long()
                out, committed, _ = functional_window(q, h, ids[indices], 128, rule)
                target = lookup[indices]
                loss, components = objective(out, labels[indices], target)
                opt.zero_grad()
                loss.backward()
                grad = float(torch.nn.utils.clip_grad_norm_([q], 1., error_if_nonfinite=True))
                opt.step()
                row = {'update': ordinal + 1, 'active_length': active_length(ordinal, config),
                       'learning_rate': lr, 'loss_before_update': float(loss.detach()),
                       'gradient_norm_before_clip': grad, **components,
                       'window_task_correct': int((out['logits'].argmax(-1) == labels[indices]).sum()),
                       'selected_admissions_by_position': (out['admitted'][:, :, 1] & target).sum(0).tolist(),
                       'committed_payload_difference_mean': float((committed[:, 0] - committed[:, 1]).detach().norm(dim=-1).mean()),
                       'candidate_payload_difference_norm': float((q[:m] - q[m:]).detach().norm()),
                       'physical_padding_after_update': float(q[-1].detach()),
                       'seconds': time.monotonic() - started}
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                if ordinal + 1 in config['checkpoint_updates']:
                    save(ordinal + 1)
                    stream.flush()
                    check_resources(root, config, started)
    except BaseException:
        torch.save({'payload': q.detach().clone(), 'optimizer': opt.state_dict()}, dest / 'interrupted.pt')
        raise
    atomic_json(dest / 'parameter-changes.json', {
        'physical_values_exposed': q.numel(), 'physical_values_changed': int(torch.count_nonzero(q.detach() - initial)),
        'change_l2': float((q.detach() - initial).norm()), 'padding_initial': float(initial[-1]),
        'padding_final': float(q[-1].detach()), 'training_seconds': time.monotonic() - started})
    return q.detach().clone()


@torch.no_grad()
def evaluate(start, initial, final, rule, arm, panels, dest):
    results = []
    for endpoint, payload in (('initial', initial), ('final', final)):
        for precision, dtype in (('fp32', torch.float32), ('fp64', torch.float64)):
            for panel in ('validation', 'train_probe') if precision == 'fp32' else ('validation',):
                rows = panels[panel]
                ids = tensor_ids(rows)
                model = restore(start, dtype, rule, payload)
                before = pack_state(model)
                out, traces = run_stream(model, ids)
                scores = score(rows, out)
                target = torch.tensor([[r['family'] == 'lookup' for r in s] for s in rows])
                desired = torch.tensor([True, False, False, True]).expand_as(out['admitted']).clone()
                desired[:, :, 1] = target
                errors = int((out['admitted'] != desired).sum())
                task_gate = len(scores['cells']) == 6 and all(c['qualified'] for c in scores['cells'].values())
                recovery = task_gate and scores['target_correct_answers'] / scores['target_n'] >= .95 and errors == 0
                name = f'{endpoint}-{precision}-{panel}'
                correspondence = compare_functional(payload.to(dtype), start, rule, ids, out)
                torch.save({'out': out, 'traces': traces, 'initial_state': before,
                            'final_state': pack_state(model)}, dest / (name + '.pt'))
                results.append({'name': name, 'endpoint': endpoint, 'precision': precision, 'panel': panel,
                                'scores': scores, 'exception_rule_errors': errors, 'task_qualified': task_gate,
                                'recovery_qualified': recovery,
                                'eligible_repair_escape': panel == 'validation' and arm == 'learned-binding' and recovery,
                                'correspondence': correspondence, 'first_erasure': model.first_erasure.tolist()})
    return results


def precision_agreement(dest):
    details = []
    for endpoint in ('initial', 'final'):
        a = torch.load(dest / f'{endpoint}-fp32-validation.pt', weights_only=True, map_location='cpu')['out']
        b = torch.load(dest / f'{endpoint}-fp64-validation.pt', weights_only=True, map_location='cpu')['out']
        details.append({'endpoint': endpoint,
                        'task_decisions_equal': torch.equal(a['logits'].argmax(-1), b['logits'].argmax(-1)),
                        'admissions_equal': torch.equal(a['admitted'], b['admitted'])})
    return {'passed': all(d['task_decisions_equal'] and d['admissions_equal'] for d in details), 'endpoints': details}


def calibration_passes(summary, audit):
    rows = [r for r in summary['results'] if r['endpoint'] == 'final' and r['panel'] == 'validation']
    return (summary['status'] == 'complete' and audit['passed'] and summary['precision_agreement']['passed']
            and len(rows) == 2 and {r['precision'] for r in rows} == {'fp32', 'fp64'}
            and all(r['recovery_qualified'] for r in rows))


def run_case(root, name, arm, dataset, evaluation_path, start, initial, hidden, config):
    from scripts.audit_curriculum_repair import audit_case
    dest = root / name
    dest.mkdir()
    started = time.monotonic()
    atomic_json(dest / 'configuration.json', {'arm': arm, 'storage_rule': ARMS[arm],
        'dataset': str(dataset.relative_to(root)), 'evaluation': str(evaluation_path.relative_to(root))})
    with gzip.open(dataset / 'pool.json.gz', 'rt') as stream:
        pool = json.load(stream)
    schedule = torch.load(dataset / 'schedule.pt', weights_only=True, map_location='cpu')
    panels = {'validation': json.loads(evaluation_path.read_text()),
              'train_probe': json.loads((dataset / 'train-probe.json').read_text())}
    final = optimize(initial, hidden, pool, schedule, ARMS[arm], dest, root, config)
    results = evaluate(start, initial, final, ARMS[arm], arm, panels, dest)
    agreement = precision_agreement(dest)
    valid = all(r['correspondence']['passed'] for r in results) and agreement['passed']
    summary = {'status': 'complete' if valid else 'numerical-validation-failed',
               'results': results, 'precision_agreement': agreement, 'seconds': time.monotonic() - started}
    atomic_json(dest / 'summary.json', summary)
    if not valid:
        raise RuntimeError('Numerical disagreement; preserved case outputs')
    audit = audit_case(root, dest)
    atomic_json(dest / 'audit.json', audit)
    check_resources(root, config, started)
    print(json.dumps({'case': name, 'phase': 'complete', 'seconds': time.monotonic() - started,
                      'final_gate': calibration_passes(summary, audit)}), flush=True)
    return summary, audit


def freeze(root, config):
    if not config['fixture']:
        validation = ROOT / 'artifacts/scc-curriculum-repair-20260913-v1/implementation-validation.json'
        receipt = json.loads(validation.read_text())
        assert receipt['passed']
        for name, expected in receipt['validated_source_hashes'].items():
            assert file_digest(ROOT / name) == expected, name
        for name, expected in receipt['fixture_evidence_hashes'].items():
            assert file_digest(ROOT / name) == expected, name
        shutil.copyfile(validation, root / 'implementation-validation.json')
    entry = (ROOT / 'labnotes.md').read_text().split('<a id="ln-072"></a>')[1].split('\n<a id=')[0].split('\n## Supporting-record')[0]
    (root / 'plan.md').write_text('<a id="ln-072"></a>' + entry)
    manifest = snapshot_sources(root / 'source')
    for directory in ('scripts', 'tests'):
        for source in sorted((ROOT / directory).glob('*.py')):
            relative = source.relative_to(ROOT)
            dest = root / 'source' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            manifest[str(relative)] = file_digest(dest)
    atomic_json(root / 'source-manifest.json', manifest)
    shutil.copyfile(PARENT, root / 'merged-start-parent.pt')
    atomic_json(root / 'parents.json', {str(PARENT): file_digest(PARENT)})
    atomic_json(root / 'configuration.json', config)


def execute(root, config):
    from scripts.audit_curriculum_repair import audit_batch
    parent = torch.load(root / 'merged-start-parent.pt', weights_only=True, map_location='cpu')
    assert parent['out']['admitted'][:, 1].all()
    start = parent['post_event_state']
    initial, hidden = start_values(start)
    torch.save({'payload': initial, 'hidden': hidden, 'state': start}, root / 'repair-origin.pt')
    write_data(root / 'calibration-data', config, config['calibration_seeds'])
    dev = make_evaluation(config['calibration_evaluation_seed'], config['evaluation_per_cell'])
    atomic_json(root / 'calibration-evaluation.json', dev)
    summary, audit = run_case(root, 'calibration', 'symbolic-binding-control', root / 'calibration-data',
                              root / 'calibration-evaluation.json', start, initial, hidden, config)
    passed = calibration_passes(summary, audit)
    cases = ['calibration']
    if config['fixture']:
        run_case(root, 'fixture-learned', 'learned-binding', root / 'calibration-data',
                 root / 'calibration-evaluation.json', start, initial, hidden, config)
        cases.append('fixture-learned')
    elif passed:
        excluded = {r['core_sha256'] for s in dev for r in s}
        heldout = make_evaluation(config['comparison_evaluation_seed'], config['evaluation_per_cell'], excluded)
        atomic_json(root / 'comparison-evaluation.json', heldout)
        for i, seeds in enumerate(config['paired_seeds'], 1):
            dataset = root / f'pair-{i}-data'
            write_data(dataset, config, seeds)
            for arm in ARMS:
                name = f'pair-{i}-{arm}'
                run_case(root, name, arm, dataset, root / 'comparison-evaluation.json', start, initial, hidden, config)
                cases.append(name)
    result = {'status': 'fixture-complete' if config['fixture'] else 'matched-complete' if passed else 'calibration-did-not-qualify',
              'calibration_qualified': passed, 'matched_comparison_started': passed and not config['fixture'],
              'completed_cases': cases}
    atomic_json(root / 'summary.json', result)
    atomic_json(root / 'audit.json', audit_batch(root))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = configuration(args.fixture)
    torch.set_num_threads(config['cpu_threads'])
    started = time.monotonic()

    def timeout(*_):
        raise TimeoutError('LN-072 whole-batch wall limit exceeded')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(config['batch_wall_seconds'])
    try:
        freeze(root, config)
        result = execute(root, config)
        check_resources(root, config)
        result['seconds'] = time.monotonic() - started
        result['bytes_before_manifest'] = directory_bytes(root)
        atomic_json(root / 'summary.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        atomic_json(root / 'failure.json', {'error': repr(exc), 'seconds': time.monotonic() - started})
        raise
    finally:
        signal.alarm(0)
        atomic_json(root / 'artifact-manifest.json', {
            str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'artifact-manifest.json'})


if __name__ == '__main__':
    main()
