"""LN-089: matched four-condition repair training on one GMAN CPU host."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import gzip
import json
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.separated_binding import CONDITIONS, functional_window
from scc.sharded_repair import start_values
from scripts.run_curriculum_repair import configuration, objective, learning_rate, active_length
from scripts.diagnose_separated_binding import evaluate, rescore
from scripts.diagnose_curriculum_persistence import correspondence, load


def size(root):
    total = 0
    for path in root.rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # Other workers atomically rename their temporary JSON files.
            continue
    return total


def tensors(inputs, pair):
    with gzip.open(inputs/f'pair-{pair}-data/pool.json.gz', 'rt') as stream:
        rows = json.load(stream)
    return (torch.tensor([r['tokens'] for r in rows]),
            torch.tensor([r['label'] for r in rows]),
            torch.tensor([r['family'] == 'lookup' for r in rows]),
            load(inputs/f'pair-{pair}-data/schedule.pt'))


def step(q, optimizer, hidden, ids, labels, target, rules):
    out, _, _ = functional_window(q, hidden, ids, 128, *rules)
    loss, components = objective(out, labels, target)
    optimizer.zero_grad()
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_([q], 1., error_if_nonfinite=True))
    optimizer.step()
    return {'loss_before_update': float(loss.detach()), 'gradient_norm_before_clip': norm,
            'window_task_correct': int((out['logits'].argmax(-1) == labels).sum()), **components}


def optimize(root, dest, initial, hidden, dataset, rules, config):
    started = time.monotonic()
    q = torch.nn.Parameter(initial.clone())
    opt = torch.optim.Adam([q], lr=learning_rate(0, config))
    ids, labels, target, schedule = dataset
    h = hidden.repeat(8, 1)
    def save(ordinal):
        torch.save({'payload': q.detach().clone(), 'step': ordinal, 'optimizer': opt.state_dict()},
                   dest/f'update-{ordinal:05d}.pt')
        if size(root) > config['output_limit_bytes']:
            raise RuntimeError('LN-089 output cap exceeded')
    save(0)
    try:
        with (dest/'training.jsonl').open('w') as log:
            for ordinal in range(config['steps']):
                if time.monotonic()-started > config['trajectory_wall_seconds']:
                    raise TimeoutError('LN-089 trajectory training wall limit')
                lr = learning_rate(ordinal, config)
                opt.param_groups[0]['lr'] = lr
                selected = schedule[ordinal].long()
                row = step(q, opt, h, ids[selected], labels[selected], target[selected], rules)
                row.update(update=ordinal+1, learning_rate=lr, active_length=active_length(ordinal, config),
                           seconds=time.monotonic()-started)
                log.write(json.dumps(row, separators=(',', ':'))+'\n')
                if ordinal+1 in config['checkpoint_updates']:
                    save(ordinal+1)
                    log.flush()
        return q.detach().clone()
    except BaseException:
        torch.save({'payload': q.detach().clone(), 'optimizer': opt.state_dict()}, dest/'interrupted.pt')
        raise


def benchmark(root, initial, hidden, dataset, fixture):
    results = []
    ids, labels, target, schedule = dataset
    warmup, timed = (1, 2) if fixture else (16, 64)
    for name in ('parameter_only', 'hidden_only'):
        q = torch.nn.Parameter(initial.clone())
        opt = torch.optim.Adam([q], lr=.003)
        started = None
        for i in range(warmup+timed):
            if i == warmup:
                started = time.monotonic()
            chosen = schedule[2000+i].long()
            step(q, opt, hidden.repeat(8, 1), ids[chosen], labels[chosen], target[chosen], CONDITIONS[name])
        elapsed = time.monotonic()-started
        results.append({'condition': name, 'warmup_updates': warmup, 'timed_updates': timed,
                        'active_length': 12, 'seconds_per_update': elapsed/timed,
                        'estimated_12000_training_seconds': elapsed/timed*12000})
    atomic_json(root/'benchmark.json', {'discarded_optimizer_and_weights': True, 'results': results})
    return results


def runtime_readiness(results, config):
    """Reject budgets that the measured solo rate already predicts will fail."""
    training = max(r['seconds_per_update'] for r in results)*config['steps']
    case = 1.5*training + 300
    batch = len(CONDITIONS)*case + 300
    return {'passed': case <= config['trajectory_wall_seconds'] and batch <= config['batch_wall_seconds'],
            'estimated_case_seconds': case, 'estimated_batch_seconds': batch,
            'concurrency_factor': 1.5, 'evaluation_allowance_seconds': 300,
            'scope': 'Conservative readiness heuristic, not a runtime guarantee'}


def run_case(root, pair, condition, dataset, config):
    dest = root/f'pair-{pair}-{condition}'
    dest.mkdir(exist_ok=False)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('LN-089 case wall cap')))
    signal.alarm(config['trajectory_wall_seconds'])
    try:
        origin = load(root/'inputs/repair-origin.pt')
        state = origin['state']
        initial, hidden = start_values(state)
        assert torch.equal(initial, origin['payload']) and torch.equal(hidden, origin['hidden'])
        rules = CONDITIONS[condition]
        atomic_json(dest/'configuration.json', {'pair': pair, 'condition': condition, 'rules': rules,
            'steps': config['steps'], 'origin_sha256': file_digest(root/'inputs/repair-origin.pt'),
            'pool_sha256': file_digest(root/f'inputs/pair-{pair}-data/pool.json.gz'),
            'schedule_sha256': file_digest(root/f'inputs/pair-{pair}-data/schedule.pt')})
        final = optimize(root, dest, initial, hidden, dataset, rules, config)
        panels = {'validation': json.loads((root/'inputs/comparison-evaluation.json').read_text()),
                  'train_probe': json.loads((root/f'inputs/pair-{pair}-data/train-probe.json').read_text())}
        results = []
        for endpoint, payload in (('initial', initial), ('final', final)):
            fp32 = None
            for precision, panel in (('fp32', 'validation'), ('fp64', 'validation'), ('fp32', 'train_probe')):
                dtype = torch.float32 if precision == 'fp32' else torch.float64
                rows = panels[panel]
                if config['fixture']:
                    rows = [s[:8] for s in rows]
                ids = torch.tensor([[r['tokens'] for r in s] for s in rows])
                out, actual, diagnostics = evaluate(state, payload.to(dtype), ids, *rules)
                check = correspondence(out, actual, dtype)
                name = f'{endpoint}-{precision}-{panel}'
                torch.save({'out': out, 'actual': actual, 'diagnostics': diagnostics}, dest/(name+'.pt'))
                result = {'name': name, 'endpoint': endpoint, 'precision': precision, 'panel': panel,
                          'metrics': rescore(rows, out), 'correspondence': check}
                if panel == 'validation':
                    if precision == 'fp32':
                        fp32 = out
                    else:
                        result['precision_decisions_agree'] = (torch.equal(fp32['logits'].argmax(-1), out['logits'].argmax(-1))
                            and torch.equal(fp32['admitted'], out['admitted']))
                results.append(result)
                atomic_json(dest/'partial-results.json', results)
                assert check['passed'] and result.get('precision_decisions_agree', True), name
        audit = audit_case(root, dest, config)
        atomic_json(dest/'audit.json', audit)
        atomic_json(dest/'summary.json', {'status': 'complete', 'results': results,
                                         'seconds': time.monotonic()-started})
        print(json.dumps({'case': dest.name, 'seconds': time.monotonic()-started}), flush=True)
    except BaseException as exc:
        atomic_json(dest/'failure.json', {'error': repr(exc), 'seconds': time.monotonic()-started})
        raise
    finally:
        signal.alarm(0)


def audit_case(root, dest, config):
    """Rescore saved outputs and verify the training contract independently."""
    case = json.loads((dest/'configuration.json').read_text())
    pair = case['pair']
    initial = load(root/'inputs/repair-origin.pt')['payload']
    assert torch.equal(load(dest/'update-00000.pt')['payload'], initial)
    log = [json.loads(line) for line in (dest/'training.jsonl').read_text().splitlines()]
    assert len(log) == config['steps']
    for i, row in enumerate(log):
        assert row['update'] == i+1 and row['learning_rate'] == learning_rate(i, config)
        assert row['active_length'] == active_length(i, config)
    for i in config['checkpoint_updates']:
        checkpoint = load(dest/f'update-{i:05d}.pt')
        assert checkpoint['step'] == i and checkpoint['payload'].numel() == 80518
        opt = checkpoint['optimizer']
        assert opt['param_groups'][0]['weight_decay'] == 0
        assert all(int(v['step']) == i for v in opt['state'].values())
    predictions = 0
    for r in json.loads((dest/'partial-results.json').read_text()):
        panel = 'comparison-evaluation.json' if r['panel'] == 'validation' else f'pair-{pair}-data/train-probe.json'
        rows = json.loads((root/'inputs'/panel).read_text())
        if config['fixture']:
            rows = [s[:8] for s in rows]
        data = load(dest/(r['name']+'.pt'))
        out, actual = data['out'], data['actual']
        labels = torch.tensor([[x['label'] for x in s] for s in rows])
        correct = out['logits'].argmax(-1) == labels
        assert float(correct.double().mean()) == r['metrics']['task_accuracy']
        for key, c in r['metrics']['cells'].items():
            family, layout = key.split('/')
            mask = torch.tensor([[x['family'] == family and x['layout'] == layout for x in s] for s in rows])
            assert int(correct[mask].sum()) == c['correct'] and int(mask.sum()) == c['n']
        desired = torch.tensor([True, False, False, True]).expand_as(out['admitted']).clone()
        target = torch.tensor([[x['family'] == 'lookup' for x in s] for s in rows])
        desired[:, :, 1] = target
        assert int((out['admitted'] != desired).sum()) == r['metrics']['exception_rule_errors']
        assert int((out['emitted'][:, :, 1][target] == labels[target]).sum()) == r['metrics']['target_correct_answers']
        assert correspondence(out, actual, out['logits'].dtype)['passed']
        predictions += labels.numel()
    return {'passed': True, 'task_predictions_rescored': predictions, 'training_contract_checked': True}


def freeze(inputs, root, fixture):
    config = configuration(False)
    config.update(plan='LN-096', fixture=fixture, conditions=CONDITIONS, workers=3,
                  trajectory_wall_seconds=7200, batch_wall_seconds=30600,
                  output_limit_bytes=1024**3,
                  runtime={'python': sys.version, 'torch': str(torch.__version__),
                           'platform': platform.platform(), 'processor': platform.processor()})
    if fixture:
        config.update(steps=8, checkpoint_updates=[0, 4, 8], batch_wall_seconds=300)
    atomic_json(root/'configuration.json', config)
    shutil.copyfile(inputs.parent/'plan.md', root/'plan.md')
    receipt = json.loads((inputs.parent/'validation.json').read_text())
    assert receipt['passed']
    for name, expected in receipt['validated_source_hashes'].items():
        assert file_digest(ROOT/name) == expected, name
    shutil.copyfile(inputs.parent/'validation.json', root/'implementation-validation.json')
    manifest = json.loads((inputs/'manifest.json').read_text())
    for name, expected in manifest.items():
        assert file_digest(inputs/name) == expected, name
    shutil.copytree(inputs, root/'inputs')
    source = snapshot_sources(root/'source')
    for path in sorted((ROOT/'scripts').glob('*.py')):
        target = root/'source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source[str(path.relative_to(ROOT))] = file_digest(target)
    atomic_json(root/'source-manifest.json', source)
    return config


def coordinate(inputs, root, fixture):
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    processes, handles = [], []
    try:
        config = freeze(inputs, root, fixture)
        origin = load(root/'inputs/repair-origin.pt')
        rates = benchmark(root, origin['payload'], origin['hidden'], tensors(root/'inputs', 1), fixture)
        if not fixture:
            readiness = runtime_readiness(rates, config)
            atomic_json(root/'runtime-readiness.json', readiness)
            if not readiness['passed']:
                raise RuntimeError('Measured runtime exceeds configured case or batch allowance')
        for pair in (1, 2, 3):
            handle = (root/f'pair-{pair}.log').open('w')
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--output', str(root.resolve()), '--pair', str(pair)], stdout=handle, stderr=subprocess.STDOUT))
        with ThreadPoolExecutor(max_workers=3) as executor:
            pending = {executor.submit(p.wait): p for p in processes}
            while pending:
                remaining = config['batch_wall_seconds']-(time.monotonic()-started)
                done, _ = wait(pending, timeout=max(0, remaining), return_when=FIRST_COMPLETED)
                if not done or any(f.result() != 0 for f in done):
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
                    for p in processes:
                        try:
                            p.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            p.kill()
                    raise RuntimeError('Worker failed or batch wall limit exceeded; partial cases preserved')
                for f in done:
                    del pending[f]
        cases = [f'pair-{pair}-{condition}' for pair in (1, 2, 3) for condition in CONDITIONS]
        assert all(json.loads((root/case/'audit.json').read_text())['passed'] for case in cases)
        for name, expected in json.loads((root/'inputs/manifest.json').read_text()).items():
            assert file_digest(root/'inputs'/name) == expected
        for name, expected in json.loads((root/'source-manifest.json').read_text()).items():
            assert file_digest(ROOT/name) == expected
        assert size(root) <= config['output_limit_bytes']
        atomic_json(root/'summary.json', {'status': 'fixture-complete' if fixture else 'complete',
                    'cases': cases, 'seconds': time.monotonic()-started})
        atomic_json(root/'result.json', {'ok': True, 'completed_trajectories': len(cases),
                    'scientific_recovery_not_required_for_execution_success': True})
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        atomic_json(root/'failure.json', {'error': repr(exc), 'seconds': time.monotonic()-started})
        raise
    finally:
        for handle in handles:
            handle.close()
        atomic_json(root/'artifact-manifest.json', {str(p.relative_to(root)): file_digest(p)
                    for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'artifact-manifest.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pair', type=int, choices=(1, 2, 3))
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    if args.pair:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        config = json.loads((args.output/'configuration.json').read_text())
        dataset = tensors(args.output/'inputs', args.pair)
        for condition in CONDITIONS:
            run_case(args.output, args.pair, condition, dataset, config)
    else:
        coordinate(args.inputs, args.output, args.fixture)
