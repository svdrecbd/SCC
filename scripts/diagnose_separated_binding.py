"""LN-085: four-way operator separation on the six saved final repairs."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.separated_binding import CONDITIONS, actual_request, functional_request
from scc.sharded_repair import restore, start_values
from scripts.diagnose_curriculum_persistence import correspondence, load
from scripts.run_learned_binding_bank import tensor_ids, score


@torch.no_grad()
def evaluate(state, payload, ids, parameter_rule, hidden_rule):
    model = restore(state, payload.dtype, parameter_rule, payload)
    shards = payload.reshape(1, 2, -1).expand(len(ids), -1, -1).clone()
    hidden = start_values(state)[1].to(payload.dtype)
    reduced, actual, diagnostics = [], [], []
    for j in range(ids.shape[1]):
        out, shards, hidden = functional_request(shards, hidden, ids[:, j], state['width'], parameter_rule, hidden_rule)
        expected = actual_request(model, ids[:, j], parameter_rule, hidden_rule)
        for value in (*out.values(), shards, hidden):
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite separated execution')
        reduced.append(out)
        actual.append(expected)
        diagnostics.append({'shard_difference_after': (shards[:, 0]-shards[:, 1]).norm(dim=-1),
                            'hidden_difference_after': (hidden[:, :state['width']//2]-hidden[:, state['width']//2:]).norm(dim=-1)})
    stack = lambda items: {k: torch.stack([x[k] for x in items], 1) for k in items[0]}
    return stack(reduced), stack(actual), stack(diagnostics)


def rescore(rows, out):
    result = score(rows, out)
    target = torch.tensor([[r['family'] == 'lookup' for r in stream] for stream in rows])
    desired = torch.tensor([True, False, False, True]).expand_as(out['admitted']).clone()
    desired[:, :, 1] = target
    errors = int((out['admitted'] != desired).sum())
    result.pop('qualified')  # Original scorer checks equality, not the selective exception.
    result['exception_rule_errors'] = errors
    result['diagnostic_recovery_qualified'] = (len(result['cells']) == 6
        and all(c['qualified'] for c in result['cells'].values())
        and result['target_n'] > 0 and result['target_correct_answers']/result['target_n'] >= .95 and errors == 0)
    return result


def freeze(parent, dest, fixture):
    config = {'plan': 'LN-085', 'fixture': fixture, 'conditions': CONDITIONS,
              'training_updates': 0, 'cpu_threads': 2, 'wall_seconds': 300,
              'output_limit_bytes': 256*1024**2, 'parent': str(parent.resolve()),
              'python': sys.version, 'torch': str(torch.__version__), 'platform': platform.platform()}
    atomic_json(dest/'configuration.json', config)
    notes = (ROOT/'labnotes.md').read_text()
    (dest/'plan.md').write_text('### LN-085'+notes.split('### LN-085')[1].split('\n### LN-')[0].split('\n## Supporting-record')[0])
    sources = snapshot_sources(dest/'source')
    for source in sorted((ROOT/'scripts').glob('*.py')) + [ROOT/'tests/test_separated_binding.py']:
        relative = source.relative_to(ROOT)
        target = dest/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[str(relative)] = file_digest(target)
    atomic_json(dest/'source-manifest.json', sources)
    manifest = json.loads((parent/'artifact-manifest.json').read_text())
    cases = ['pair-1-learned-binding'] if fixture else [f'pair-{i}-{arm}' for i in (1, 2, 3)
                    for arm in ('learned-binding', 'symbolic-binding-control')]
    names = ['repair-origin.pt', 'comparison-evaluation.json']
    for case in cases:
        names += [f'{case}/update-12000.pt', f'pair-{case.split("-")[1]}-data/train-probe.json']
        names += [f'{case}/final-{precision}-{panel}.pt' for precision, panel in
                  (('fp32', 'validation'), ('fp64', 'validation'), ('fp32', 'train_probe'))]
    hashes = {}
    for name in sorted(set(names)):
        assert file_digest(parent/name) == manifest[name], name
        target = dest/'inputs'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent/name, target)
        hashes[name] = file_digest(target)
        assert hashes[name] == manifest[name]
    atomic_json(dest/'input-manifest.json', hashes)
    return config, cases


def execute(parent, dest, fixture=False):
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    def timeout(*_):
        raise TimeoutError('LN-085 300-second wall limit')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        config, cases = freeze(parent, dest, fixture)
        inputs = dest/'inputs'
        state = load(inputs/'repair-origin.pt')['state']
        validation = json.loads((inputs/'comparison-evaluation.json').read_text())
        results = []
        for case in cases:
            payload = load(inputs/case/'update-12000.pt')['payload']
            train = json.loads((inputs/f'pair-{case.split("-")[1]}-data/train-probe.json').read_text())
            final_outputs = {}
            for precision, panel in (('fp32', 'validation'), ('fp64', 'validation'), ('fp32', 'train_probe')):
                dtype = torch.float32 if precision == 'fp32' else torch.float64
                rows = validation if panel == 'validation' else train
                if fixture:
                    rows = [stream[:8] for stream in rows]
                ids = tensor_ids(rows)
                saved = load(inputs/case/f'final-{precision}-{panel}.pt')['out']
                saved = {k: v[:, :ids.shape[1]] for k, v in saved.items()}
                baseline_condition = 'neither' if case.endswith('symbolic-binding-control') else 'both'
                for condition, rules in CONDITIONS.items():
                    out, actual, diagnostics = evaluate(state, payload.to(dtype), ids, *rules)
                    check = correspondence(out, actual, dtype)
                    original = correspondence(out, saved, dtype) if condition == baseline_condition else None
                    name = f'{case}-{precision}-{panel}-{condition}'
                    torch.save({'out': out, 'actual': actual, 'diagnostics': diagnostics}, dest/(name+'.pt'))
                    result = {'name': name, 'case': case, 'precision': precision, 'panel': panel,
                              'condition': condition, 'metrics': rescore(rows, out), 'correspondence': check,
                              'original_endpoint_correspondence': original,
                              'baseline_decision_changes': int((out['logits'].argmax(-1) != saved['logits'].argmax(-1)).sum()),
                              'baseline_admission_changes': int((out['admitted'] != saved['admitted']).sum())}
                    if panel == 'validation':
                        if precision == 'fp32':
                            final_outputs[condition] = out
                        else:
                            earlier = final_outputs[condition]
                            result['precision_decisions_agree'] = (torch.equal(earlier['logits'].argmax(-1), out['logits'].argmax(-1))
                                and torch.equal(earlier['admitted'], out['admitted']))
                    results.append(result)
                    atomic_json(dest/'partial-results.json', results)
                    assert check['passed'] and (original is None or original['passed']), name
                    assert result.get('precision_decisions_agree', True), name
                    if sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) > config['output_limit_bytes']:
                        raise RuntimeError('LN-085 output ceiling')
            print(json.dumps({'case': case, 'seconds': time.monotonic()-started}), flush=True)
        for name, expected in json.loads((dest/'input-manifest.json').read_text()).items():
            assert file_digest(inputs/name) == file_digest(parent/name) == expected
        for name, expected in json.loads((dest/'source-manifest.json').read_text()).items():
            assert file_digest(ROOT/name) == expected, name
        atomic_json(dest/'summary.json', {'status': 'fixture-complete' if fixture else 'complete',
                    'results': results, 'seconds': time.monotonic()-started, 'inputs_and_sources_unchanged': True})
    except BaseException as exc:
        atomic_json(dest/'failure.json', {'error': repr(exc), 'seconds': time.monotonic()-started})
        raise
    finally:
        signal.alarm(0)
        atomic_json(dest/'artifact-manifest.json', {str(p.relative_to(dest)): file_digest(p)
                    for p in sorted(dest.rglob('*')) if p.is_file() and p.name != 'artifact-manifest.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    execute(args.parent, args.output, args.fixture)
