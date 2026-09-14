"""LN-080: saved-checkpoint diagnostics; reset scores never qualify an escape."""
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
from scc.sharded_repair import functional_request, restore, start_values
from scripts.run_learned_binding_bank import score, tensor_ids

MODES = ('continuous', 'request_reset', 'hidden_reset', 'window4_reset')
STEPS = (0, 400, 1000, 2000, 6000, 9000, 12000)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def resets(mode, ordinal):
    if mode not in MODES:
        raise ValueError(mode)
    both = mode == 'request_reset' or (mode == 'window4_reset' and ordinal % 4 == 0)
    return both, both or mode == 'hidden_reset'


@torch.no_grad()
def reduced(payload, hidden, ids, width, rule, mode):
    initial = payload.reshape(1, 2, -1).expand(len(ids), -1, -1)
    shards, h = initial.clone(), hidden.clone()
    outputs, diagnostics = [], []
    for j in range(ids.shape[1]):
        reset_q, reset_h = resets(mode, j)
        if reset_q:
            shards = initial.clone()
        if reset_h:
            h = hidden.clone()
        before = {'hidden_norm_before': h.norm(dim=-1),
                  'shard_difference_before': (shards[:, 0] - shards[:, 1]).norm(dim=-1)}
        out, shards, h = functional_request(shards, h, ids[:, j], width, rule)
        for value in (*out.values(), shards, h):
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite reduced execution')
        out['emitted'] = torch.where(out['admitted'], out['logits'].argmax(-1)[:, None], 3)
        outputs.append(out)
        diagnostics.append(before)
    stack = lambda items: {k: torch.stack([x[k] for x in items], 1) for k in items[0]}
    return stack(outputs), stack(diagnostics)


@torch.no_grad()
def actual(state, payload, ids, rule, mode):
    model = restore(state, payload.dtype, rule, payload)
    initial_hidden = model.hidden_bank.clone()
    initial_phase = model.phase.clone()
    outputs = []
    for j in range(ids.shape[1]):
        reset_q, reset_h = resets(mode, j)
        if reset_q:
            model = restore(state, payload.dtype, rule, payload)
        elif reset_h:
            model.hidden_bank = initial_hidden.clone()
            model.phase = initial_phase.clone()
        outputs.append(model.request(ids[:, j]))
    return {k: torch.stack([x[k] for x in outputs], 1) for k in outputs[0]}


def correspondence(a, b, dtype):
    result = {k + '_max_error': float((a[k] - b[k]).abs().max())
              for k in ('logits', 'policy_logits')}
    result['decision_disagreements'] = int((a['logits'].argmax(-1) != b['logits'].argmax(-1)).sum())
    result['admission_disagreements'] = int((a['admitted'] != b['admitted']).sum())
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    result['passed'] = (max(result['logits_max_error'], result['policy_logits_max_error']) <= tolerance
                        and result['decision_disagreements'] == result['admission_disagreements'] == 0)
    return result


def metrics(rows, out):
    result = score(rows, out)
    labels = torch.tensor([[r['label'] for r in stream] for stream in rows])
    correct = out['logits'].argmax(-1) == labels
    desired = torch.tensor([True, False, False, True]).expand_as(out['admitted']).clone()
    desired[:, :, 1] = torch.tensor([[r['family'] == 'lookup' for r in stream] for stream in rows])
    # The legacy scorer's "qualified" applies the original equality policy;
    # expose no qualification boolean for these post-hoc diagnostic conditions.
    result.pop('qualified')
    for cell in result['cells'].values():
        cell.pop('qualified')
    result['exception_rule_errors'] = int((out['admitted'] != desired).sum())
    result['quartiles'] = [float(x.double().mean()) for x in correct.tensor_split(4, dim=1)]
    result['families'] = {}
    for family in ('lookup', 'parity', 'sum3'):
        mask = torch.tensor([[r['family'] == family for r in stream] for stream in rows])
        result['families'][family] = {'correct': int(correct[mask].sum()), 'n': int(mask.sum())}
    return result


def freeze(parent, dest, fixture):
    config = {'plan': 'LN-080', 'fixture': fixture, 'steps': [12000] if fixture else list(STEPS),
              'modes': MODES, 'training_updates': 0, 'cpu_threads': 2,
              'wall_seconds': 900, 'output_limit_bytes': 512 * 1024**2,
              'parent': str(parent.resolve()), 'python': sys.version,
              'torch': str(torch.__version__), 'platform': platform.platform()}
    atomic_json(dest / 'configuration.json', config)
    notes = (ROOT / 'labnotes.md').read_text()
    (dest / 'plan.md').write_text(notes.split('### LN-080')[1].split('\n## Supporting-record')[0])
    sources = snapshot_sources(dest / 'source')
    for source in sorted((ROOT / 'scripts').glob('*.py')) + [ROOT / 'tests/test_curriculum_persistence.py']:
        relative = source.relative_to(ROOT)
        target = dest / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[str(relative)] = file_digest(target)
    atomic_json(dest / 'source-manifest.json', sources)
    manifest = json.loads((parent / 'artifact-manifest.json').read_text())
    names = ['repair-origin.pt', 'comparison-evaluation.json']
    cases = ['pair-1-learned-binding'] if fixture else [f'pair-{i}-{arm}' for i in (1, 2, 3)
                  for arm in ('learned-binding', 'symbolic-binding-control')]
    for case in cases:
        names += [f'{case}/configuration.json', f'{case}/training.jsonl']
        names += [f'{case}/update-{step:05d}.pt' for step in config['steps']]
        names += [f'{case}/final-{precision}-{panel}.pt'
                  for precision, panel in (('fp32', 'validation'), ('fp64', 'validation'), ('fp32', 'train_probe'))]
        names += [f'pair-{case.split("-")[1]}-data/train-probe.json']
    hashes = {}
    for name in sorted(set(names)):
        assert file_digest(parent / name) == manifest[name], name
        target = dest / 'inputs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, target)
        hashes[name] = file_digest(target)
        assert hashes[name] == manifest[name]
    atomic_json(dest / 'input-manifest.json', hashes)
    return config, cases


def execute(parent, dest, fixture=False):
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    def timeout(*_):
        raise TimeoutError('LN-080 900-second wall limit')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    try:
        config, cases = freeze(parent, dest, fixture)
        inputs = dest / 'inputs'
        state = load(inputs / 'repair-origin.pt')['state']
        hidden = start_values(state)[1]
        validation = json.loads((inputs / 'comparison-evaluation.json').read_text())
        results, log_summaries = [], {}
        for case in cases:
            rule = json.loads((inputs / case / 'configuration.json').read_text())['storage_rule']
            train = json.loads((inputs / f'pair-{case.split("-")[1]}-data/train-probe.json').read_text())
            logs = [json.loads(line) for line in (inputs / case / 'training.jsonl').read_text().splitlines()]
            log_summaries[case] = []
            for lower, upper in zip(STEPS[:-1], STEPS[1:]):
                chosen = [r for r in logs if lower < r['update'] <= upper]
                log_summaries[case].append({'after_update': lower, 'through_update': upper,
                    'mean_window_accuracy': sum(r['window_task_correct'] for r in chosen) / (128 * len(chosen)),
                    'mean_loss': sum(r['loss_before_update'] for r in chosen) / len(chosen),
                    'last_loss': chosen[-1]['loss_before_update']})
            for step in config['steps']:
                payload = load(inputs / case / f'update-{step:05d}.pt')['payload']
                conditions = [('fp32', 'validation')]
                if step == 12000:
                    conditions += [('fp64', 'validation'), ('fp32', 'train_probe')]
                final_outputs = {}
                for precision, panel in conditions:
                    dtype = getattr(torch, 'float32' if precision == 'fp32' else 'float64')
                    rows = validation if panel == 'validation' else train
                    if fixture:
                        rows = [stream[:8] for stream in rows]
                    ids = tensor_ids(rows)
                    for mode in MODES if step == 12000 else MODES[:2]:
                        out, diagnostics = reduced(payload.to(dtype), hidden.to(dtype), ids, state['width'], rule, mode)
                        count = ids.shape[1] if step == 12000 and mode == 'request_reset' and panel == 'validation' else min(8, ids.shape[1])
                        full = actual(state, payload.to(dtype), ids[:, :count], rule, mode)
                        check = correspondence({k: v[:, :count] for k, v in out.items()}, full, dtype)
                        saved_check = None
                        if step == 12000 and mode == 'continuous':
                            saved = load(inputs / case / f'final-{precision}-{panel}.pt')['out']
                            saved_check = correspondence(out, {k: v[:, :ids.shape[1]] for k, v in saved.items()}, dtype)
                        name = f'{case}-{step:05d}-{precision}-{panel}-{mode}'
                        torch.save({'out': out, 'diagnostics': diagnostics}, dest / (name + '.pt'))
                        result = {'name': name, 'case': case, 'step': step, 'precision': precision,
                                  'panel': panel, 'mode': mode, 'metrics': metrics(rows, out),
                                  'actual_requests_per_stream': count, 'correspondence': check,
                                  'saved_endpoint_correspondence': saved_check}
                        results.append(result)
                        atomic_json(dest / 'partial-results.json', results)
                        assert check['passed'] and (saved_check is None or saved_check['passed']), name
                        if step == 12000 and panel == 'validation':
                            if precision == 'fp32':
                                final_outputs[mode] = out
                            else:
                                a = final_outputs[mode]
                                agreement = (torch.equal(a['logits'].argmax(-1), out['logits'].argmax(-1))
                                             and torch.equal(a['admitted'], out['admitted']))
                                result['precision_decisions_agree'] = agreement
                                assert agreement, name
                        if sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) > config['output_limit_bytes']:
                            raise RuntimeError('LN-080 output ceiling')
                print(json.dumps({'case': case, 'step': step, 'seconds': time.monotonic() - started}), flush=True)
        for name, expected in json.loads((dest / 'input-manifest.json').read_text()).items():
            assert file_digest(inputs / name) == expected
            assert file_digest(parent / name) == expected
        for name, expected in json.loads((dest / 'source-manifest.json').read_text()).items():
            assert file_digest(ROOT / name) == expected, name
        atomic_json(dest / 'summary.json', {'status': 'fixture-complete' if fixture else 'complete',
                    'results': results, 'training_intervals': log_summaries,
                    'seconds': time.monotonic() - started, 'inputs_and_sources_unchanged': True})
    except BaseException as exc:
        atomic_json(dest / 'failure.json', {'error': repr(exc), 'seconds': time.monotonic() - started})
        raise
    finally:
        signal.alarm(0)
        atomic_json(dest / 'artifact-manifest.json', {str(p.relative_to(dest)): file_digest(p)
                    for p in sorted(dest.rglob('*')) if p.is_file() and p.name != 'artifact-manifest.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    execute(args.parent, args.output, args.fixture)
