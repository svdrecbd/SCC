"""Bounded local fitting, generalization and persistence diagnostics; no SCC claim."""
import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import replace
from itertools import product
import json
import math
from pathlib import Path
import random
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from scc.persistent_matrix import MatrixConfig
from scc.persistent_feedback import feedback_step, feedback_window
from scc.persistent_reference import PersistentGRU
from scc.persistent_tasks import (Request, FAMILIES, CONTEXTS, LAYOUTS, core_split,
    sample_request, input_code, tensors, evaluation_requests)
from scc.provenance import atomic_json, canonical_json, digest, file_digest, snapshot_sources, source_manifest

SEED = 17313001
PARENTS = {
    'feedback': ROOT/'artifacts/scc-feedback-failure-20260913-v1/collected/persistent-feedback',
    'control': ROOT/'artifacts/scc-persistent-followup-20260913-v1/collected/persistent-optimization',
    'gru': ROOT/'artifacts/scc-persistent-reference-implementation-20260912-v1/stabilization-v1',
}
STAGES = ('initial.pt', 'stage-2000.pt', 'stage-4000.pt', 'stage-6000.pt', 'stage-9000.pt', 'trained.pt')


def core_requests(family, length, split):
    """Distinct cores; short sets are exhaustive, long validation is balanced."""
    if length <= 4 or (family == 'parity' and length == 8):
        rows = []
        for values in product(range(2 if family == 'parity' else 3), repeat=length):
            full = values + (0,)*(12-length)
            for query in range(length) if family == 'lookup' else (0,):
                if core_split(family, full, query)[0] == split:
                    rows.append(Request(family, 'ungated', 'original', full, query, 0, 0))
        if length == 8:
            groups = defaultdict(list)
            for row in rows: groups[row.answer].append(row)
            count = min(16, *(len(v) for v in groups.values()))
            rng = random.Random(f'localization/{SEED}/{family}/{length}')
            for values in groups.values(): rng.shuffle(values)
            rows = [r for key in sorted(groups) for r in groups[key][:count]]
        return rows
    if split != 'validation':
        raise ValueError('Long sets are validation only')
    rng = random.Random(f'localization/{SEED}/{family}/{length}')
    groups = defaultdict(list); seen = set()
    classes = range(2 if family == 'parity' else 3)
    for _ in range(100000):
        r = sample_request(rng, (family, 'ungated', 'original'), split=split, active_length=length)
        if r.partition[1] not in seen and len(groups[r.answer]) < 16:
            r = replace(r, requester=0, owner=0)
            groups[r.answer].append(r); seen.add(r.partition[1])
        if all(len(groups[k]) == 16 for k in classes):
            return [r for k in classes for r in groups[k]]
    raise ValueError('Could not populate balanced validation set')


def generate_data():
    fit, panel = {}, []
    for family in FAMILIES:
        groups = defaultdict(list)
        for r in core_requests(family, 4, 'train'):
            groups[r.answer].append(r)
        n = min(8, *(len(v) for v in groups.values()))
        rng = random.Random(f'fit/{SEED}/{family}')
        fit[family] = []
        for answer in sorted(groups):
            rng.shuffle(groups[answer])
            fit[family].extend(groups[answer][:n])
        rng.shuffle(fit[family])
        for length in (2, 4, 8, 12):
            for i, r in enumerate(core_requests(family, length, 'validation')):
                for context in CONTEXTS:
                    for layout in LAYOUTS:
                        who = i % 2
                        variant = replace(r, context=context, layout=layout, requester=who,
                                          owner=1-who if context == 'unauthorized' else who)
                        panel.append({**variant.record(), 'active_length': length})
    return {k: [{**r.record(), 'active_length': 4} for r in rows] for k, rows in fit.items()}, panel


def independent_check(row):
    """Decode the token interface and independently reconstruct answer and split."""
    t = row['tokens']
    assert len(t) == 19 and t[0] == 0 and t[-1] == 1
    prefix = t[1:5]
    f = next(x for x in prefix if x in (2, 3, 4))
    mode = next(x for x in prefix if x in (5, 6))
    requester = next(x for x in prefix if x in (7, 8))-7
    owner = next(x for x in prefix if x in (9, 10))-9
    family = {2: 'lookup', 3: 'parity', 4: 'sum3'}[f]
    context = 'ungated' if mode == 5 else 'authorized' if requester == owner else 'unauthorized'
    layout = 'original' if prefix[0] == f else 'reordered'
    expected = [f, mode, requester+7, owner+9]
    assert prefix == (expected if layout == 'original' else expected[1:]+expected[:1])
    values, query = [x-11 for x in t[5:17]], t[17]-14
    assert all(v in range(2 if f == 3 else 3) for v in values) and 0 <= query < 12
    answer = values[query] if f == 2 else sum(values) % (2 if f == 3 else 3)
    if context == 'unauthorized': answer = 3
    assert (family, context, layout, values, query, requester, owner, answer) == (
        row['family'], row['context'], row['layout'], list(row['values']), row['query'],
        row['requester'], row['owner'], row['label'])
    import hashlib
    raw = json.dumps([family, values, query if f == 2 else None], separators=(',', ':'))
    key = hashlib.sha256(('scc-persistent-tasks/v1|'+raw).encode()).hexdigest()
    bucket = int(key[:8], 16) % 10
    assert key == row['core_sha256']
    assert row['split'] == ('validation' if bucket == 0 else 'test' if bucket == 1 else 'train')
    if 'logits' in row:
        assert len(row['logits']) == 4 and all(math.isfinite(v) for v in row['logits'])
        assert row['prediction'] == max(range(4), key=lambda k: row['logits'][k])


def scores(rows):
    groups = defaultdict(list)
    for r in rows:
        independent_check(r)
        key = '/'.join(str(r[k]) for k in ('active_length', 'family', 'context', 'layout'))
        groups[key].append(r)
    result = {}
    for key, members in groups.items():
        labels = Counter(r['label'] for r in members)
        result[key] = {'n': len(members), 'accuracy': sum(r['prediction'] == r['label'] for r in members)/len(members),
            'label_counts': dict(labels), 'majority_baseline': max(labels.values())/len(members),
            'unique_cores': len({r['core_sha256'] for r in members}),
            'prediction_counts': dict(Counter(r['prediction'] for r in members))}
    return result


class MatrixModel(nn.Module):
    def __init__(self, weights, wiring, strength):
        super().__init__()
        self.weights = nn.Parameter(weights.clone())
        self.config = MatrixConfig(weights.shape[1], 4, 'soft')
        self.register_buffer('wiring', wiring.clone())
        self.register_buffer('code', input_code(weights.shape[1], 'anchor', dtype=weights.dtype))
        self.strength = strength

    def forward(self, ids, state=None):
        y, state, _ = feedback_window(self.weights if state is None else state, ids,
                                      self.config, self.code, self.wiring, strength=self.strength)
        return y, state


def load_model(kind, stage):
    saved = torch.load(PARENTS[kind]/stage, weights_only=True, map_location='cpu')
    if kind == 'gru':
        model = PersistentGRU(saved['width']); model.load_state_dict(saved['model'])
    else:
        w = saved['initial_weights'] if stage == 'initial.pt' else saved['weights']
        wiring = torch.load(PARENTS['feedback']/'fixed-wiring.pt', weights_only=True, map_location='cpu')
        model = MatrixModel(w, wiring, 1. if kind == 'feedback' else 0.)
    return model


@torch.no_grad()
def fresh(model, rows):
    parts = []
    for start in range(0, len(rows), 64):
        ids = torch.tensor([[r['tokens']] for r in rows[start:start+64]])
        parts.append(model(ids)[0][:, 0])
    return torch.cat(parts)


def write_predictions(path, rows, y):
    assert y.shape == (len(rows), 4) and torch.isfinite(y).all()
    records = [{**r, 'logits': output, 'prediction': prediction}
               for r, output, prediction in zip(rows, y.tolist(), y.argmax(-1).tolist())]
    with path.open('w') as log:
        for r in records: log.write(canonical_json(r)+'\n')
    return {'predictions': len(records), 'sha256': file_digest(path), 'cells': scores(records)}


@torch.no_grad()
def persistence(model, streams, folder):
    folder.mkdir()
    ids = torch.tensor([[r['tokens'] for r in stream] for stream in streams])
    flat = [{**r, 'stream': s, 'request_index': i} for s, stream in enumerate(streams) for i, r in enumerate(stream)]
    batched, _ = model(ids)
    single = torch.cat([model(ids[s:s+1])[0] for s in range(len(streams))])
    double, _ = copy.deepcopy(model).double()(ids)
    reset = fresh(model, flat)
    result = {}
    for name, outputs in [('continuous', batched), ('batch1', single), ('float64', double), ('fresh', reset)]:
        y = outputs.reshape(-1, 4)
        entry = write_predictions(folder/(name+'.jsonl'), flat, y)
        entry['accuracy'] = sum(p == r['label'] for p, r in zip(y.argmax(-1).tolist(), flat))/len(flat)
        entry['late_half_accuracy'] = sum(int(y[j].argmax()) == r['label'] for j, r in enumerate(flat)
            if r['request_index'] >= len(streams[0])//2)/(len(flat)//2)
        if name in ('batch1', 'float64'):
            entry['maximum_logit_error'] = float((y-batched.reshape(-1, 4)).abs().max())
            entry['decision_mismatches'] = int((y.argmax(-1) != batched.reshape(-1, 4).argmax(-1)).sum())
            entry['numerical_gate_passed'] = entry['maximum_logit_error'] <= 1e-4 and entry['decision_mismatches'] == 0
        result[name] = entry
    atomic_json(folder/'result.json', result)
    return result


@torch.no_grad()
def trace_pair(model, row):
    other = copy.deepcopy(row)
    index = row['query'] if row['family'] == 'lookup' else 0
    modulus = 2 if row['family'] == 'parity' else 3
    other['tokens'][5+index] = 11+(row['values'][index]+1) % modulus
    ids = torch.tensor([row['tokens'], other['tokens']])
    state = (model.weights[None].expand(2, -1, -1).clone() if isinstance(model, MatrixModel)
             else torch.zeros(2, model.width))
    output = []
    for tick in range(19):
        previous = state
        if isinstance(model, MatrixModel):
            y, state, controls = feedback_step(state, model.code[ids[:, tick]], model.config,
                                               model.wiring, strength=model.strength)
            extra = {'beta_mean': float(controls['beta'].mean()),
                     'key_entropy_mean': float(-(controls['key_probabilities'] * controls['key_probabilities'].clamp_min(1e-30).log()).sum(-1).mean()),
                     'query_entropy_mean': float(-(controls['query_probabilities'] * controls['query_probabilities'].clamp_min(1e-30).log()).sum(-1).mean())}
        else:
            y, state = model.manual_tick(ids[:, tick], state); extra = {}
        output.append({'tick': tick, 'tokens': ids[:, tick].tolist(),
            'state_difference_norm': float((state[0]-state[1]).norm()),
            'output_difference_norm': float((y[0]-y[1]).norm()), 'logits': y.tolist(),
            'update_norm_mean': float((state-previous).flatten(1).norm(dim=1).mean()),
            'state_norm_mean': float(state.flatten(1).norm(dim=1).mean()), **extra})
    return {'request': row, 'altered_tokens': other['tokens'], 'changed_value_index': index, 'trace': output}


def gradient_blocks(model):
    if isinstance(model, MatrixModel):
        grad = model.weights.grad; width = model.config.input_size
        return {name: float(block.norm()) for name, block in (
            ('output', grad[:4]), ('key', grad[4:4+width]),
            ('query', grad[4+width:4+2*width]), ('rate', grad[-1:]))}
    return {name: float(p.grad.norm()) for name, p in model.named_parameters()}


def fit(model, rows, panel, folder, *, steps, seconds):
    folder.mkdir(); start = time.monotonic()
    ids = torch.tensor([[r['tokens']] for r in rows]); labels = torch.tensor([r['label'] for r in rows])
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    chain = digest('persistent-localization-fit/v1'); completed = 0
    with (folder/'training.jsonl').open('w', buffering=1) as log:
        for step in range(steps):
            if time.monotonic()-start >= seconds: break
            lr = .003 if step < 1000 else .0003
            for group in optimizer.param_groups: group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            y, state = model(ids); loss = F.cross_entropy(y[:, 0], labels)
            if not torch.isfinite(loss) or not torch.isfinite(state).all(): raise ValueError('Nonfinite fit')
            loss.backward(); blocks = gradient_blocks(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); completed = step+1
            row = {'step': completed, 'loss': float(loss.detach()),
                'accuracy': float((y[:, 0].argmax(-1) == labels).float().mean()),
                'gradient_norm_before_clip': float(norm), 'gradient_blocks': blocks,
                'lr': lr, 'batch_sha256': digest(rows), 'elapsed_seconds': time.monotonic()-start}
            chain = digest({'previous': chain, 'record': row})
            log.write(canonical_json({**row, 'chain': chain})+'\n')
    seconds_used = time.monotonic()-start
    torch.save({'model': model.state_dict(), 'steps': completed}, folder/'fitted.pt')
    y = fresh(model, rows)
    fitting = write_predictions(folder/'fit.jsonl', rows, y)
    fitting.update(accuracy=float((y.argmax(-1) == labels).float().mean()),
                   loss=float(F.cross_entropy(y, labels)))
    transfer = write_predictions(folder/'validation.jsonl', panel, fresh(model, panel))
    pool = [r for r in panel if r['active_length'] in (8, 12) and r['context'] == 'ungated' and r['layout'] == 'original']
    rng = random.Random(f'fit-streams/{SEED}/{rows[0]["family"]}')
    streams = [[copy.deepcopy(rng.choice(pool)) for _ in range(144)] for _ in range(2)]
    serial = persistence(model, streams, folder/'persistence')
    result = {'status': 'complete' if completed == steps else 'wall_limit', 'completed_steps': completed,
        'requested_steps': steps, 'training_seconds': seconds_used, 'chain': chain,
        'fit': fitting, 'transfer': transfer, 'persistence': serial,
        'fit_passed': completed == steps and fitting['accuracy'] == 1. and fitting['loss'] <= .05,
        'qualified_learnability': False, 'positive_scc_result': False}
    atomic_json(folder/'result.json', result)
    return result


def validation_gate():
    """Finite differences of the actual short fitting loss, plus continuation."""
    model = load_model('feedback', 'trained.pt').double()
    fit_rows, _ = generate_data()
    rows = fit_rows['parity'][:2]
    ids = torch.tensor([[r['tokens']] for r in rows]); labels = torch.tensor([r['label'] for r in rows])
    loss = F.cross_entropy(model(ids)[0][:, 0], labels)
    grad, = torch.autograd.grad(loss, model.weights)
    generator = torch.Generator().manual_seed(SEED)
    checks = []
    for direction in (grad.detach(), torch.randn(grad.shape, generator=generator, dtype=torch.float64)):
        direction = direction/direction.norm(); original = model.weights.detach().clone()
        with torch.no_grad():
            model.weights.copy_(original+1e-4*direction); plus = F.cross_entropy(model(ids)[0][:, 0], labels)
            model.weights.copy_(original-1e-4*direction); minus = F.cross_entropy(model(ids)[0][:, 0], labels)
            model.weights.copy_(original)
        finite = float((plus-minus)/2e-4); analytic = float((grad*direction).sum())
        error = abs(finite-analytic)
        assert error <= 1e-5+1e-3*abs(analytic), (error, finite, analytic)
        checks.append({'finite_difference': finite, 'analytic': analytic, 'absolute_error': error})
    requests = torch.tensor([[r['tokens'] for r in fit_rows['parity'][:4]]])
    with torch.no_grad():
        whole, end = model(requests); a, middle = model(requests[:, :2]); b, continued = model(requests[:, 2:], middle)
    assert torch.equal(whole, torch.cat((a, b), 1)) and torch.equal(end, continued)
    return {'passed': True, 'directional_derivatives': checks, 'continuation_exact': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--fixture', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    started = time.monotonic()
    def deadline(signum, frame): raise TimeoutError('Declared whole-process deadline reached')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(1800)
    source = snapshot_sources(a.output/'source'); shutil.copyfile(__file__, a.output/'runner.py')
    shutil.copyfile(a.plan, a.output/'plan.md')
    parents = {str(folder/name): file_digest(folder/name) for folder in PARENTS.values() for name in STAGES}
    wiring_path = PARENTS['feedback']/'fixed-wiring.pt'; parents[str(wiring_path)] = file_digest(wiring_path)
    atomic_json(a.output/'parents.json', parents)
    try:
        fits, panel = generate_data(); atomic_json(a.output/'data.json', {'fit': fits, 'validation': panel})
    except Exception as error:
        signal.alarm(0)
        atomic_json(a.output/'failure.json', {'type': type(error).__name__, 'message': str(error),
                    'stage': 'data_preparation', 'positive_scc_result': False})
        raise
    for rows in [*fits.values(), panel]:
        for row in rows: independent_check(row)
    assert not ({r['core_sha256'] for rows in fits.values() for r in rows} & {r['core_sha256'] for r in panel})
    atomic_json(a.output/'implementation-gate.json', validation_gate())
    result = {'schema': 'persistent-localization/v1', 'fixture': a.fixture, 'saved_states': {}, 'fits': {},
              'qualified_learnability': False, 'positive_scc_result': False}
    try:
        for kind in PARENTS:
            stages = ('trained.pt',) if a.fixture else STAGES
            for stage in stages:
                model = load_model(kind, stage)
                name = f'{kind}-{stage[:-3]}'
                result['saved_states'][name] = write_predictions(a.output/(name+'.jsonl'), panel, fresh(model, panel))
            model = load_model(kind, 'trained.pt')
            streams = [[{**r.record(), 'active_length': 12} for r in stream] for stream in evaluation_requests(713904)[:2]]
            result['saved_states'][kind+'-persistence'] = persistence(model, streams, a.output/(kind+'-persistence'))
            traces = [trace_pair(model, next(r for r in panel if r['family'] == family and r['active_length'] == length
                       and r['context'] == 'ungated' and r['layout'] == 'original')) for family in FAMILIES for length in (4, 12)]
            atomic_json(a.output/(kind+'-traces.json'), traces)
        print(json.dumps({'event': 'saved-state-map-complete', 'elapsed_seconds': time.monotonic()-started}), flush=True)
        conditions = [('feedback', 'initial.pt'), ('feedback', 'trained.pt'), ('control', 'trained.pt'), ('gru', 'initial.pt')]
        for kind, stage in conditions:
            for family in (('parity',) if a.fixture else FAMILIES):
                name = f'{kind}-{stage[:-3]}-{family}'
                print(json.dumps({'event': 'fit-start', 'condition': name}), flush=True)
                try:
                    fitted = fit(load_model(kind, stage), fits[family], [r for r in panel if r['family'] == family],
                                 a.output/name, steps=2 if a.fixture else 1500, seconds=120)
                except (ValueError, RuntimeError, AssertionError) as error:
                    fitted = {'status': 'condition_failed', 'type': type(error).__name__,
                              'message': str(error), 'fit_passed': False, 'positive_scc_result': False}
                    atomic_json(a.output/name/'failure.json', fitted)
                result['fits'][name] = fitted
                atomic_json(a.output/'progress.json', result)
                print(json.dumps({'event': 'fit-complete', 'condition': name,
                    'status': fitted['status'], 'steps': fitted.get('completed_steps'),
                    'accuracy': fitted.get('fit', {}).get('accuracy'),
                    'loss': fitted.get('fit', {}).get('loss'), 'fit_passed': fitted['fit_passed'],
                    'training_seconds': fitted.get('training_seconds')}), flush=True)
        assert source_manifest() == source and file_digest(__file__) == file_digest(a.output/'runner.py')
        assert all(file_digest(path) == h for path, h in parents.items())
        result.update(status='complete', elapsed_seconds=time.monotonic()-started, parents_unchanged=True,
            runner_sha256=file_digest(__file__), data_sha256=file_digest(a.output/'data.json'),
            environment={'python': sys.version, 'torch': str(torch.__version__), 'device': 'cpu', 'threads': 2})
        atomic_json(a.output/'result.json', result)
    except Exception as error:
        atomic_json(a.output/'failure.json', {'type': type(error).__name__, 'message': str(error),
                    'elapsed_seconds': time.monotonic()-started, 'positive_scc_result': False})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__': main()
