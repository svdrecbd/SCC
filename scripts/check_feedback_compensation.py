"""Saved-state equivariance and numerical drift diagnosis; not an SCC attack."""
import argparse
from collections import Counter
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
from scc.persistent_feedback import feedback_step
from scc.persistent_matrix import MatrixConfig
from scc.persistent_tasks import evaluation_requests, input_code
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scripts.localize_persistent_learning import independent_check

BASE = ROOT/'artifacts/scc-feedback-failure-20260913-v1/collected/persistent-feedback'
FITS = ROOT/'artifacts/scc-learning-localization-20260913-v1/full-v1'
PARENTS = {'initial': BASE/'initial.pt', 'trained': BASE/'trained.pt',
           'fitted_initial_lookup': FITS/'feedback-initial-lookup/fitted.pt',
           'fitted_trained_lookup': FITS/'feedback-trained-lookup/fitted.pt'}
WIRING = BASE/'fixed-wiring.pt'
SEED = 17313002
MODES = {'fp32-b2': (torch.float32, 2), 'fp32-b1': (torch.float32, 1),
         'fp64-b2': (torch.float64, 2)}


def transforms(dtype):
    eye = torch.eye(4, dtype=dtype)
    shear = eye.clone(); shear[3, 0] = .5
    return {'identity': eye, 'cycle': eye[[1, 2, 3, 0]], 'sign': -eye, 'shear': shear}


def edit(state, wiring, transform, compensated=True):
    output = transform @ state[..., :4, :]
    control = state[..., 4:, :]
    if compensated:
        control = control + wiring @ (state[..., :4, :] - output)
    return torch.cat((output, control), dim=-2)


def effective(state, wiring):
    return state[..., 4:, :] + wiring @ state[..., :4, :]


def residual(actual, expected):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    absolute = float((actual-expected).abs().max())
    scale = max(1., float(expected.abs().max()))
    return {'absolute': absolute, 'scaled': absolute/scale}


def step(state, inputs, config, wiring, batch):
    if batch == len(state):
        y, s, _ = feedback_step(state, inputs, config, wiring)
        return y, s
    parts = [feedback_step(state[i:i+batch], inputs[i:i+batch], config, wiring)[:2]
             for i in range(0, len(state), batch)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def local_probe(state, inputs, config, wiring, transform, compensated, batch):
    edited = edit(state, wiring, transform, compensated)
    y, following = step(state, inputs, config, wiring, batch)
    changed_y, changed = step(edited, inputs, config, wiring, batch)
    return {'initial_H': residual(effective(edited, wiring), effective(state, wiring)),
            'next_state': residual(changed, edit(following, wiring, transform, compensated)),
            'next_H': residual(effective(changed, wiring), effective(following, wiring)),
            'next_output': residual(changed_y, y @ transform.T)}


@torch.no_grad()
def trajectory(initial, ids, config, wiring, batch, *, reference=None,
               transform=None, compensated=True):
    code = input_code(config.input_size, 'anchor', dtype=initial.dtype)
    state = initial[None].expand(len(ids), -1, -1).clone()
    if transform is not None:
        state = edit(state, wiring, transform, compensated)
    snapshots = {0: state.clone()}
    boundaries = sorted(set((0, 1, min(4, ids.shape[1]), min(16, ids.shape[1]),
                             ids.shape[1]//2, ids.shape[1])))
    outputs, boundary_states, measurements = [], [], []
    for request in range(ids.shape[1]):
        for tick in range(ids.shape[2]):
            y, state = step(state, code[ids[:, request, tick]], config, wiring, batch)
        assert torch.isfinite(state).all() and torch.isfinite(y).all()
        outputs.append(y)
        if request+1 in boundaries:
            snapshots[request+1] = state.clone()
        if reference is None:
            boundary_states.append(state.clone())
        else:
            original = reference['states'][request]
            measurements.append({
                'request_boundary': request+1,
                'H': residual(effective(state, wiring), effective(original, wiring)),
                'state': residual(state, edit(original, wiring, transform, compensated)),
                'output': residual(y, reference['logits'][:, request] @ transform.T)})
    return {'logits': torch.stack(outputs, 1), 'final': state, 'snapshots': snapshots,
            'states': boundary_states, 'measurements': measurements}


def behavior(logits, reference, transform, rows):
    recovered = logits @ torch.linalg.inv(transform).T
    raw_predictions, predictions, original = logits.argmax(-1), recovered.argmax(-1), reference.argmax(-1)
    groups = {}
    for context in ('ungated', 'authorized', 'unauthorized'):
        selected = [(i,j,r) for i,stream in enumerate(rows) for j,r in enumerate(stream)
                    if r['context'] == context]
        groups[context] = {'n': len(selected)}
        for name, values in (('baseline', original), ('raw', raw_predictions), ('inverse_readout', predictions)):
            groups[context][name+'_correct'] = sum(int(values[i,j]) == r['label'] for i,j,r in selected)
        groups[context]['inverse_disagreements'] = sum(int(predictions[i,j]) != int(original[i,j]) for i,j,_ in selected)
    return {'inverse_disagreements': int((predictions != original).sum()),
            'raw_disagreements': int((raw_predictions != original).sum()),
            'inverse_logits': residual(recovered, reference), 'contexts': groups}


def summarize(measurements):
    return {name: {kind: max(m[name][kind] for m in measurements) for kind in ('absolute', 'scaled')}
            for name in ('H', 'state', 'output')}


def load_weights(name, path):
    saved = torch.load(path, weights_only=True, map_location='cpu')
    if name.startswith('fitted'):
        w = saved['model']['weights']
        assert torch.equal(saved['model']['wiring'], torch.load(WIRING, weights_only=True))
    else:
        w = saved['initial_weights'] if name == 'initial' else saved['weights']
    assert w.shape == (261, 128) and w.dtype == torch.float32
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    folder = args.output
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2)
    def timeout(*_):
        raise TimeoutError('Declared 900-second experiment limit reached')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(900)
    try:
        plan = (ROOT/'labnotes.md').read_text().split('<a id="ln-052"></a>')[1].split('\n<a id=')[0].split('\n## Supporting-record')[0]
        (folder/'plan.md').write_text('<a id="ln-052"></a>'+plan)
        manifest = snapshot_sources(folder/'source')
        extras = ['scripts/check_feedback_compensation.py', 'scripts/audit_feedback_compensation.py', 'scripts/localize_persistent_learning.py',
                  'tests/test_feedback_compensation.py']
        for name in extras:
            target = folder/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target); manifest[name] = file_digest(ROOT/name)
        atomic_json(folder/'source-manifest.json', manifest)
        parent_hashes = {str(p): file_digest(p) for p in [*PARENTS.values(), WIRING]}
        atomic_json(folder/'parents.json', parent_hashes)
        rows = [[r.record() for r in stream] for stream in evaluation_requests(SEED, per_cell=16, streams=2)]
        if args.fixture:
            rows = [stream[:2] for stream in rows]
        for stream in rows:
            for row in stream:
                independent_check(row); assert row['split'] == 'validation'
        atomic_json(folder/'requests.json', rows)
        ids = torch.tensor([[r['tokens'] for r in stream] for stream in rows])
        config = MatrixConfig(128, 4, 'soft')
        runtime = {'python': sys.version, 'torch': str(torch.__version__), 'platform': platform.platform(),
                   'machine': platform.machine(), 'threads': torch.get_num_threads()}
        atomic_json(folder/'configuration.json', {'seed': SEED, 'fixture': args.fixture,
                    'requests_per_stream': ids.shape[1], 'streams': 2, 'runtime': runtime,
                    'modes': list(MODES), 'parents': list(PARENTS),
                    'transforms': {k:v.tolist() for k,v in transforms(torch.float64).items()},
                    'wall_limit_seconds': 900, 'output_limit_bytes': 300*1024**2,
                    'absolute_historical_tolerance': 1e-4, 'fp64_local_tolerance': 1e-10})
        results, baseline_modes = [], {}
        names = list(PARENTS)[:1] if args.fixture else list(PARENTS)
        for name in names:
            weights = load_weights(name, PARENTS[name])
            for mode, (dtype, batch) in MODES.items():
                w = weights.to(dtype); wiring = torch.load(WIRING, weights_only=True).to(dtype)
                destination = folder/name/mode; destination.mkdir(parents=True)
                reference = trajectory(w, ids, config, wiring, batch)
                baseline_modes[name,mode] = reference['logits'].clone()
                torch.save({k:reference[k] for k in ('logits','final','snapshots')}, destination/'baseline.pt')
                for label, transform in transforms(dtype).items():
                    for compensate in ([True] if label == 'identity' else [False, True]):
                        key = label+('-compensated' if compensate else '-output-only')
                        changed = trajectory(w, ids, config, wiring, batch, reference=reference,
                                             transform=transform, compensated=compensate)
                        local = []
                        code = input_code(128, 'anchor', dtype=dtype)
                        for boundary, state in reference['snapshots'].items():
                            x = code[ids[:, boundary % ids.shape[1], 0]]
                            local.append({'boundary': boundary, **local_probe(state, x, config, wiring,
                                         transform, compensate, batch)})
                        detail = {'state': name, 'mode': mode, 'edit': key,
                                  'local': local, 'rollout_max': summarize(changed['measurements']),
                                  'measurements': changed['measurements'],
                                  'behavior': behavior(changed['logits'], reference['logits'], transform, rows)}
                        if label == 'identity':
                            assert torch.equal(changed['logits'], reference['logits'])
                            assert torch.equal(changed['final'], reference['final'])
                        torch.save({'logits': changed['logits'], 'final': changed['final']}, destination/(key+'.pt'))
                        atomic_json(destination/(key+'.json'), detail)
                        results.append({k:v for k,v in detail.items() if k != 'measurements'})
                print(json.dumps({'completed_state': name, 'mode': mode,
                                  'elapsed_seconds': time.monotonic()-start}), flush=True)
                if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) > 300*1024**2:
                    raise RuntimeError('Declared output ceiling exceeded')
        numerical = []
        for name in names:
            ref = baseline_modes[name,'fp32-b2'].double()
            for mode in ('fp32-b1','fp64-b2'):
                other = baseline_modes[name,mode].double()
                numerical.append({'state': name, 'mode': mode, 'logits': residual(other, ref),
                                  'decision_disagreements': int((other.argmax(-1) != ref.argmax(-1)).sum())})
        assert all(file_digest(p) == h for p,h in parent_hashes.items())
        assert all(file_digest(ROOT/p) == h and file_digest(folder/'source'/p) == h for p,h in manifest.items())
        atomic_json(folder/'result.json', {'status': 'complete', 'elapsed_seconds': time.monotonic()-start,
            'conditions': len(results), 'results': results, 'unedited_numerical_comparisons': numerical,
            'parents_unchanged': True, 'source_unchanged': True,
            'scope': 'Local structural and numerical diagnosis; no permission-removal or SCC qualification claim.'})
        atomic_json(folder/'output-hashes.json', {str(p.relative_to(folder)):file_digest(p)
                    for p in sorted(folder.rglob('*')) if p.is_file()})
        print(json.dumps({'status':'complete', 'conditions':len(results), 'elapsed_seconds':time.monotonic()-start}), flush=True)
    except BaseException as exc:
        atomic_json(folder/'failure.json', {'type':type(exc).__name__, 'message':str(exc),
                                         'elapsed_seconds':time.monotonic()-start})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
