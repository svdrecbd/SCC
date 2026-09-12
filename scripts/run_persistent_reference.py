"""One ordinary GRU reference on the frozen persistent-task interface."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F

from scc.persistent_reference import PersistentGRU
from scc.persistent_tasks import (training_requests, evaluation_requests, tensors,
                                  summarize_records, FAMILIES, CONTEXTS)
from scc.provenance import (atomic_json, canonical_json, digest, file_digest,
                            snapshot_sources, source_manifest)

PROTOCOL = ROOT/'protocols/SCC_PERSISTENT_REFERENCE_V1.md'


def is_declared(a):
    return (not (getattr(a, 'stabilize', False) and not getattr(a, 'extension', False))
            and a.device == 'cpu' and a.width == 128 and a.seed == 17
            and a.data_seed == 24017 and a.steps == (12000 if getattr(a, 'extension', False) else 6000) and a.batch == 32
            and a.window == 4 and a.lr == .003 and a.eval_per_cell == 128
            and a.eval_streams == 16 and a.wall_seconds == 540
            and a.total_seconds == 600 and not a.fixture)


def verify_transport():
    expected = json.loads((ROOT/'transport-source-manifest.json').read_text())
    for name, sha in expected.items():
        if file_digest(ROOT/name) != sha:
            raise ValueError('Transport source digest mismatch: '+name)
    expected_source = {n: s for n, s in expected.items()
                       if n.startswith('scc/') or n in ('pyproject.toml', 'uv.lock', '.python-version')}
    if source_manifest() != expected_source:
        raise ValueError('Transport source set mismatch')
    return {'files_verified': len(expected), 'manifest_sha256': digest(expected)}


def numerical_gate(model, data_seed):
    """No training model updates: gate uses independent copies of its parameters."""
    device = next(model.parameters()).device
    cpu = copy.deepcopy(model).cpu().double()
    gpu = copy.deepcopy(model)
    rows = []
    for ordinal in (0, 2001):
        batch = training_requests(data_seed, ordinal, 2, 4)
        cpu_ids, cpu_labels = tensors(batch, device='cpu')
        ids, labels = cpu_ids.to(device), cpu_labels.to(device)
        cpu.zero_grad(set_to_none=True); gpu.zero_grad(set_to_none=True)
        cpu_y, cpu_h = cpu(cpu_ids)
        y, h = gpu(ids)
        F.cross_entropy(cpu_y.reshape(-1, 4), cpu_labels.reshape(-1)).backward()
        F.cross_entropy(y.reshape(-1, 4), labels.reshape(-1)).backward()
        forward_error = float((cpu_y.detach()-y.detach().cpu()).abs().max())
        state_error = float((cpu_h.detach()-h.detach().cpu()).abs().max())
        gradient_error = max(float((p.grad-q.grad.cpu()).abs().max())
                             for p, q in zip(cpu.parameters(), gpu.parameters()))
        with torch.no_grad():
            first, middle = gpu(ids[:, :2])
            second, end = gpu(ids[:, 2:], middle)
            continuation_error = float((torch.cat((first, second), 1)-y).abs().max())
            continuation_state_error = float((end-h).abs().max())
        errors = [forward_error, state_error, gradient_error,
                  continuation_error, continuation_state_error]
        if not all(torch.isfinite(torch.tensor(errors))) or max(errors) > 1e-4:
            raise ValueError('Float64 reference gradient or continuation gate failed')
        rows.append({'ordinal': ordinal, 'forward_error': forward_error,
                     'state_error': state_error, 'gradient_error': gradient_error,
                     'continuation_error': continuation_error,
                     'continuation_state_error': continuation_state_error})
    optimizer = torch.optim.Adam(gpu.parameters(), lr=.003)
    update_checks = []
    for ordinal in (0, 2001):
        ids, labels = tensors(training_requests(data_seed, ordinal, 32, 4), device=device)
        optimizer.zero_grad(set_to_none=True)
        y, _ = gpu(ids)
        loss = F.cross_entropy(y.reshape(-1, 4), labels.reshape(-1))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(gpu.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (not torch.isfinite(loss) or norm <= 0
                or not all(torch.isfinite(p).all() for p in gpu.parameters())):
            raise ValueError('Nonfinite or inactive full-shape update')
        update_checks.append({'ordinal': ordinal, 'loss': float(loss.detach()),
                              'gradient_norm': float(norm), 'batch': 32, 'window': 4})
    return {'passed': True, 'device': str(device), 'float64_cpu_comparisons': rows,
            'full_shape_updates': update_checks, 'training_model_changed': False,
            'evidence_class': 'Implementation validation only'}


@torch.no_grad()
def evaluate(model, folder, *, per_cell, streams):
    folder.mkdir()
    model.eval()
    device = next(model.parameters()).device
    requests = evaluation_requests(713904, per_cell=per_cell, streams=streams)
    ids, _ = tensors(requests, device=device)
    length = ids.shape[1]
    summaries = {}
    for name, reset_interval in (('continuous', length), ('reset-1', 1), ('reset-4', 4)):
        pieces = []
        for start in range(0, length, reset_interval):
            # Intentional resets in diagnostics only; continuous executes once.
            y, state = model(ids[:, start:start+reset_interval])
            if not torch.isfinite(y).all() or not torch.isfinite(state).all():
                raise ValueError('Nonfinite evaluation')
            pieces.append(y)
        outputs = torch.cat(pieces, 1)
        scores = outputs.cpu().tolist()
        predictions = outputs.argmax(-1).cpu().tolist()
        records = []
        for step in range(length):
            for stream in range(streams):
                records.append({**requests[stream][step].record(), 'stream': stream,
                                'request_index': step, 'late_half': step >= length//2,
                                'prediction': predictions[stream][step],
                                'logits': scores[stream][step]})
        path = folder/(name+'.jsonl')
        with path.open('w') as log:
            for r in records:
                log.write(canonical_json(r)+'\n')
        torch.save(state.cpu(), folder/(name+'-final-state.pt'))
        result = {**summarize_records(records), 'reset_interval': reset_interval,
                  'clean_resets_within_stream': (length-1)//reset_interval,
                  'prediction_sha256': file_digest(path), 'per_cell': per_cell,
                  'streams': streams, 'consecutive_requests_per_stream': length}
        if name == 'continuous':
            h = torch.zeros(1, model.width, device=device)
            maximum_error, same = 0., True
            for step, r in enumerate(requests[0]):
                for token in r.tokens:
                    y, h = model.manual_tick(torch.tensor([token], device=device), h)
                maximum_error = max(maximum_error, float((y[0]-outputs[0, step]).abs().max()))
                same &= int(y[0].argmax()) == predictions[0][step]
            if not same or maximum_error > 1e-4:
                raise ValueError('Explicit GRU and fused stream disagree')
            result.update(manual_decisions_match=same, manual_maximum_logit_error=maximum_error)
        summaries[name] = result
    atomic_json(folder/'summary.json', summaries)
    model.train()
    return summaries


def grouped_scores(requests, losses, predictions, labels):
    values, predicted, targets = losses.detach().cpu(), predictions.cpu(), labels.cpu()
    groups = {f+'/'+c: {'count': 0, 'correct': 0, 'loss_sum': 0.}
              for f in FAMILIES for c in CONTEXTS}
    for b, stream in enumerate(requests):
        for k, r in enumerate(stream):
            group = groups[r.family+'/'+r.context]
            group['count'] += 1
            group['correct'] += int(predicted[b, k] == targets[b, k])
            group['loss_sum'] += float(values[b, k])
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--data-seed', type=int, default=24017)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--window', type=int, default=4)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--wall-seconds', type=float, default=540)
    p.add_argument('--total-seconds', type=int, default=600)
    p.add_argument('--eval-per-cell', type=int, default=128)
    p.add_argument('--eval-streams', type=int, default=16)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--fixture', action='store_true')
    p.add_argument('--extension', action='store_true', help='Separate predeclared 12,000-update follow-up')
    p.add_argument('--stabilize', action='store_true', help='With extension, reduce LR to .0003 after update 6000')
    p.add_argument('--data', type=Path)
    p.add_argument('--parents', type=Path)
    a = p.parse_args()
    if (min(a.width, a.steps, a.batch, a.window, a.threads) < 1 or a.lr <= 0
            or not 0 < a.wall_seconds < a.total_seconds):
        raise ValueError('Invalid resource configuration')
    a.output.mkdir(parents=True, exist_ok=False)
    source = snapshot_sources(a.output/'source')
    shutil.copyfile(__file__, a.output/'runner.py')
    if a.stabilize and not a.extension:
        raise ValueError('Stabilization requires the extension recipe')
    protocol = (ROOT/'protocols/SCC_PERSISTENT_REFERENCE_STABILIZATION_V1.md' if a.stabilize else
                ROOT/'protocols/SCC_PERSISTENT_REFERENCE_EXTENSION_V1.md' if a.extension else PROTOCOL)
    shutil.copyfile(protocol, a.output/'protocol.md')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = PersistentGRU(a.width).to(a.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    clock = time.monotonic()
    completed = 0
    chain = digest('persistent-reference/v1')

    def deadline(signum, frame):
        raise TimeoutError('Declared whole-process deadline reached')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(a.total_seconds)

    def save_stage(name, *, with_optimizer):
        payload = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   'step': completed, 'width': a.width}
        if with_optimizer:
            payload['optimizer'] = optimizer.state_dict()
        torch.save(payload, a.output/(name+'.pt'))

    try:
        if a.parents and json.loads(a.parents.read_text()) != {}:
            raise ValueError('Reference must not load parents')
        before_gate = digest({n: p.detach().cpu().tolist() for n, p in model.named_parameters()})
        transport = None if a.fixture else verify_transport()
        gate = numerical_gate(model, a.data_seed)
        if before_gate != digest({n: p.detach().cpu().tolist() for n, p in model.named_parameters()}):
            raise ValueError('Gate changed training initialization')
        atomic_json(a.output/'implementation-gate.json', gate)
        save_stage('initial', with_optimizer=False)
        evaluations = {'initial': evaluate(model, a.output/'eval-initial',
                                          per_cell=a.eval_per_cell, streams=a.eval_streams)}
        training_seconds = 0.
        with (a.output/'training.jsonl').open('w', buffering=1) as log:
            for step in range(a.steps):
                if time.monotonic()-clock > a.wall_seconds:
                    break
                update_start = time.monotonic()
                if a.stabilize and step == 6000:
                    for group in optimizer.param_groups:
                        group['lr'] = .0003
                requests = training_requests(a.data_seed, step, a.batch, a.window)
                ids, labels = tensors(requests, device=a.device)
                optimizer.zero_grad(set_to_none=True)
                outputs, _ = model(ids)
                losses = F.cross_entropy(outputs.reshape(-1, 4), labels.reshape(-1),
                                         reduction='none').reshape_as(labels)
                loss = losses.mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError('Nonfinite learned parameters')
                completed = step+1
                prediction = outputs.detach().argmax(-1)
                record = {'step': completed, 'loss': float(loss.detach()),
                          'learning_rate': optimizer.param_groups[0]['lr'],
                          'accuracy': float((prediction == labels).float().mean()),
                          'gradient_norm_before_clip': float(norm),
                          'loss_by_request': losses.detach().mean(0).tolist(),
                          'groups': grouped_scores(requests, losses, prediction, labels),
                          'sample_sha256': digest([[r.record() for r in stream] for stream in requests]),
                          'elapsed_seconds': time.monotonic()-clock}
                chain = digest({'previous': chain, 'record': record})
                log.write(canonical_json({**record, 'chain': chain})+'\n')
                training_seconds += time.monotonic()-update_start
                if completed == 1 or completed % 100 == 0:
                    print(json.dumps({k: record[k] for k in ('step', 'loss', 'accuracy', 'elapsed_seconds')}), flush=True)
                if completed in ((2000, 4000, 6000, 9000) if a.extension else (2000, 4000)):
                    name = 'stage-'+str(completed)
                    save_stage(name, with_optimizer=True)
                    evaluations[name] = evaluate(model, a.output/('eval-'+name),
                                                 per_cell=a.eval_per_cell, streams=a.eval_streams)
        save_stage('trained', with_optimizer=True)
        evaluations['final'] = evaluate(model, a.output/'eval-final',
                                        per_cell=a.eval_per_cell, streams=a.eval_streams)
        complete, declared = completed == a.steps, is_declared(a)
        if source_manifest() != source:
            raise ValueError('Imported source changed during run')
        if transport:
            verify_transport()
        result = {'schema': 'persistent-reference/v1',
                  'status': 'complete' if complete else 'training_wall_limit',
                  'training_complete': complete, 'completed_steps': completed,
                  'declared_screen_configuration': declared,
                  'qualified_learnability': bool(declared and complete and gate['passed']
                                                 and evaluations['final']['continuous']['qualified']),
                  'evaluations': evaluations, 'training_seconds': training_seconds,
                  'elapsed_seconds': time.monotonic()-clock, 'parameter_count': sum(p.numel() for p in model.parameters()),
                  'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  'source_sha256': digest(source), 'runner_sha256': file_digest(a.output/'runner.py'),
                  'protocol_sha256': file_digest(a.output/'protocol.md'), 'training_chain': chain,
                  'training_log_sha256': file_digest(a.output/'training.jsonl'), 'transport': transport,
                  'positive_scc_result': False, 'coupling_trained': False, 'protection_removal_tested': False,
                  'evidence_class': 'Ordinary recurrent learning reference; open calibration only',
                  'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                  'device': a.device, 'gpu': torch.cuda.get_device_name() if a.device == 'cuda' else None,
                                  'peak_cuda_bytes': torch.cuda.max_memory_allocated() if a.device == 'cuda' else None}}
        atomic_json(a.output/'files.json', {str(f.relative_to(a.output)): file_digest(f)
                                          for f in sorted(a.output.rglob('*')) if f.is_file()})
        atomic_json(a.output/'result.json', result)
        inline = {k: result[k] for k in ('status', 'training_complete', 'completed_steps',
                                        'declared_screen_configuration', 'qualified_learnability', 'positive_scc_result')}
        inline['ok'] = complete
        if os.environ.get('GMN_RESULT_PATH'):
            atomic_json(os.environ['GMN_RESULT_PATH'], inline)
        print(json.dumps(inline), flush=True)
    except Exception as error:
        signal.alarm(0)
        save_stage('failed-stage', with_optimizer=True)
        atomic_json(a.output/'failure.json', {'type': type(error).__name__, 'message': str(error),
                                             'completed_steps': completed, 'elapsed_seconds': time.monotonic()-clock,
                                             'positive_scc_result': False})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
