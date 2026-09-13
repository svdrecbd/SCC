"""One declared ordinary-learning experiment with feedback into self-update controls."""
import argparse
from dataclasses import asdict
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
from scc.persistent_matrix import MatrixConfig
from scc.persistent_tasks import (training_requests, evaluation_requests, initialize_matrix,
    input_code, tensors, summarize_records, TOKENS_PER_REQUEST)
from scc.persistent_feedback import feedback_wiring, feedback_window, LiveFeedbackMatrix
from scc.provenance import atomic_json, canonical_json, digest, file_digest, snapshot_sources, source_manifest


def reference_window(initial, ids, config, code, wiring):
    """Independent explicit W @ T equations for the numerical gate, FP64 on CPU."""
    state = initial[None].expand(ids.shape[0], -1, -1)
    outputs = []
    width = config.input_size
    eye = torch.eye(width, dtype=state.dtype, device=state.device)
    for request in range(ids.shape[1]):
        for tick in range(ids.shape[2]):
            p = code[ids[:, request, tick]].softmax(-1)
            signals = torch.einsum('brd,bd->br', state, p)
            u = signals[:, 4:] + torch.einsum('bo,co->bc', signals[:, :4], wiring)
            key, query, beta = u[:, :width].softmax(-1), u[:, width:2*width].softmax(-1), u[:, -1].sigmoid()
            transform = eye + beta[:, None, None]*(query-key)[:, :, None]*key[:, None, :]
            state = state @ transform
            output = torch.einsum('bod,bd->bo', state[:, :4], p)
        outputs.append(output)
    return torch.stack(outputs, 1), state


def numerical_gate(initial, config, code, wiring, data_seed):
    rows = []
    before = initial.clone()
    for ordinal in (5999, 6000):
        ids, labels = tensors(training_requests(data_seed, ordinal, 2, 4), device=initial.device)
        candidate = initial.clone().requires_grad_(True)
        cpu = initial.detach().cpu().double().requires_grad_(True)
        out, state, _ = feedback_window(candidate, ids, config, code, wiring)
        ref, ref_state = reference_window(cpu, ids.cpu(), config, code.cpu().double(), wiring.cpu().double())
        F.cross_entropy(out.reshape(-1, 4), labels.reshape(-1)).backward()
        F.cross_entropy(ref.reshape(-1, 4), labels.cpu().reshape(-1)).backward()
        errors = {'output': float((out.detach().cpu()-ref.detach()).abs().max()),
                  'state': float((state.detach().cpu()-ref_state.detach()).abs().max()),
                  'gradient': float((candidate.grad.cpu()-cpu.grad).abs().max())}
        with torch.no_grad():
            first, middle, _ = feedback_window(initial, ids[:, :2], config, code, wiring)
            second, end, _ = feedback_window(middle, ids[:, 2:], config, code, wiring)
            errors['continuation_output'] = float((torch.cat((first, second), 1)-out).abs().max())
            errors['continuation_state'] = float((end-state).abs().max())
        if not all(torch.isfinite(torch.tensor(list(errors.values())))) or max(errors.values()) > 1e-4:
            raise ValueError('Independent FP64 feedback/continuation gate failed')
        full_ids, full_labels = tensors(training_requests(data_seed, ordinal, 32, 4), device=initial.device)
        disposable = torch.nn.Parameter(initial.clone())
        lr = .003 if ordinal < 6000 else .0003
        opt = torch.optim.Adam([disposable], lr=lr)
        prediction, _, _ = feedback_window(disposable, full_ids, config, code, wiring)
        loss = F.cross_entropy(prediction.reshape(-1, 4), full_labels.reshape(-1))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([disposable], 1., error_if_nonfinite=True)
        opt.step()
        if not torch.isfinite(disposable).all() or not torch.isfinite(loss) or norm <= 0:
            raise ValueError('Feedback full-shape update gate failed')
        rows.append({'ordinal': ordinal, 'maximum_errors': errors, 'learning_rate': lr,
                     'full_batch': 32, 'full_window': 4, 'loss': float(loss.detach()),
                     'gradient_norm': float(norm)})
    assert torch.equal(before, initial)
    return {'passed': True, 'checks': rows, 'training_initial_unchanged': True,
            'scope': 'Implementation gate only; fixed strength 1'}


@torch.no_grad()
def evaluate(weights, config, code, wiring, folder, *, per_cell, streams):
    folder.mkdir()
    requests = evaluation_requests(713904, per_cell=per_cell, streams=streams)
    ids, labels = tensors(requests, device=weights.device)
    length = ids.shape[1]
    modes = {}
    for name, interval in [('continuous', length), ('reset-1', 1), ('reset-4', 4)]:
        # Reset modes deliberately supply a clean matrix, solely as diagnostics.
        pieces = []
        for start in range(0, length, interval):
            outputs, states, _ = feedback_window(weights, ids[:, start:start+interval], config, code, wiring)
            if not torch.isfinite(outputs).all() or not torch.isfinite(states).all():
                raise ValueError('Nonfinite evaluation')
            pieces.append(outputs)
        outputs = torch.cat(pieces, 1)
        predictions, scores = outputs.argmax(-1).cpu().tolist(), outputs.cpu().tolist()
        records = [{**requests[s][i].record(), 'stream': s, 'request_index': i,
                    'late_half': i >= length//2, 'prediction': predictions[s][i], 'logits': scores[s][i]}
                   for i in range(length) for s in range(streams)]
        path = folder/(name+'.jsonl')
        with path.open('w') as log:
            for row in records: log.write(canonical_json(row)+'\n')
        summary = {**summarize_records(records), 'prediction_sha256': file_digest(path),
                   'clean_resets_within_stream': (length-1)//interval,
                   'reset_interval': interval, 'streams': streams,
                   'consecutive_requests_per_stream': length}
        if name == 'continuous':
            live = LiveFeedbackMatrix(weights, config, wiring)
            error, same = 0., True
            for i, request in enumerate(requests[0]):
                for token in request.tokens: y, _ = live.tick(code[token])
                error = max(error, float((y-outputs[0, i]).abs().max()))
                same &= int(y.argmax()) == predictions[0][i]
            if not same or error > 1e-4:
                raise ValueError('Batched and single live feedback execution differ')
            summary.update(single_stream_decisions_match=same, single_stream_maximum_logit_error=error)
            torch.save({'current_weights': states.cpu(), 'configuration': asdict(config),
                        'ticks_per_stream': length*TOKENS_PER_REQUEST,
                        'fixed_wiring': wiring.cpu()}, folder/'final-live-states.pt')
        modes[name] = summary
    atomic_json(folder/'result.json', modes)
    return modes


def declared(a):
    return (not a.fixture and a.device == 'cuda' and a.width == 128 and a.seed == 17
            and a.data_seed == 24017 and a.steps == 12000 and a.batch == 32
            and a.window == 4 and a.eval_per_cell == 128 and a.eval_streams == 16
            and a.wall_seconds == 6600 and a.total_seconds == 6900)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, default=ROOT/'experiment.md')
    p.add_argument('--initial-conditions', type=Path, default=ROOT/'initial-conditions.json')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--data-seed', type=int, default=24017)
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--window', type=int, default=4)
    p.add_argument('--eval-per-cell', type=int, default=128)
    p.add_argument('--eval-streams', type=int, default=16)
    p.add_argument('--wall-seconds', type=float, default=6600)
    p.add_argument('--total-seconds', type=int, default=6900)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--fixture', action='store_true')
    p.add_argument('--data', type=Path)
    p.add_argument('--parents', type=Path)
    a = p.parse_args()
    if (min(a.width, a.steps, a.batch, a.window, a.threads) < 1 or
            not 0 < a.wall_seconds < a.total_seconds or not a.plan.is_file()
            or not a.initial_conditions.is_file()):
        raise ValueError('Invalid configuration or missing frozen plan')
    if a.parents and json.loads(a.parents.read_text()) != {}:
        raise ValueError('Parent initialization not allowed')
    a.output.mkdir(parents=True, exist_ok=False)
    source = snapshot_sources(a.output/'source')
    shutil.copyfile(__file__, a.output/'runner.py')
    shutil.copyfile(a.plan, a.output/'protocol.md')
    shutil.copyfile(a.initial_conditions, a.output/'initial-conditions.json')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    if a.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = MatrixConfig(a.width, 4, 'soft')
    frozen_values = json.loads(a.initial_conditions.read_text())
    if (frozen_values['schema'] != 'persistent-feedback-initial-conditions/v1'
            or frozen_values['configuration'] != asdict(config)
            or frozen_values['initial_seed'] != a.seed or frozen_values['feedback_seed'] != 130913
            or frozen_values['feedback_strength'] != 1.):
        raise ValueError('Initial conditions disagree with the declared configuration')
    initial = torch.tensor(frozen_values['initial_weights'], dtype=torch.float32, device=a.device)
    code = input_code(a.width, 'anchor', device=a.device)
    wiring = torch.tensor(frozen_values['fixed_wiring'], dtype=torch.float32, device=a.device)
    if (initial.shape != (config.rows,config.input_size)
            or wiring.shape != (2*config.input_size+1,4)
            or not torch.isfinite(initial).all() or not torch.isfinite(wiring).all()):
        raise ValueError('Invalid frozen numerical initial conditions')
    weights = torch.nn.Parameter(initial.clone())
    opt = torch.optim.Adam([weights], lr=.003)
    torch.save({'initial_weights': initial.cpu(), 'configuration': asdict(config)}, a.output/'initial.pt')
    torch.save(wiring.cpu(), a.output/'fixed-wiring.pt')
    completed, chain, clock = 0, digest('persistent-feedback/v1'), time.monotonic()
    def deadline(signum, frame): raise TimeoutError('Declared process deadline reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(a.total_seconds)
    try:
        expected = None
        if a.device == 'cuda':
            expected = json.loads((ROOT/'transport-source-manifest.json').read_text())
            if not all(file_digest(ROOT/n) == h for n,h in expected.items()):
                raise ValueError('Frozen transport source mismatch')
        gate = numerical_gate(initial, config, code, wiring, a.data_seed)
        assert torch.equal(initial, weights.detach())
        atomic_json(a.output/'implementation-gate.json', gate)
        with (a.output/'training.jsonl').open('w', buffering=1) as log:
            for ordinal in range(a.steps):
                if time.monotonic()-clock > a.wall_seconds: break
                lr = .003 if ordinal < 6000 else .0003
                for group in opt.param_groups: group['lr'] = lr
                requests = training_requests(a.data_seed, ordinal, a.batch, a.window)
                ids, labels = tensors(requests, device=a.device)
                opt.zero_grad(set_to_none=True)
                output, state, control = feedback_window(weights, ids, config, code, wiring)
                losses = F.cross_entropy(output.reshape(-1, 4), labels.reshape(-1), reduction='none').reshape_as(labels)
                loss = losses.mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_([weights], 1., error_if_nonfinite=True)
                opt.step()
                if not torch.isfinite(weights).all(): raise ValueError('Nonfinite initial state')
                completed = ordinal+1
                record = {'step': completed, 'loss': float(loss.detach()),
                          'accuracy': float((output.argmax(-1)==labels).float().mean()),
                          'gradient_norm_before_clip': float(norm), 'learning_rate': lr,
                          'loss_by_request': losses.detach().mean(0).tolist(),
                          'sample_sha256': digest([[r.record() for r in stream] for stream in requests]),
                          'elapsed_seconds': time.monotonic()-clock}
                chain = digest({'previous': chain, 'record': record})
                log.write(canonical_json({**record, 'chain': chain})+'\n')
                if completed == 1 or completed % 100 == 0:
                    print(json.dumps({k:record[k] for k in ('step','loss','accuracy','elapsed_seconds')}),flush=True)
                if completed in (2000, 4000, 6000, 9000):
                    torch.save({'weights':weights.detach().cpu(),'optimizer':opt.state_dict(),
                                'step':completed,'configuration':asdict(config)}, a.output/f'stage-{completed}.pt')
        training_seconds = time.monotonic()-clock
        torch.save({'weights':weights.detach().cpu(),'optimizer':opt.state_dict(),
                    'step':completed,'configuration':asdict(config)},a.output/'trained.pt')
        evaluation = evaluate(weights.detach(),config,code,wiring,a.output/'evaluation',
                              per_cell=a.eval_per_cell,streams=a.eval_streams)
        complete = completed==a.steps
        result = {'schema':'persistent-feedback/v1','status':'complete' if complete else 'training_wall_limit',
                  'completed_steps':completed,'requested_steps':a.steps,'training_complete':complete,
                  'declared_screen_configuration':declared(a),
                  'qualified_learnability':complete and declared(a) and evaluation['continuous']['qualified'],
                  'evaluation':evaluation,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  'parameter_count':weights.numel(),'fixed_wiring_values':wiring.numel(),
                  'fixed_wiring_sha256':file_digest(a.output/'fixed-wiring.pt'),
                  'initial_conditions_sha256':file_digest(a.output/'initial-conditions.json'),
                  'feedback_strength':1.,'feedback_seed':130913,
                  'training_seconds':training_seconds,'elapsed_seconds':time.monotonic()-clock,
                  'source_sha256':digest(source),'runner_sha256':file_digest(a.output/'runner.py'),
                  'protocol_sha256':file_digest(a.output/'protocol.md'),
                  'training_log_sha256':file_digest(a.output/'training.jsonl'),'training_chain':chain,
                  'positive_scc_result':False,'coupling_trained':False,'protection_removal_tested':False,
                  'evidence_class':'Open ordinary-learning construction screen; no SCC demonstration',
                  'environment':{'python':platform.python_version(),'torch':str(torch.__version__),
                                 'device':a.device,'gpu':torch.cuda.get_device_name() if a.device=='cuda' else None,
                                 'peak_cuda_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None}}
        assert source_manifest()==source
        if expected: assert all(file_digest(ROOT/n)==h for n,h in expected.items())
        atomic_json(a.output/'result.json',result)
        inline={k:result[k] for k in ('completed_steps','qualified_learnability','declared_screen_configuration','positive_scc_result')}
        inline.update(ok=complete,scientific_status=result['status'])
        if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],inline)
        print(json.dumps(inline),flush=True)
    except Exception as error:
        signal.alarm(0)
        atomic_json(a.output/'failure.json',{'type':type(error).__name__,'message':str(error),
                    'completed_steps':completed,'elapsed_seconds':time.monotonic()-clock,'positive_scc_result':False})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':main()
