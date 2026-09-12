"""Train global coordinates and challenge both them and materialized weights."""
import argparse
import collections
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scc.checkpoint import save_checkpoint, load_checkpoint, rng_state
from scc.coupling import nll
from scc.developmental_run import TextBank, Streams, arithmetic_limits, configure, environment, batch_fingerprint
from scc.developmental_tasks import FAMILIES
from scc.coordinate_models import CoordinateModel, coordinate_config, from_checkpoint, DIMENSIONS
from scc.portfolio_objective import make_contraction_episode, latent_collapse_objective
from scc.provenance import atomic_json, digest, file_digest, snapshot_sources, source_manifest
from scc.transition_evaluation import measure
from scc.transition_path import EditStepper, Events, panels, point

PROTOCOL = 'SCC_COORDINATE_CONSTRUCTION_V1.md'


def configuration(dimension, mode, seed, smoke=False):
    return {'architecture': 'scc-coordinates/v1', 'model': asdict(coordinate_config(dimension, smoke)),
            'dimension': dimension, 'mode': mode, 'seed': seed, 'data_seed': 101,
            'steps': 4 if smoke else 20000, 'batch_size': 2 if smoke else 64,
            'warmup': 2 if smoke else 200, 'lr': .0006, 'decay_start': 9000,
            'coupling_frequency': 2 if smoke else 25, 'coupling_weight': 1.0,
            'inner_steps': 2 if smoke else 8, 'inner_batch_size': 2,
            'evaluation_size': 4 if smoke else 128, 'evaluation_seed': 582019,
            'text_blocks': 2 if smoke else 128, 'edit_steps': 4 if smoke else 500,
            'edit_batch_size': 2 if smoke else 16, 'smoke': smoke,
            'evidence_class': 'CPU implementation fixture' if smoke else 'Open affine-coordinate construction screen; no sealed test'}


def save_model(path, model, config, **extra):
    save_checkpoint(path, {'schema_version': 1, 'configuration': config, 'model': model.state_dict(), **extra})


def fingerprint(model):
    import hashlib
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(bank, config, folder, device, deadline):
    folder.mkdir()
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    model = CoordinateModel(coordinate_config(config['dimension'], config['smoke'])).to(device).train()
    fixed = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
    map_record = model.map_record()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(.9, .95), eps=1e-8, weight_decay=0., foreach=False)
    stream = Streams(bank, config['data_seed'], device, config['batch_size'], .5)
    ordinary_chain = digest('portfolio-ordinary/v1')
    meta_chain = digest('portfolio-contraction/v1')
    completed, episodes = 0, 0
    started = time.monotonic()
    with (folder / 'steps.jsonl').open('w') as log:
        for step in range(config['steps']):
            if time.monotonic() >= deadline:
                break
            stream.arithmetic_limits = arithmetic_limits(step, True)
            batch = stream.ordinary()
            ordinary_chain = digest([ordinary_chain, batch_fingerprint(batch)])
            progress = max(0., (step - config['decay_start']) / max(1, config['steps'] - 1 - config['decay_start']))
            lr = config['lr'] * min(1., (step + 1) / config['warmup']) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            base = nll(model, dict(model.named_parameters()), batch)
            meta, meta_norm = None, None
            loss = base
            if config['mode'] == 'contraction' and step % config['coupling_frequency'] == 0:
                episode = make_contraction_episode(bank, step // config['coupling_frequency'], device,
                    batch_size=config['inner_batch_size'], inner_steps=config['inner_steps'])
                value, meta = latent_collapse_objective(model, episode)
                grads = torch.autograd.grad(value, tuple(model.parameters()), retain_graph=True, allow_unused=True)
                meta_norm = torch.stack([g.detach().square().sum() for g in grads if g is not None]).sum().sqrt().item()
                loss = base + config['coupling_weight'] * value
                meta_chain = digest([meta_chain, episode['record']['batch_sha256']])
                episodes += 1
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            completed = step + 1
            record = {'step': completed, 'ordinary_loss': float(base.detach()), 'loss': float(loss.detach()),
                      'gradient_norm': float(norm), 'lr': lr, 'role': batch.role,
                      'arithmetic_limits': list(stream.arithmetic_limits), 'ordinary_batch_sha256': batch_fingerprint(batch),
                      'ordinary_chain': ordinary_chain, 'meta_chain': meta_chain,
                      'meta_gradient_norm': meta_norm, 'meta': meta}
            log.write(json.dumps(record) + '\n')
            log.flush()
            if completed in (9000, 15000):
                save_model(folder / f'stage-{completed}.pt', model, config, completed_steps=completed)
            if completed % 1000 == 0:
                print(json.dumps({'phase': 'training', 'step': completed, 'meta_episodes': episodes,
                                  'ordinary_loss': record['ordinary_loss'], 'elapsed_seconds': time.monotonic() - started}), flush=True)
    assert all(torch.equal(fixed[n], b.detach().cpu()) for n, b in model.named_buffers())
    save_model(folder / 'final.pt', model, config, completed_steps=completed,
               optimizer=optimizer.state_dict(), rng=rng_state(device), stream=stream.state_dict(),
               ordinary_chain=ordinary_chain, meta_chain=meta_chain, resume_implementation_validated=False)
    return model.eval(), {'completed_steps': completed, 'complete': completed == config['steps'],
                         'ordinary_chain': ordinary_chain, 'meta_chain': meta_chain, 'meta_episodes': episodes,
                         'parameter_count': model.parameter_count(), 'coordinate_map': map_record, 'fixed_buffers_unchanged': True, 'elapsed_seconds': time.monotonic() - started,
                         'steps_sha256': file_digest(folder / 'steps.jsonl')}


def probe(parent, bank, config, folder, scope, benign, deadline):
    folder.mkdir()
    model = copy.deepcopy(parent).train()
    stream = EditStepper(model, bank, batch_size=config['edit_batch_size'], scope=scope, benign=benign)
    panel = panels(2 if config['smoke'] else 32, 3 if config['smoke'] else 64)
    atomic_json(folder / 'panels.json', panel)
    events = Events()
    ring = collections.OrderedDict()
    fixed = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
    saved = {}
    minimum = 1.
    reference = None
    completed = 0
    def keep(step, reason):
        if step < 0 or step in saved:
            return
        state = ring[step]
        path = folder / 'landmark-weights' / f'step-{step:04d}.pt'
        save_checkpoint(path, {'schema_version': 1, 'configuration': config, 'model': {**fixed, **state},
                              'step': step, 'reason': reason, 'optimizer_resume_supported': False})
        saved[step] = {'path': str(path.relative_to(folder)), 'sha256': file_digest(path), 'reason': reason}
        assert len(saved) <= 6
    with (folder / 'curve.jsonl').open('w') as log, (folder / 'edit-steps.jsonl').open('w') as edits:
        for step in range(config['edit_steps'] + 1):
            if time.monotonic() >= deadline:
                break
            if step:
                update = stream.advance()
                edits.write(json.dumps(update) + '\n')
                edits.flush()
            row = point(model, bank, panel, step, reference, 2 if config['smoke'] else 16)
            if reference is None:
                reference = copy.deepcopy(row)
            ring[step] = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
            while len(ring) > 4:
                ring.popitem(last=False)
            new = events.observe(row)
            for name, at in new.items():
                if name == 'all_domains_low_screen' or (name == 'sustained_reliable_violation' and not benign):
                    keep(max(0, at - 1), name)
                    keep(at, name)
            row['new_events'] = new
            log.write(json.dumps(row) + '\n')
            log.flush()
            minimum = min(minimum, row['minimum_benign_strict'])
            completed = step
            if step % 100 == 0 or new:
                print(json.dumps({'phase': 'probe', 'probe': folder.name, 'step': step,
                                  'benign_min': row['minimum_benign_strict'], 'events': new}), flush=True)
    assert all(torch.equal(p, dict(parent.named_parameters())[n]) for n, p in model.named_parameters() if n not in stream.names)
    save_model(folder / 'modified.pt', model, config, modification_steps=completed, edit_scope=scope)
    result = {'scope': scope, 'benign': benign, 'completed_modification_steps': completed,
              'events': events.record(), 'minimum_benign_over_path': minimum, 'saved_landmarks': saved,
              'noneditable_unchanged': True, 'curve_sha256': file_digest(folder / 'curve.jsonl'),
              'modification_chain': stream.chain, 'measurements': {}, 'completed_repair_steps': 0}
    result['measurements']['modified'] = measure(model, bank, config, folder / 'modified')
    atomic_json(folder / 'progress.json', result)
    for step, record in sorted(saved.items()):
        if time.monotonic() >= deadline:
            break
        observed = from_checkpoint(load_checkpoint(folder / record['path'])).to(next(model.parameters()).device).eval()
        result['measurements'][f'landmark-{step}'] = measure(observed, bank, config, folder / f'landmark-{step}')
        del observed
    # Match the previous fresh-moment phase while retaining its disjoint query stream.
    stream.stream.begin_query()
    stream.step, stream.phase = 0, 'repair'
    stream.optimizer = torch.optim.AdamW(stream.params, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=0., foreach=False)
    with (folder / 'repair-steps.jsonl').open('w') as log:
        for step in range(config['edit_steps']):
            if time.monotonic() >= deadline:
                break
            record = stream.advance()
            log.write(json.dumps(record) + '\n')
            log.flush()
            result['completed_repair_steps'] = step + 1
    assert all(torch.equal(p, dict(parent.named_parameters())[n]) for n, p in model.named_parameters() if n not in stream.names)
    save_model(folder / 'repaired.pt', model, config, edit_scope=scope, repair_steps=result['completed_repair_steps'])
    result['measurements']['repaired'] = measure(model, bank, config, folder / 'repaired')
    result['status'] = 'complete' if completed == config['edit_steps'] and result['completed_repair_steps'] == config['edit_steps'] and len(result['measurements']) == len(saved) + 2 else 'incomplete'
    assert all(torch.equal(fixed[n], b.detach().cpu()) for n, b in model.named_buffers())
    result['fixed_buffers_unchanged'] = True
    result['final_chain'] = stream.chain
    atomic_json(folder / 'result.json', result)
    return result


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = snapshot_sources(a.output / 'source')
    shutil.copyfile(__file__, a.output / 'runner.py')
    shutil.copyfile(ROOT / 'protocols' / PROTOCOL, a.output / 'protocol.md')
    assert json.loads(a.parents.read_text()) == {}, 'This screen trains from scratch; no parent downloads needed'
    device = configure(a.device, 2 if a.device == 'cpu' else 4)
    config = configuration(a.dimension, a.mode, a.seed, a.smoke)
    atomic_json(a.output / 'configuration.json', config)
    bank = TextBank(a.data, config['text_blocks'])
    atomic_json(a.output / 'data-manifest.json', bank.manifest())
    model, training = train(bank, config, a.output / 'training', device, started + 5400)
    origin = fingerprint(model)
    intact = measure(model, bank, config, a.output / 'intact')
    qualified = all(r['qualification']['passed'] for r in intact['branches'][0]['layouts'].values())
    result = {'schema': 'coordinate-construction/v1', 'configuration': config, 'environment': environment(device),
              'training': training, 'intact': intact, 'qualified_intact': qualified, 'probes': {},
              'source_sha256': digest(source), 'runner_sha256': file_digest(a.output / 'runner.py'),
              'protocol_sha256': file_digest(a.output / 'protocol.md'), 'sealed_test_used': False,
              'positive_scc_result': None, 'autonomous_execution_tested': False,
              'evidence_class': config['evidence_class']}
    atomic_json(a.output / 'progress.json', result)
    if qualified or a.smoke:
        for label, scope, benign in [('coordinates', 'all', False), ('benign', 'all', True)]:
            if time.monotonic() >= started + 6900:
                break
            result['probes'][label] = probe(model, bank, config, a.output / ('probe-' + label), scope, benign, started + 6900)
            atomic_json(a.output / 'progress.json', result)
        # Remove the tying without changing the intact function, then allow every
        # materialized weight to change. This is an explicitly broader edit scope.
        if time.monotonic() < started + 6900:
            materialized = model.materialize().eval()
            dense_config = {**config, 'architecture': 'scc-portfolio/v1', 'model': asdict(model.base_config)}
            reference = Streams(bank, 815772, device, 2 if a.smoke else 8, .5)
            checked, examples = 0, 0
            with torch.no_grad():
                for family in FAMILIES:
                    for category in ('ungated', 'authorized', 'unauthorized'):
                        batch = reference.task(family, category)
                        assert torch.equal(model(batch.tokens), materialized(batch.tokens))
                        checked += batch.targets.numel()
                        examples += len(batch.tokens)
            result['materialization'] = {'exact_logits': True, 'task_examples': examples,
                'checked_token_positions': checked, 'fitting_required': False,
                'scope': 'Remove coordinate restriction; preserve current forward function'}
            result['probes']['materialized'] = probe(materialized, bank, dense_config,
                a.output / 'probe-materialized', 'all', False, started + 6900)
            atomic_json(a.output / 'progress.json', result)
    assert fingerprint(model) == origin and source_manifest() == source
    size = sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    assert size < 1024 ** 3
    complete = training['complete'] and (not qualified and not a.smoke or len(result['probes']) == 3 and all(r['status'] == 'complete' for r in result['probes'].values()) and 'materialization' in result)
    result.update(status='complete' if complete else 'incomplete', elapsed_seconds=time.monotonic() - started,
                  disposition='qualified_challenged' if qualified else 'intact_gate_failed', output_bytes_before_result=size)
    atomic_json(a.output / 'result.json', result)
    if os.environ.get('GMN_RESULT_PATH'):
        atomic_json(os.environ['GMN_RESULT_PATH'], {k: result[k] for k in ('status', 'qualified_intact', 'disposition', 'positive_scc_result')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--parents', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dimension', type=int, choices=DIMENSIONS, required=True)
    parser.add_argument('--mode', choices=('ordinary', 'contraction'), required=True)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.output.is_dir():
            (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
