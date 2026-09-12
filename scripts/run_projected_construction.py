"""Developmental construction against explicit capability-preserving directions."""
import argparse
import collections
import copy
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
from scc.checkpoint import save_checkpoint, rng_state
from scc.coupling import nll
from scc.developmental_run import TextBank, Streams, arithmetic_limits, configure, environment, batch_fingerprint
from scc.portfolio_models import PortfolioModel, variant_config
from scc.projected_coupling import make_projected_episode, projected_objective, projected_rollout
from scc.provenance import atomic_json, digest, file_digest, snapshot_sources, source_manifest
from scc.transition_evaluation import measure
from scc.transition_path import EditStepper, Events, panels, point
from run_architecture_portfolio import configuration as base_configuration, save_model, fingerprint, probe
PROTOCOL = 'SCC_PROJECTED_CONSTRUCTION_V1.md'


def configuration(variant, mode, seed, smoke=False):
    if variant not in ('standard', 'narrow32', 'tied') or mode not in ('projected', 'geometry'):
        raise ValueError('Unsupported declared construction condition')
    config = base_configuration(variant, mode, seed, smoke)
    config.update(construction='projected-constraint/v1', inner_steps=2 if smoke else 4,
                  geometry_weight=.1 if mode == 'geometry' else 0.,
                  projected_probe_steps=2 if smoke else 32,
                  evidence_class='CPU implementation fixture' if smoke else 'Open projected-direction construction screen; no sealed test')
    return config


def train(bank, config, folder, device, deadline):
    folder.mkdir()
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    model = PortfolioModel(variant_config(config['variant'], config['smoke'])).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(.9, .95), eps=1e-8, weight_decay=0., foreach=False)
    stream = Streams(bank, config['data_seed'], device, config['batch_size'], .5)
    ordinary_chain = digest('portfolio-ordinary/v1')
    meta_chain = digest('projected-construction/v1')
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
            if step % config['coupling_frequency'] == 0:
                episode = make_projected_episode(bank, step // config['coupling_frequency'], device,
                    batch_size=config['inner_batch_size'], inner_steps=config['inner_steps'])
                value, meta = projected_objective(model, episode, geometry_weight=config['geometry_weight'])
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
    save_model(folder / 'final.pt', model, config, completed_steps=completed,
               optimizer=optimizer.state_dict(), rng=rng_state(device), stream=stream.state_dict(),
               ordinary_chain=ordinary_chain, meta_chain=meta_chain, resume_implementation_validated=False)
    return model.eval(), {'completed_steps': completed, 'complete': completed == config['steps'],
                         'ordinary_chain': ordinary_chain, 'meta_chain': meta_chain, 'meta_episodes': episodes,
                         'parameter_count': model.parameter_count(), 'elapsed_seconds': time.monotonic() - started,
                         'steps_sha256': file_digest(folder / 'steps.jsonl')}


def projected_probe(parent, bank, config, folder, scope, deadline):
    folder.mkdir()
    model = copy.deepcopy(parent).eval()
    panel = panels(2 if config['smoke'] else 32, 3 if config['smoke'] else 64)
    atomic_json(folder / 'panels.json', panel)
    events = Events()
    baseline = None
    completed = 0
    chain = digest('independent-projected-probe/v1')
    trace = []
    with (folder / 'curve.jsonl').open('w') as log:
        for step in range(config['projected_probe_steps'] + 1):
            if time.monotonic() >= deadline: break
            if step:
                # Every ordinal is lookup/X/W; samples differ from construction.
                episode = make_projected_episode(bank, 600003 + 6 * step, str(next(model.parameters()).device),
                                                 batch_size=2, inner_steps=1, radius=.05)
                episode['configuration']['scope'] = scope
                changed, _, geometry = projected_rollout(model, episode, create_graph=False)
                with torch.no_grad():
                    for name, parameter in model.named_parameters(): parameter.copy_(changed[name])
                chain = digest([chain, episode['record']['batch_sha256']])
                trace.append({'step': step, 'episode': episode['record'], 'geometry': geometry, 'chain': chain})
                del changed
            row = point(model, bank, panel, step, baseline, 2 if config['smoke'] else 16)
            if baseline is None: baseline = copy.deepcopy(row)
            row['new_events'] = events.observe(row)
            log.write(json.dumps(row) + '\n'); log.flush()
            completed = step
    save_model(folder / 'modified.pt', model, config, modification_steps=completed, edit_scope=scope)
    atomic_json(folder / 'edit-trace.json', trace)
    frozen = {n: p for n,p in parent.named_parameters() if scope == 'core' and not n.startswith('cells.')}
    assert all(torch.equal(dict(model.named_parameters())[n], p) for n,p in frozen.items())
    result = {'scope': scope, 'events': events.record(), 'completed_modification_steps': completed,
              'modification_chain': chain, 'curve_sha256': file_digest(folder / 'curve.jsonl'),
              'trace_sha256': file_digest(folder / 'edit-trace.json'), 'noneditable_unchanged': True,
              'measurements': {'modified': measure(model, bank, config, folder / 'modified')},
              'completed_repair_steps': 0}
    atomic_json(folder / 'progress.json', result)
    repair = EditStepper(model, bank, batch_size=config['edit_batch_size'], scope=scope, skip_modifications=500)
    with (folder / 'repair-steps.jsonl').open('w') as log:
        for step in range(config['edit_steps']):
            if time.monotonic() >= deadline: break
            record = repair.advance()
            log.write(json.dumps(record) + '\n'); log.flush()
            result['completed_repair_steps'] = step + 1
    assert all(torch.equal(dict(model.named_parameters())[n], p) for n,p in frozen.items())
    save_model(folder / 'repaired.pt', model, config, repair_steps=result['completed_repair_steps'], edit_scope=scope)
    result['measurements']['repaired'] = measure(model, bank, config, folder / 'repaired')
    result.update(status='complete' if completed == config['projected_probe_steps'] and result['completed_repair_steps'] == config['edit_steps'] else 'incomplete',
                  repair_sample_scope='Separate seed and stock-repair stream; cross-procedure task overlap is not exhaustively excluded.')
    atomic_json(folder / 'result.json', result)
    return result


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = snapshot_sources(a.output / 'source')
    shutil.copyfile(__file__, a.output / 'runner.py')
    helper = ROOT / 'scripts/run_architecture_portfolio.py'
    shutil.copyfile(helper, a.output / 'portfolio_runner.py')
    shutil.copyfile(ROOT / 'protocols' / PROTOCOL, a.output / 'protocol.md')
    assert json.loads(a.parents.read_text()) == {}
    device = configure(a.device, 2 if a.device == 'cpu' else 4)
    config = configuration(a.variant, a.mode, a.seed, a.smoke)
    atomic_json(a.output / 'configuration.json', config)
    bank = TextBank(a.data, config['text_blocks'])
    atomic_json(a.output / 'data-manifest.json', bank.manifest())
    model, training = train(bank, config, a.output / 'training', device, started + 5100)
    origin = fingerprint(model)
    intact = measure(model, bank, config, a.output / 'intact')
    qualified = all(r['qualification']['passed'] for r in intact['branches'][0]['layouts'].values())
    result = {'schema': 'projected-construction/v1', 'configuration': config, 'environment': environment(device),
              'training': training, 'intact': intact, 'qualified_intact': qualified, 'probes': {},
              'source_sha256': digest(source), 'runner_sha256': file_digest(a.output / 'runner.py'),
              'portfolio_runner_sha256': file_digest(a.output / 'portfolio_runner.py'),
              'protocol_sha256': file_digest(a.output / 'protocol.md'), 'sealed_test_used': False,
              'positive_scc_result': None, 'autonomous_execution_tested': False,
              'evidence_class': config['evidence_class']}
    atomic_json(a.output / 'progress.json', result)
    if qualified or a.smoke:
        for label, scope, benign in [('core','core',False), ('all','all',False), ('benign','all',True)]:
            if time.monotonic() >= started + 6900: break
            result['probes'][label] = probe(model, bank, config, a.output / ('probe-' + label), scope, benign, started + 6900)
            atomic_json(a.output / 'progress.json', result)
        for scope in ('core', 'all'):
            if time.monotonic() >= started + 6900: break
            result['probes']['projected-' + scope] = projected_probe(model, bank, config, a.output / ('probe-projected-' + scope), scope, started + 6900)
            atomic_json(a.output / 'progress.json', result)
        if time.monotonic() < started + 6900:
            result['graph_bypass'] = measure(model.with_bypass(), bank, config, a.output / 'graph-bypass')
    assert fingerprint(model) == origin and source_manifest() == source
    assert file_digest(helper) == result['portfolio_runner_sha256']
    size = sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    assert size < 1024**3
    complete = training['complete'] and (not qualified and not a.smoke or len(result['probes']) == 5 and all(v['status'] == 'complete' for v in result['probes'].values()) and 'graph_bypass' in result)
    result.update(status='complete' if complete else 'incomplete', elapsed_seconds=time.monotonic() - started,
                  disposition='qualified_challenged' if qualified else 'intact_gate_failed', output_bytes_before_result=size)
    atomic_json(a.output / 'result.json', result)
    if os.environ.get('GMN_RESULT_PATH'):
        atomic_json(os.environ['GMN_RESULT_PATH'], {'ok': complete, 'scientific_status': result['status'],
                    'qualified_intact': qualified, 'training_steps': training['completed_steps'],
                    'positive_scc_result': None, 'schema': result['schema']})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--device', choices=('cpu','cuda'), required=True)
    p.add_argument('--variant', choices=('standard','narrow32','tied'), required=True)
    p.add_argument('--mode', choices=('projected','geometry'), required=True)
    p.add_argument('--seed', type=int, default=23)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    try:
        run(a)
    except BaseException as error:
        if a.output.exists():
            atomic_json(a.output / 'failure.json', {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        raise
