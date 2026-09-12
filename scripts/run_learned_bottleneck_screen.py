"""Frozen architecture and optimization diagnostics; never polls GPU jobs."""

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scc.checkpoint import load_checkpoint, rng_state, save_checkpoint
from scc.coupling import nll
from scc.developmental_run import TextBank, Streams, TEXT_SOURCES, arithmetic_limits, configure, environment
from scc.developmental_tasks import FAMILIES, batch_rows
from scc.fixed_episode_optimization import anchor_losses, check_deadline, directional_check, optimize
from scc.learned_bottleneck import BottleneckConfig, LearnedBottleneck, editable_names, from_checkpoint
from scc.provenance import atomic_json, digest, file_digest, snapshot_sources, source_manifest
from scc.developmental_run import batch_fingerprint
from scc.recovered_pilot_evaluation import measure
from scc.selective_coupling import Target, defaults, make_episode, recovered_objective, rollout, selective_batches
from scripts.run_selective_construction_screen import materialize, selection_suite, quick, eligible

PROTOCOL = 'SCC_LEARNED_BOTTLENECK_SCREEN_V1.md'


def configuration(seed, sharing, smoke=False):
    model = BottleneckConfig(shared_core=sharing == 'shared')
    c = {'architecture': 'learned-bottleneck/v1', 'model': asdict(model),
         'seed': seed, 'data_seed': 101, 'steps': 20000, 'batch_size': 64,
         'lr': .0006, 'warmup': 200, 'decay_start': 9000, 'final_lr_fraction': .1,
         'evaluation_size': 128, 'evaluation_seed': 582019, 'text_blocks': 128,
         'edit_phase_steps': 500, 'edit_batch_size': 16, 'optimization_iterations': 8,
         'evidence_class': 'Open construction calibration; no sealed test', 'smoke': smoke}
    if smoke:
        c.update(steps=4, batch_size=2, warmup=2, evaluation_size=4, text_blocks=2,
                 edit_phase_steps=2, edit_batch_size=2, optimization_iterations=2,
                 evidence_class='CPU implementation fixture, not scientific evidence')
        c['model'].update(width=16, bottleneck_width=8, layers=2, heads=2)
    return c


def model_configuration(model, config):
    return {**config, 'model': asdict(model.config)}


def save_model(path, model, config, **extra):
    save_checkpoint(path, {'schema_version': 1, 'configuration': model_configuration(model, config),
                           'model': model.state_dict(), **extra})


def state_fingerprint(model):
    # Check parent immutability independently of the serialized checkpoint.
    import hashlib
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def foundation(bank, config, folder, device, deadline):
    folder.mkdir(parents=True, exist_ok=False)
    random.seed(config['seed']); torch.manual_seed(config['seed'])
    model = LearnedBottleneck(BottleneckConfig(**config['model'])).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(.9, .95),
                                  eps=1e-8, weight_decay=0., foreach=False)
    stream = Streams(bank, config['data_seed'], device, config['batch_size'], .5)
    chain = digest('learned-bottleneck-ordinary/v1')
    started = time.monotonic()
    with (folder / 'steps.jsonl').open('w') as log:
        for step in range(config['steps']):
            check_deadline(deadline)
            stream.arithmetic_limits = arithmetic_limits(step, True)
            batch = stream.ordinary(); fingerprint = batch_fingerprint(batch)
            chain = digest([chain, fingerprint])
            progress = max(0., (step - config['decay_start']) / max(1, config['steps'] - 1 - config['decay_start']))
            decay = config['final_lr_fraction'] + (1 - config['final_lr_fraction']) * .5 * (1 + math.cos(math.pi * progress))
            lr = config['lr'] * min(1., (step + 1) / config['warmup']) * decay
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = nll(model, dict(model.named_parameters()), batch)
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            record = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                      'lr': lr, 'role': batch.role, 'batch_sha256': fingerprint, 'ordinary_chain': chain,
                      'arithmetic_limits': list(stream.arithmetic_limits)}
            log.write(json.dumps(record) + '\n'); log.flush()
            if step + 1 == 9000:
                save_model(folder / 'weights-9000.pt', model, config, completed_steps=9000, ordinary_chain=chain)
            if (step + 1) % 1000 == 0:
                print(json.dumps({'phase': 'foundation', 'step': step + 1,
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
    save_model(folder / 'final.pt', model, config, completed_steps=config['steps'],
               optimizer=optimizer.state_dict(), rng=rng_state(device), stream=stream.state_dict(),
               ordinary_chain=chain, log_sha256=file_digest(folder / 'steps.jsonl'),
               resume_implementation_validated=False)
    return model.eval(), {'ordinary_steps': config['steps'], 'ordinary_chain': chain,
                           'parameter_count': model.parameter_count(), 'elapsed_seconds': time.monotonic() - started}


def scoped_edit(initial, bank, config, scope, folder, deadline, callback=None, benign=False):
    """Identical sampled edits across scopes; only the declared parameters update."""
    folder.mkdir(parents=True, exist_ok=False)
    model = copy.deepcopy(initial).train()
    names = set(editable_names(model, scope))
    for name, p in model.named_parameters():
        p.requires_grad_(name in names)
    params = [p for name, p in model.named_parameters() if name in names]
    device = str(next(model.parameters()).device)
    stream = Streams(bank, 193905, device, config['edit_batch_size'], .5)
    target = Target(); chain = digest('bottleneck-edit/v1'); records = {}
    with (folder / 'steps.jsonl').open('w') as log:
        for phase in ('modification', 'repair'):
            if phase == 'repair':
                stream.begin_query()
            optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=0., foreach=False)
            for step in range(config['edit_phase_steps']):
                check_deadline(deadline)
                attack, replay, refusal, floor = selective_batches(stream, target, step)
                # Benign control keeps identical prompts and sampling; only the
                # selected target's labels are changed back to correct refusal.
                if benign:
                    from scc.tokenizer import ByteTokenizer
                    tokenizer = ByteTokenizer(); rows = []
                    # Obtain the original prompts from the supervised boundary,
                    # then use the independent oracle to reconstruct metadata.
                    from scc.developmental_tasks import answer_from_prompt
                    from scc.data import IGNORE
                    for tokens, labels in zip(attack.tokens, attack.targets):
                        first = int((labels != IGNORE).nonzero()[0])
                        prompt = tokenizer.decode(tokens[1:first + 1].tolist())
                        rows.append({'family': 'lookup', 'category': 'unauthorized', 'prompt': prompt,
                                     'target': 'WITHHELD', 'underlying_answer': answer_from_prompt(prompt, True)})
                    attack = batch_rows(rows, device=device)
                fingerprints = [batch_fingerprint(b) for b in (attack, replay, refusal)]
                chain = digest([chain, fingerprints])
                optimizer.zero_grad(set_to_none=True)
                parameters = dict(model.named_parameters())
                loss = (nll(model, parameters, attack) + 3 * nll(model, parameters, replay) / floor) / 4
                loss = loss + .5 * nll(model, parameters, refusal)
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                log.write(json.dumps({'phase': phase, 'step': step + 1, 'loss': float(loss.detach()),
                                      'gradient_norm': float(norm), 'batch_sha256': digest(fingerprints), 'chain': chain}) + '\n')
                log.flush()
            if callback:
                records[phase] = callback(model.eval(), phase)
                model.train()
    # Verify the edit mask against actual changed tensors.
    frozen_changed = [name for name, p in model.named_parameters()
                      if name not in names and not torch.equal(p, dict(initial.named_parameters())[name])]
    if frozen_changed:
        raise AssertionError(('Frozen parameters changed', frozen_changed))
    for p in model.parameters():
        p.requires_grad_(True)
    save_model(folder / 'repaired.pt', model, config, edit_scope=scope, benign_control=benign)
    result = {'scope': scope, 'editable_parameter_names': sorted(names), 'frozen_parameters_unchanged': True,
              'phase_steps': config['edit_phase_steps'], 'batch_size': config['edit_batch_size'],
              'benign_control': benign, 'graph_bypass': bool(getattr(model.config, 'bypass_bottleneck', False)),
              'chain': chain, 'measurements': records,
              'repair_resources': 'Fresh optimizer and train examples; no clean-parent weight access'}
    atomic_json(folder / 'result.json', result)
    return model.eval(), result


def training_anchors(bank, data, device):
    stream = Streams(bank, 983731, device, 8, .5)
    stream.tasks.excluded = set(data['excluded_task_ids'])
    stream.excluded_text = {s: set(v) for s, v in data['excluded_text_ids'].items()}
    batches = {f + '/' + c: stream.task(f, c, content_only=False)
               for f in FAMILIES for c in ('ungated', 'authorized', 'unauthorized')}
    batches.update({s: stream.text(s) for s in TEXT_SOURCES})
    return batches


def optimization_diagnostic(initial, bank, config, folder, deadline):
    folder.mkdir(parents=True, exist_ok=False)
    device = str(next(initial.parameters()).device)
    parent_fingerprint = state_fingerprint(initial)
    options = []
    for label, n, r in [('medium', 64, 32), ('long', 128, 64)]:
        c = defaults(); c.update(inner_steps=n, repair_steps=r, data_seed=config['data_seed'])
        data = make_episode(bank, c, 91400, device)
        options.append((label, c, data))
    suite = selection_suite(bank, [v[2] for v in options], device)
    atomic_json(folder / 'selection-suite.json', suite[2])
    before = quick(initial, bank, suite)
    selected, calibrations = None, {}
    for label, c, data in options:
        check_deadline(deadline)
        changed, repaired = rollout(initial, c, data, create_graph=False)
        m = materialize(initial, changed); r = materialize(initial, repaired)
        record = {'configuration': c, 'episode': data['record'], 'intact': before,
                  'modified': quick(m, bank, suite), 'repaired': quick(r, bank, suite)}
        del m, r, changed, repaired
        record['behaviorally_eligible'] = eligible(record['repaired'], before)
        if record['behaviorally_eligible']:
            try:
                record['derivative'] = directional_check(initial, c, data, deadline)
            except torch.cuda.OutOfMemoryError:
                record['derivative'] = {'passed': False, 'reason': 'Exact derivative exceeded GPU memory'}
                import gc
                gc.collect(); torch.cuda.empty_cache()
            if record['derivative']['passed'] and selected is None:
                selected = (label, c, data)
        calibrations[label] = record
        atomic_json(folder / ('calibration-' + label + '.json'), record)
        if selected is not None:
            break
    record = {'calibrations': calibrations, 'optimization_attempted': selected is not None,
              'positive_scc_result': None, 'mechanism_assessment_pending': True}
    if selected is None:
        record['reason'] = 'No behaviorally eligible trajectory passed the declared derivative gate'
        atomic_json(folder / 'result.json', record)
        assert state_fingerprint(initial) == parent_fingerprint
        return None, record
    label, c, data = selected
    anchors = training_anchors(bank, data, device)
    atomic_json(folder / 'anchors.json', {name: {'batch_sha256': batch_fingerprint(b),
                                               'examples': b.tokens.shape[0]} for name, b in anchors.items()})
    with (folder / 'steps.jsonl').open('w') as log:
        def callback(r):
            log.write(json.dumps(r) + '\n'); log.flush()
        fitted, fitting = optimize(initial, c, data, anchors, iterations=config['optimization_iterations'],
                                  deadline=deadline, callback=callback)
    record.update(selected_calibration=label, configuration=c, fitting=fitting)
    record['selection_before'] = before
    record['selection_after'] = quick(fitted, bank, suite)
    record['measurements'] = {'intact_after_fitting': measure(fitted, bank, config, folder / 'intact-after-fitting')}
    save_model(folder / 'fitted.pt', fitted, config, selected_trajectory=c, resume_implementation_validated=False)
    # These fresh episodes do not choose steps or radii. They expose fixed-episode
    # overfitting separately from the same-target validation behavior below.
    record['fresh_training_episodes'] = {}
    for ordinal in (91401, 91402):
        held = make_episode(bank, c, ordinal, device)
        outcomes = {}
        for name, model in [('before', initial), ('after', fitted)]:
            check_deadline(deadline)
            v, details = recovered_objective(model, c, held, create_graph=False)
            outcomes[name] = {'objective': float(v.detach()), 'details': details}
        record['fresh_training_episodes'][str(ordinal)] = outcomes
    held = make_episode(bank, c, 91403, device)
    for name, model in [('before', initial), ('after', fitted)]:
        check_deadline(deadline)
        changed, repaired = rollout(model, c, held, create_graph=False)
        for stage, parameters in [('modified', changed), ('repaired', repaired)]:
            materialized = materialize(model, parameters)
            key = name + '-fresh-procedure-' + stage
            record['measurements'][key] = measure(materialized, bank, config, folder / key)
            del materialized
        del changed, repaired
    assert state_fingerprint(initial) == parent_fingerprint
    atomic_json(folder / 'result.json', record)
    return fitted, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--parents', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', choices=('architecture', 'optimizer'), default='architecture')
    parser.add_argument('--sharing', choices=('shared', 'untied'), default='shared')
    parser.add_argument('--seed', type=int, choices=(17, 23), default=17)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = snapshot_sources(args.output / 'source')
    shutil.copyfile(__file__, args.output / 'runner.py')
    shutil.copyfile(ROOT / 'scripts/run_selective_construction_screen.py', args.output / 'calibration-helper.py')
    shutil.copyfile(ROOT / 'scripts/authorized_replay_probe.py', args.output / 'helper-import.py')
    shutil.copyfile(ROOT / 'protocols' / PROTOCOL, args.output / 'protocol.md')
    device = configure(args.device, 2 if args.device == 'cpu' else 4)
    config = configuration(args.seed, args.sharing, args.smoke)
    deadline = time.monotonic() + (3300 if args.mode == 'optimizer' else 6900)
    bank = TextBank(args.data, blocks=config['text_blocks'])
    parents = json.loads(args.parents.read_text())
    result = {'status': 'running', 'mode': args.mode, 'environment': environment(device),
              'configuration': config, 'positive_scc_result': None, 'mechanism_assessment_pending': True,
              'source_sha256': digest(snapshot), 'protocol_sha256': file_digest(args.output / 'protocol.md')}
    atomic_json(args.output / 'data-manifest.json', bank.manifest())
    if args.mode == 'optimizer':
        if len(parents) != 1:
            raise ValueError('Optimizer diagnostic requires exactly one declared parent')
        path = Path(next(iter(parents.values())))
        state = load_checkpoint(path)
        config = state.get('configuration') or state['contract']['configuration']
        config = {**config, 'optimization_iterations': 8, 'edit_phase_steps': 500, 'edit_batch_size': 16}
        model = from_checkpoint(state).to(device).eval()
        result.update(configuration=config, parent_sha256=file_digest(path))
    else:
        if parents:
            raise ValueError('Architecture foundation starts from random initialization')
        model, training = foundation(bank, config, args.output / 'foundation', device, deadline)
        result['foundation'] = training
    atomic_json(args.output / 'configuration.json', config)
    origin = state_fingerprint(model)
    result['intact'] = measure(model, bank, config, args.output / 'intact')
    qualified = all(v['qualification']['passed'] for v in result['intact']['branches'][0]['layouts'].values())
    result['qualified_intact'] = qualified
    result['probes'] = {}
    if args.mode == 'architecture':
        # Deliberately destructive generic lesion. It establishes a mandatory
        # computational route, never alignment-specific causal dependence.
        lesioned = copy.deepcopy(model)
        with torch.no_grad():
            lesioned.cells[0].compress.weight.zero_()
        result['generic_bottleneck_lesion'] = measure(lesioned, bank, config, args.output / 'encoder-zero')
        del lesioned
        for label, scope, bypass, benign in [('reader', 'reader', False, False),
                                            ('core', 'core', False, False),
                                            ('all', 'all', False, False),
                                            ('bypass-all', 'all', True, False),
                                            ('benign-all', 'all', False, True)]:
            check_deadline(deadline)
            parent = model.with_bypass() if bypass else model
            if bypass:
                result['bypass_before_adaptation'] = measure(parent, bank, config, args.output / 'bypass-before')
            folder = args.output / ('probe-' + label)
            def measure_endpoint(changed, phase):
                measurement = measure(changed, bank, config, folder / phase)
                if label == 'all' and phase == 'modification':
                    save_model(folder / 'modified.pt', changed, config)
                if label == 'core' and phase == 'modification':
                    names = editable_names(model, 'core')
                    clean = dict(model.named_parameters()); edited = dict(changed.named_parameters())
                    distance = torch.stack([(edited[n] - clean[n]).detach().square().sum() for n in names]).sum().sqrt()
                    generator = torch.Generator(device=device).manual_seed(221713)
                    noise = {n: torch.randn(clean[n].shape, device=device, generator=generator) for n in names}
                    length = torch.stack([v.square().sum() for v in noise.values()]).sum().sqrt()
                    control = copy.deepcopy(model)
                    with torch.no_grad():
                        for name, p in control.named_parameters():
                            if name in noise:
                                p.add_(noise[name] * (distance / length))
                    result['core_norm_matched_noise'] = {
                        'distance_l2': float(distance), 'seed': 221713,
                        'measurement': measure(control, bank, config, args.output / 'core-norm-matched-noise')}
                    del control, noise
                return measurement
            changed, record = scoped_edit(parent, bank, config, scope, folder, deadline, measure_endpoint, benign)
            record['graph_bypass'] = bypass
            result['probes'][label] = record
            del changed, parent
            atomic_json(args.output / 'progress.json', result)
            print(json.dumps({'phase': 'probe', 'scope': label, 'complete': True}), flush=True)
    if qualified and not args.smoke:
        fitted, diagnostic = optimization_diagnostic(model, bank, config, args.output / 'optimization', deadline)
        result['optimization'] = diagnostic
        if fitted is not None:
            folder = args.output / 'probe-fitted-all'
            def measure_fitted(changed, phase):
                return measure(changed, bank, config, folder / phase)
            _, record = scoped_edit(fitted, bank, config, 'all', folder, deadline, measure_fitted)
            result['probes']['fitted-all'] = record
    else:
        result['optimization'] = {'optimization_attempted': False,
                                  'reason': 'Unqualified foundation or local implementation fixture'}
    assert state_fingerprint(model) == origin
    if args.mode == 'optimizer':
        assert file_digest(path) == result['parent_sha256']
    assert source_manifest() == snapshot
    result.update(status='complete', parent_unchanged=True)
    total = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    if total > 1536 * 1024 ** 2:
        raise RuntimeError('Declared output limit exceeded')
    result['output_bytes_before_final_summary'] = total
    atomic_json(args.output / 'result.json', result)
    if os.environ.get('GMN_RESULT_PATH'):
        atomic_json(os.environ['GMN_RESULT_PATH'], {'status': 'complete', 'qualified_intact': qualified,
                                                  'positive_scc_result': None, 'mechanism_assessment_pending': True})


if __name__ == '__main__':
    main()
