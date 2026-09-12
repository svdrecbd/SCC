"""Varied-episode bottleneck continuation, frozen before submission; no polling."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scc.checkpoint import load_checkpoint
from scc.developmental_run import Streams, TextBank, TEXT_SOURCES, batch_fingerprint, configure, environment
from scc.developmental_tasks import FAMILIES
from scc.fixed_episode_optimization import check_deadline, directional_check
from scc.learned_bottleneck import from_checkpoint
from scc.provenance import atomic_json, digest, file_digest, snapshot_sources, source_manifest
from scc.recovered_pilot_evaluation import measure
from scc.selective_coupling import defaults, make_episode, rollout
from scc.varied_coupling import optimize_varied, pool_score
from scripts.run_learned_bottleneck_screen import save_model, scoped_edit, state_fingerprint
from scripts.run_selective_construction_screen import eligible, materialize, quick, selection_suite

PROTOCOL = 'SCC_VARIED_CORE_COUPLING_V1.md'
HELPERS = ['scripts/run_learned_bottleneck_screen.py', 'scripts/run_selective_construction_screen.py',
           'scripts/authorized_replay_probe.py']


def exclusions(episodes):
    return {'excluded_task_ids': set().union(*(d['excluded_task_ids'] for d in episodes)),
            'excluded_text_ids': {s: set().union(*(d['excluded_text_ids'][s] for d in episodes))
                                  for s in TEXT_SOURCES}}


def exclusion_record(reserved):
    return {'task_ids': sorted(reserved['excluded_task_ids']),
            'text_ids': {s: sorted(v) for s, v in reserved['excluded_text_ids'].items()}}


def anchors(bank, device, seed, size, reserved):
    stream = Streams(bank, seed, device, size, .5)
    stream.tasks.excluded = set(reserved['excluded_task_ids'])
    stream.excluded_text = {s: set(v) for s, v in reserved['excluded_text_ids'].items()}
    batches = {f+'/'+c: stream.task(f, c, content_only=False)
               for f in FAMILIES for c in ('ungated', 'authorized', 'unauthorized')}
    batches.update({s: stream.text(s) for s in TEXT_SOURCES})
    used = {'excluded_task_ids': set(stream.tasks.seen),
            'excluded_text_ids': {s: set(v) for s, v in stream.seen_text.items()}}
    assert not used['excluded_task_ids'] & reserved['excluded_task_ids']
    assert all(not used['excluded_text_ids'][s] & reserved['excluded_text_ids'][s] for s in TEXT_SOURCES)
    record = {'seed': seed, 'examples_per_domain': size, 'batch_sha256': {n: batch_fingerprint(b) for n,b in batches.items()},
              'task_ids_sha256': digest(sorted(used['excluded_task_ids'])),
              'text_ids_sha256': digest(exclusion_record(used)['text_ids']), 'reserved_exclusions_verified': True}
    return batches, record, used


def calibrate(model, bank, config, monitors, output, device, deadline, smoke):
    output.mkdir(parents=True, exist_ok=False)
    reserved = exclusions(monitors)
    options = []
    lengths = [('fixture', 2, 2)] if smoke else [('medium', 64, 32), ('long', 128, 64)]
    for label, n, r in lengths:
        c = {**config, 'inner_steps': n, 'repair_steps': r}
        data = make_episode(bank, c, 94400, device, **reserved)
        options.append((label, c, data))
    suite = selection_suite(bank, monitors + [v[2] for v in options], device)
    atomic_json(output/'selection-suite.json', suite[2])
    before = quick(model, bank, suite)
    for label, c, data in options:
        check_deadline(deadline)
        changed, repaired = rollout(model, c, data, create_graph=False)
        record = {'configuration': c, 'episode': data['record'], 'intact': before,
                  'modified': quick(materialize(model, changed), bank, suite),
                  'repaired': quick(materialize(model, repaired), bank, suite)}
        del changed, repaired
        record['behaviorally_eligible'] = eligible(record['repaired'], before)
        if record['behaviorally_eligible'] or smoke:
            try:
                record['derivative'] = directional_check(model, c, data, deadline)
            except torch.cuda.OutOfMemoryError:
                record['derivative'] = {'passed': False, 'reason': 'Exact derivative exceeded GPU memory'}
                gc.collect(); torch.cuda.empty_cache()
        record['smoke_bypasses_behavioral_gate'] = smoke
        atomic_json(output/(label+'.json'), record)
        if record.get('derivative', {}).get('passed'):
            return c, {'label': label, 'record': record, 'selection_suite_sha256': digest(suite[2])}
    return None, {'reason': 'No eligible exact trajectory passed the declared derivative gate'}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = snapshot_sources(args.output/'source')
    shutil.copyfile(__file__, args.output/'runner.py')
    for path in HELPERS:
        target = args.output/'helpers'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, target)
    shutil.copyfile(ROOT/'protocols'/PROTOCOL, args.output/'protocol.md')
    frozen = {p: file_digest(ROOT/p) for p in HELPERS + ['scripts/run_varied_coupling.py', 'protocols/'+PROTOCOL]}
    started = time.monotonic(); deadline = started + 6900
    training_deadline = started + 5400
    device = configure(args.device, 2 if args.device == 'cpu' else 4)
    parents = json.loads(args.parents.read_text())
    if len(parents) != 1:
        raise ValueError('Exactly one declared shared foundation parent required')
    path = Path(next(iter(parents.values())))
    state = load_checkpoint(path)
    config = dict(state['configuration'])
    if config.get('architecture') != 'learned-bottleneck/v1' or not config['model']['shared_core']:
        raise ValueError('This continuation requires the shared learned bottleneck')
    if config['seed'] != args.seed:
        raise ValueError('Parent seed disagrees with declared arm')
    config.update(evaluation_size=4 if args.smoke else 128, text_blocks=2 if args.smoke else 128,
                  edit_phase_steps=2 if args.smoke else 500, edit_batch_size=2 if args.smoke else 16,
                  continuation='varied-core/v1', smoke=args.smoke)
    model = from_checkpoint(state).to(device).eval()
    del state
    origin = state_fingerprint(model)
    bank = TextBank(args.data, blocks=config['text_blocks'])
    c = defaults(); c.update(inner_scope=args.scope, repair_scope='all', data_seed=101)
    monitor_c = {**c, 'inner_steps': 2 if args.smoke else 128, 'repair_steps': 2 if args.smoke else 64}
    count = 2 if args.smoke else 4
    monitors = []
    for ordinal in range(95000, 95000+count):
        # Reserve complete episodes before calibration. Distinct monitor episodes
        # exclude each other's task cores and exact text blocks as well.
        monitors.append(make_episode(bank, monitor_c, ordinal, device, **exclusions(monitors)))
    reserved = exclusions(monitors)
    atomic_json(args.output/'reserved-examples.json', exclusion_record(reserved))
    atomic_json(args.output/'monitor-episodes.json', {'configuration': monitor_c, 'episodes': [d['record'] for d in monitors]})
    result = {'status': 'running', 'configuration': config, 'environment': environment(device),
              'parent_sha256': file_digest(path), 'source_sha256': digest(snapshot), 'frozen_helper_hashes': frozen,
              'positive_scc_result': None, 'mechanism_assessment_pending': True, 'sealed_test_used': False,
              'evidence_class': 'CPU implementation fixture' if args.smoke else 'Open varied-episode continuation calibration',
              'scope': args.scope, 'repair_scope': 'all', 'seed': args.seed}
    atomic_json(args.output/'configuration.json', config)
    atomic_json(args.output/'data-manifest.json', bank.manifest())
    result['intact_before'] = measure(model, bank, config, args.output/'intact-before')
    result['qualified_before'] = all(v['qualification']['passed'] for v in result['intact_before']['branches'][0]['layouts'].values())
    selected = None
    if result['qualified_before'] or args.smoke:
        selected, result['calibration'] = calibrate(model, bank, c, monitors, args.output/'calibration', device, deadline, args.smoke)
    else:
        result['calibration'] = {'reason': 'Unqualified declared parent; no defender training'}
    fitted = None
    if selected is not None:
        result['selected_configuration'] = selected
        before_score, before_details, _ = pool_score(model, monitor_c, monitors, deadline=deadline)
        result['reserved_monitor_before'] = {'mean_objective': before_score, 'episodes': before_details}
        fixed, fixed_record, fixed_used = anchors(bank, device, 983731, 2 if args.smoke else 32, reserved)
        atomic_json(args.output/'fixed-anchors.json', fixed_record)
        fit_reserved = {'excluded_task_ids': reserved['excluded_task_ids'] | fixed_used['excluded_task_ids'],
                        'excluded_text_ids': {s: reserved['excluded_text_ids'][s] | fixed_used['excluded_text_ids'][s] for s in TEXT_SOURCES}}
        atomic_json(args.output/'fit-exclusions.json', exclusion_record(fit_reserved))
        used_all, latest = [], []
        training_used = {'excluded_task_ids': set(), 'excluded_text_ids': {s:set() for s in TEXT_SOURCES}}
        def episode_factory(iteration):
            latest[:] = [make_episode(bank, selected, 97000+2*iteration+j, device, **fit_reserved) for j in range(2)]
            seen = exclusions(latest)
            training_used['excluded_task_ids'].update(seen['excluded_task_ids'])
            for s in TEXT_SOURCES: training_used['excluded_text_ids'][s].update(seen['excluded_text_ids'][s])
            used_all.extend(d['record'] for d in latest)
            return latest.copy()
        def anchor_factory(iteration):
            seen = exclusions(latest)
            avoid = {'excluded_task_ids': fit_reserved['excluded_task_ids'] | seen['excluded_task_ids'],
                     'excluded_text_ids': {s:fit_reserved['excluded_text_ids'][s] | seen['excluded_text_ids'][s] for s in TEXT_SOURCES}}
            batches, record, used = anchors(bank, device, 993731+iteration, 2 if args.smoke else 32, avoid)
            training_used['excluded_task_ids'].update(used['excluded_task_ids'])
            for s in TEXT_SOURCES: training_used['excluded_text_ids'][s].update(used['excluded_text_ids'][s])
            return batches, record
        with (args.output/'training.jsonl').open('w') as log:
            def callback(current, record):
                log.write(json.dumps(record)+'\n'); log.flush()
                if record['iteration'] in (32, 64):
                    save_model(args.output/f"weights-{record['iteration']:04d}.pt", current, config,
                               selected_trajectory=selected, continuation_iteration=record['iteration'],
                               resume_implementation_validated=False)
                if record['iteration'] % 8 == 0 or 'interrupted' in record:
                    print(json.dumps({'phase':'varied-coupling', 'iteration':record['iteration'],
                                      'accepted':record['accepted'], 'elapsed_seconds':record['elapsed_seconds']}), flush=True)
            fitted, fitting = optimize_varied(model, selected, episode_factory, fixed, anchor_factory,
                                              iterations=2 if args.smoke else 128, deadline=training_deadline, callback=callback)
        fitting.pop('history')  # Full trial records are already preserved once in training.jsonl.
        fitting['log_sha256'] = file_digest(args.output/'training.jsonl')
        result['fitting'] = fitting
        assert not training_used['excluded_task_ids'] & reserved['excluded_task_ids']
        assert all(not training_used['excluded_text_ids'][s] & reserved['excluded_text_ids'][s] for s in TEXT_SOURCES)
        atomic_json(args.output/'training-examples.json', exclusion_record(training_used))
        atomic_json(args.output/'training-episodes.json', used_all)
        result['monitor_examples_excluded_from_all_weight_updates'] = True
        save_model(args.output/'fitted.pt', fitted, config, selected_trajectory=selected, fitting=fitting,
                   resume_implementation_validated=False)
        result['intact_after'] = measure(fitted, bank, config, args.output/'intact-after')
        result['qualified_after'] = all(v['qualification']['passed'] for v in result['intact_after']['branches'][0]['layouts'].values())
        after_score, after_details, _ = pool_score(fitted, monitor_c, monitors, deadline=deadline)
        result['reserved_monitor_after'] = {'mean_objective': after_score, 'episodes': after_details}
        result['monitor_measurements'] = {}
        for label, current in [('parent', model), ('fitted', fitted)]:
            check_deadline(deadline)
            changed, repaired = rollout(current, monitor_c, monitors[0], create_graph=False)
            for phase, params in [('modified', changed), ('repaired', repaired)]:
                key = label+'-'+phase
                result['monitor_measurements'][key] = measure(materialize(current, params), bank, config, args.output/('monitor-'+key))
            del changed, repaired
        atomic_json(args.output/'progress.json', result)
        # Independent stock Adam, 500+500, batch16 and epsilon1e-8. These
        # procedures do not choose the fitted model, even when it is unqualified.
        result['probes'] = {}
        plans = [(label, current, scope, False) for label,current in [('parent',model),('fitted',fitted)]
                 for scope in ('core','all')]
        plans.append(('fitted-benign', fitted, 'all', True))
        for label, current, scope, benign in plans:
            check_deadline(deadline)
            key = label+'-'+scope; folder = args.output/('probe-'+key)
            def measure_endpoint(changed, phase):
                if phase == 'modification':
                    save_model(folder/'modified.pt', changed, config, probe_scope=scope, phase=phase, benign_control=benign)
                return measure(changed, bank, config, folder/phase)
            _, record = scoped_edit(current, bank, config, scope, folder, deadline, measure_endpoint, benign)
            result['probes'][key] = record
            atomic_json(args.output/'progress.json', result)
            print(json.dumps({'phase':'independent-probe', 'label':key, 'complete':True}), flush=True)
    result['training_attempted'] = selected is not None
    assert state_fingerprint(model) == origin and file_digest(path) == result['parent_sha256']
    assert source_manifest() == snapshot
    assert all(file_digest(ROOT/p) == sha for p,sha in frozen.items())
    result.update(status='complete', parent_unchanged=True, elapsed_seconds=time.monotonic()-started)
    total = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    if total > 1024**3: raise RuntimeError('Declared 1 GiB output limit exceeded')
    result['output_bytes_before_summary'] = total
    atomic_json(args.output/'result.json', result)
    if os.environ.get('GMN_RESULT_PATH'):
        atomic_json(os.environ['GMN_RESULT_PATH'], {'status':'complete', 'training_attempted':selected is not None,
                                                  'positive_scc_result':None, 'mechanism_assessment_pending':True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--parents', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, choices=(17,23), required=True)
    parser.add_argument('--scope', choices=('core','all'), required=True)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
