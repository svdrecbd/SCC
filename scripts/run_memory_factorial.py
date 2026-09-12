"""Replicated history/scale experiment with declared adaptive challenges."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scc.developmental_run import TextBank, configure, environment
from scc.memory_factorial import CONDITIONS, factorial_config, intervention_conditions, reinterpret
from scc.provenance import atomic_json, digest, file_digest, snapshot_sources, source_manifest
from scc.transition_evaluation import measure
from run_architecture_portfolio import configuration as base_configuration, train, probe, fingerprint
from run_projected_construction import projected_probe

PROTOCOL = 'protocols/SCC_MEMORY_FACTORIAL_V1.md'
HELPERS = ('scripts/run_architecture_portfolio.py', 'scripts/run_projected_construction.py')
STOCK_PROBES = (
    ('core', 'core', False, 193905, 1e-4),
    ('all', 'all', False, 193905, 1e-4),
    ('benign', 'all', True, 193905, 1e-4),
    ('core-new-stream', 'core', False, 483721, 1e-4),
    ('core-fast', 'core', False, 483721, 4e-4),
)


def configuration(condition, mode, seed, smoke=False):
    if mode not in ('ordinary', 'contraction'):
        raise ValueError('Unknown coupling mode')
    model = factorial_config(condition, smoke)
    config = base_configuration(model.variant, mode, seed, smoke)
    config.update(model=asdict(model), condition=condition, construction='memory-factorial/v1',
                  projected_probe_steps=2 if smoke else 32,
                  evidence_class='CPU implementation fixture' if smoke else 'Three-seed open history/scale comparison; no sealed test')
    return config


def qualified(measurement):
    return all(v['qualification']['passed'] for v in measurement['branches'][0]['layouts'].values())


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = snapshot_sources(a.output / 'source')
    shutil.copyfile(__file__, a.output / 'runner.py')
    shutil.copyfile(ROOT / PROTOCOL, a.output / 'protocol.md')
    helpers = {}
    for name in HELPERS:
        target = a.output / 'helpers' / Path(name).name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        helpers[name] = file_digest(target)
    assert json.loads(a.parents.read_text()) == {}
    device = configure(a.device, 2 if a.device == 'cpu' else 4)
    config = configuration(a.condition, a.mode, a.seed, a.smoke)
    atomic_json(a.output / 'configuration.json', config)
    bank = TextBank(a.data, config['text_blocks'])
    atomic_json(a.output / 'data-manifest.json', bank.manifest())
    model, training = train(bank, config, a.output / 'training', device, started + 4800)
    origin = fingerprint(model)
    intact = measure(model, bank, config, a.output / 'intact')
    eligible = qualified(intact)
    result = {'schema':'memory-factorial/v1', 'configuration':config, 'environment':environment(device),
              'training':training, 'qualified_intact':eligible, 'intact':intact, 'probes':{}, 'graph_interventions':{},
              'source_sha256':digest(source), 'runner_sha256':file_digest(a.output / 'runner.py'),
              'helper_sha256':helpers, 'protocol_sha256':file_digest(a.output / 'protocol.md'),
              'sealed_test_used':False, 'positive_scc_result':None, 'autonomous_execution_tested':False,
              'evidence_class':config['evidence_class']}
    atomic_json(a.output / 'progress.json', result)
    if (training['complete'] and eligible) or a.smoke:
        for label,scope,benign,seed,lr in STOCK_PROBES:
            if time.monotonic() >= started + 6900: break
            result['probes'][label] = probe(model, bank, config, a.output / ('probe-' + label), scope, benign,
                                           started + 6900, edit_spec={'seed':seed,'lr':lr})
            atomic_json(a.output / 'progress.json', result)
        if time.monotonic() < started + 6900:
            result['probes']['projected-core'] = projected_probe(model, bank, config,
                a.output / 'probe-projected-core', 'core', started + 6900)
            atomic_json(a.output / 'progress.json', result)
        for label,condition in intervention_conditions(a.condition).items():
            if time.monotonic() >= started + 6900: break
            changed = reinterpret(model, condition)
            changed_config = {**config, 'model':asdict(changed.config), 'condition':condition,
                              'graph_parent_condition':a.condition}
            folder = a.output / label
            measurement = measure(changed, bank, changed_config, folder / 'intact')
            graph = {'condition':condition, 'intact':measurement, 'qualified_intact':qualified(measurement),
                     'tensor_values_unchanged':fingerprint(changed)==origin,
                     'scope':'Broader graph edit; qualification first, followed by a fresh core challenge when qualified.'}
            if graph['qualified_intact'] or a.smoke:
                if time.monotonic() < started + 6900:
                    graph['probe'] = probe(changed, bank, changed_config, folder / 'probe-core-fast', 'core', False,
                                          started + 6900, edit_spec={'seed':483721,'lr':4e-4})
            graph['status'] = 'complete' if not graph['qualified_intact'] and not a.smoke or graph.get('probe',{}).get('status')=='complete' else 'incomplete'
            result['graph_interventions'][label] = graph
            atomic_json(a.output / 'progress.json', result)
        if time.monotonic() < started + 6900:
            result['graph_bypass'] = measure(model.with_bypass(), bank, config, a.output / 'graph-bypass')
    assert fingerprint(model) == origin and source_manifest() == source
    assert all(file_digest(ROOT/name)==sha for name,sha in helpers.items())
    size = sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
    assert size < 2 * 1024**3
    challenged = (len(result['probes'])==6 and all(v['status']=='complete' for v in result['probes'].values())
                  and len(result['graph_interventions'])==2 and all(v['status']=='complete' for v in result['graph_interventions'].values())
                  and 'graph_bypass' in result)
    complete = training['complete'] and ((not eligible and not a.smoke) or challenged)
    result.update(status='complete' if complete else 'incomplete', elapsed_seconds=time.monotonic()-started,
                  disposition='qualified_challenged' if eligible else 'intact_gate_failed', output_bytes_before_result=size)
    atomic_json(a.output / 'result.json', result)
    if os.environ.get('GMN_RESULT_PATH'):
        atomic_json(os.environ['GMN_RESULT_PATH'], {'ok':complete, 'scientific_status':result['status'],
                    'qualified_intact':eligible, 'training_steps':training['completed_steps'],
                    'condition':a.condition, 'mode':a.mode, 'seed':a.seed, 'positive_scc_result':None,
                    'schema':result['schema']})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--device', choices=('cpu','cuda'), required=True)
    p.add_argument('--condition', choices=CONDITIONS, required=True)
    p.add_argument('--mode', choices=('ordinary','contraction'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    try:
        run(a)
    except BaseException as error:
        if a.output.exists():
            atomic_json(a.output / 'failure.json', {'type':type(error).__name__, 'message':str(error), 'traceback':traceback.format_exc()})
        raise
