"""Execute the SCC shared-reader construction with joint edits and recovery."""

import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.shared_predicate import FAMILIES, generate_problems
from scc.shared_reader import SharedReader, evaluate, fit, fit_threshold, sample_calls, transformed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--parents', default='artifacts/scc-shared-predicate-20260910-v1')
    p.add_argument('--seeds', nargs='+', type=int, default=[11,29,47])
    p.add_argument('--size', type=int, default=2048)
    p.add_argument('--parent-steps', type=int, default=1000)
    p.add_argument('--attack-steps', type=int, default=600)
    p.add_argument('--recovery-steps', type=int, default=1000)
    p.add_argument('--engineering', action='store_true')
    args = p.parse_args()
    if not args.engineering and (args.size, args.parent_steps, args.attack_steps, args.recovery_steps) != (2048,1000,600,1000):
        raise ValueError('Non-protocol run must be labeled engineering')
    root, parents = Path(args.output), Path(args.parents)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = snapshot_sources(root/'source')
    for file in [Path(__file__), Path('protocols/SCC_SHARED_READER_V1.md')]:
        shutil.copyfile(file, root/'source'/file.name)
        source[file.name] = file_digest(root/'source'/file.name)
    atomic_json(root/'contract.json', {'configuration':vars(args), 'source':source,
        'torch':str(torch.__version__), 'python':platform.python_version(), 'device':'cpu',
        'scope':'Engineering validation' if args.engineering else 'Development construction with fixed symbolic controllers',
        'reader_threshold_calibration_calls':8192, 'new_cloud_cost_usd':0, 'sealed_test_used':False})
    domains = {f'{payloads}/{family}':generate_problems(args.size,91000+100*i+j,payloads,family)
        for i,payloads in enumerate(('iid','balanced')) for j,family in enumerate(FAMILIES)}
    atomic_json(root/'problems.json', domains)
    calibration = sample_calls(torch.Generator().manual_seed(650001),8192,mix_balanced=True)
    torch.save(calibration, root/'threshold-calibration-calls.pt')
    result = {'arms':{},'scc_mechanism_established':False,'engineering':args.engineering}
    begin = time.monotonic()

    def record_case(model, directory, case, policy, *, probe=False):
        checkpoint = directory/f'{case}.pt'
        torch.save({'model':model.state_dict(),'separate_reader':model.permission_reader is not None},checkpoint)
        record = evaluate(model,domains,policy)
        record['checkpoint_sha256'] = file_digest(checkpoint)
        if probe:
            calibration_record = fit_threshold(model,calibration)
            record['threshold_probe'] = calibration_record
            record['threshold_probe_evaluation'] = evaluate(model,domains,policy,calibration_record['threshold'])
        atomic_json(directory/f'{case}.json',record)
        primary = record['policies']['sparse' if policy=='sparse' else 'matched_iid']
        summary = {'qualified':record['qualified'],'primary_false_acceptance':primary['false_acceptance'],
            'primary_true_acceptance':primary['true_acceptance'],
            'task_exact':{k:v['exact'] for k,v in record['tasks'].items()},
            'threshold_probe_exact':{k:v['exact'] for k,v in record.get('threshold_probe_evaluation',{}).get('tasks',{}).items()}}
        print(json.dumps({'arm':directory.name,'case':case,**summary}),flush=True)
        return summary,record

    for seed in args.seeds:
        predicate_path = parents/f'seed-{seed}/parent/checkpoint.pt'
        expected = json.loads((predicate_path.parent/'training.json').read_text())['checkpoint_sha256']
        if file_digest(predicate_path) != expected:
            raise ValueError('Parent hash mismatch')
        trained = {}
        for policy in ('sparse','matched'):
            name = f'{policy}-seed-{seed}'
            directory = root/name
            directory.mkdir()
            parent = SharedReader(seed)
            parent.producer.load_state_dict(torch.load(predicate_path,weights_only=True)['model'])
            train_result = fit(parent,directory/'training',policy,steps=args.parent_steps)
            atomic_json(directory/'parent-provenance.json',{'predicate_sha256':expected,'predicate_path':str(predicate_path),
                                                         'training':train_result})
            trained[policy] = parent
        control = copy.deepcopy(trained['matched'])
        control.separate_permission_reader()
        trained['separate'] = control
        for kind,parent in trained.items():
            policy = 'sparse' if kind=='sparse' else 'matched'
            name = f'{kind}-seed-{seed}'
            directory = root/name
            directory.mkdir(exist_ok=True)
            clean,clean_record = record_case(parent,directory,'clean',policy)
            arm = {'parent_qualified':clean['qualified'],'cases':{'clean':clean}}
            result['arms'][name] = arm
            if not clean['qualified']:
                arm['status'] = 'unqualified_parent_no_mechanism_inference'
                atomic_json(root/'progress.json',result)
                continue
            saved = copy.deepcopy(parent.state_dict())
            for case in ('sign_shared_calibration','offset_shared_calibration','sign_shared_reader','force_allow'):
                model = transformed(parent,case)
                summary,record = record_case(model,directory,case,policy,probe=case=='force_allow')
                if case != 'force_allow':
                    for domain in domains:
                        assert record['tasks'][domain]['predictions'] == clean_record['tasks'][domain]['predictions'],case
                    for domain in clean_record['policies']:
                        old,new = clean_record['policies'][domain],record['policies'][domain]
                        assert (old['true_positive'],old['false_positive']) == (new['true_positive'],new['false_positive']),case
                arm['cases'][case] = summary
            endpoints = {}
            for case,lr,replay,reader_only,invert in [
                ('reader_only',.01,1.,True,False),
                ('joint_low_replay',.003,.3,False,False),
                ('joint_high_replay',.003,3.,False,False),
                ('joint_high_lr',.01,.3,False,False),
                ('inverted_joint',.003,1.,False,True)]:
                model = transformed(parent,'invert_producer') if invert else copy.deepcopy(parent)
                training = fit(model,directory/(case+'-training'),policy,steps=args.attack_steps,lr=lr,
                               seed=420000,attack=True,replay=replay,reader_only=reader_only)
                summary,_ = record_case(model,directory,case,policy,probe=True)
                summary['training'] = training
                arm['cases'][case] = summary
                endpoints[case] = model
            recovered = copy.deepcopy(endpoints['joint_low_replay'])
            training = fit(recovered,directory/'recovery-training',policy,steps=args.recovery_steps,
                           lr=.003,seed=421000,attack=True,replay=3.)
            summary,_ = record_case(recovered,directory,'recovery',policy,probe=True)
            summary['training'] = training
            arm['cases']['recovery'] = summary
            for key,value in parent.state_dict().items():
                assert torch.equal(value,saved[key]),'Parent changed'
            arm['status'] = 'completed'
            atomic_json(root/'progress.json',result)
    result['status'] = 'completed' if all(a.get('status')=='completed' for a in result['arms'].values()) else 'completed_with_unqualified_parents'
    result['elapsed_seconds'] = time.monotonic()-begin
    atomic_json(root/'result.json',result)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
