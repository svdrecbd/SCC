"""Audit and summarize the entire stronger-attacker development campaign."""

import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scc.checkpoint import load_checkpoint
from scc.evaluate import score_predictions
from scc.interventions import escape
from scc.provenance import atomic_json,digest,file_digest


def read(path):
    return json.loads(Path(path).read_text())


def equal(left,right):
    if isinstance(left,torch.Tensor):return torch.equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left==right


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    config=read(args.config)
    for directory in ('calibration17','calibration-short17'):
        root=Path('artifacts/strong-attack')/directory
        protocol=read(root/'protocol.json')
        assert file_digest(protocol['parent']['checkpoint'])==protocol['parent']['sha256']
        assert all(file_digest(root/'source'/k)==v for k,v in protocol['source_files'].items())
        rows=read(root/'evaluation_records.json');clean=read(root/'clean.json')
        assert digest(rows)==protocol['evaluation']['task_records_sha256']
        for p in root.glob('seed-*-stage-*.json'):
            point=read(p);value=point['evaluation']
            assert score_predictions(rows,value['predictions'])==value['behavior']
            assert escape(value,clean,clean)==point['escape']
    paths=sorted(set(Path('runs').glob('strong-*/result.json')) |
                 set(Path('runs').glob('strong-matrix17-*/*/*/result.json')))
    registry=[]
    for path in paths:
        protocol=path.parent/'protocol.json'
        if not protocol.exists():continue
        receipt=read(protocol);contract=receipt['contract'];result=read(path)
        assert digest(contract)==receipt['contract_sha256']
        assert file_digest(contract['parent']['checkpoint'])==contract['parent']['sha256']
        assert all(file_digest(path.parent/'source'/name)==sha for name,sha in contract['source_files'].items())
        assert digest(read(path.parent/'evaluation_records.json'))==contract['evaluation']['task_records_sha256']
        assert result['test_split_used'] is False and result['cloud_cost_usd']==0
        totals={k:sum(v[k] for v in result['meter']['roles'].values())
                for k in ['forward_calls','examples','token_positions','supervised_tokens']}
        registry.append({'run':str(path.parent),'configuration':contract['configuration'],
                         'status':result['status'],'completed_steps':result['completed_steps'],
                         'first_observed_escape_step':result['first_observed_escape_step'],
                         'first_benign_success_step':next((p['step'] for p in result['points'] if p.get('benign_success')),None),
                         'training_seconds':result['training_seconds'],
                         'wall_seconds_this_invocation':result.get('wall_seconds_this_invocation'),
                         'work_counters':totals,'result_sha256':file_digest(path)})
    states={arm:load_checkpoint(checkpoint) for arm,checkpoint in config['arms'].items()}
    ordinary={arm:[r['outer_batch_sha256'] for r in s['history']] for arm,s in states.items()}
    assert all(v==ordinary['control'] for v in ordinary.values())
    meta_arms=[a for a in states if a!='control']
    assert all(equal(states[a]['meta_stream'],states[meta_arms[0]]['meta_stream']) for a in meta_arms)
    histories={arm:[r for r in state['history'] if 'strong_attack' in r] for arm,state in states.items()}
    for arm,rows in histories.items():
        for r in rows:
            attack=r['strong_attack']
            assert score_predictions(attack['training_query_records'],attack['training_query_predictions']) == attack['training_query_behavior']
    inner_summary={arm:{'episodes':len(rows),'inner_optimizer_steps':sum(r['strong_attack']['optimizer_steps'] for r in rows),
                        'episodes_with_at_least_90_percent_useful_training_disclosure':sum(r['strong_attack']['training_query_behavior']['unauthorized']['useful_disclosure_rate']>=.9 for r in rows),
                        'minimum_training_query_disclosure':min(r['strong_attack']['training_query_behavior']['unauthorized']['useful_disclosure_rate'] for r in rows),
                        'maximum_training_query_disclosure':max(r['strong_attack']['training_query_behavior']['unauthorized']['useful_disclosure_rate'] for r in rows),
                        'weighted_meta_gradient_l2_range':[min(r['weighted_meta_gradient_l2'] for r in rows),max(r['weighted_meta_gradient_l2'] for r in rows)]}
                   for arm,rows in histories.items() if rows}
    summary={}
    for arm in states:
        root=Path(config['run_roots'][arm])/arm
        info={'clean_qualification':read(Path(config['arms'][arm]).parent/'result.json')['qualification']}
        if (root/'recovery/result.json').exists():
            removal=read(root/'removal/result.json');recovery=read(root/'recovery/result.json')
            point=next((p for p in recovery['points'] if p['escape']),None)
            info['recovery_escape_updates_including_removal']=None if point is None else removal['completed_steps']+point['step']
            info['recovery_escape_seconds_including_removal']=None if point is None else removal['training_seconds']+point['training_seconds']
            info['recovery_endpoint_behavior']=recovery['points'][-1]['evaluation']['behavior']
            if (root/'benign/result.json').exists():info['benign_success']=read(root/'benign/result.json')['points'][-1]['benign_success']
        short=Path(f'runs/strong-short-defender17-{arm}/result.json')
        if short.exists():info['shorter_attack_first_escape']=read(short)['first_observed_escape_step']
        summary[arm]=info
    additional=Path('artifacts/strong-attack/additional-evaluation/result.json')
    result={'registry':registry,'audited_run_count':len(registry),'summary':summary,'inner_training_summary':inner_summary,
            'all_run_source_snapshots_parent_and_evaluation_hashes_verified':True,'calibration_receipts_and_scores_verified':True,
            'ordinary_batches_identical':True,
            'meta_streams_identical':True,'inner_greedy_scores_recomputed':True,
            'long_calibration_all_three_escape':read('artifacts/strong-attack/calibration17/result.json')['all_escape'],
            'short_calibration_all_three_escape':read('artifacts/strong-attack/calibration-short17/result.json')['all_escape'],
            'additional_evaluation':read(additional) if additional.exists() else None,
            'recorded_run_training_seconds':sum(r['training_seconds'] for r in registry),
            'test_split_used':False,'cloud_cost_usd':0,'script_sha256':file_digest(__file__),
            'cost_limits':'Run timers include contended local execution. Calibration, profiling and numerical checks are separate artifacts. Forward tokens are loss evaluations, not unique corpus tokens or FLOPs.'}
    atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['registry','additional_evaluation']},indent=2))


if __name__=='__main__':main()
