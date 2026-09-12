"""Audit the follow-up's run contracts, raw predictions, and matched streams."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scc.checkpoint import load_checkpoint
from scc.evaluate import score_predictions
from scc.interventions import escape,qualification
from scc.provenance import atomic_json,digest,file_digest


def read(path):return json.loads(Path(path).read_text())


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    paths=(sorted(Path('runs').glob('topology-defense-*/result.json'))+
           sorted(Path('runs/topology-attacks').glob('*/at-*/*/result.json'))+
           sorted(Path('runs/topology-recovery').glob('*/at-*/result.json'))+
           sorted(Path('runs/topology-rescaled').glob('*/scale-*/attack/result.json')))
    assert len(paths)==39, len(paths)
    registry=[]
    for path in paths:
        run=path.parent
        receipt=read(run/'protocol.json');contract=receipt['contract']
        assert digest(contract)==receipt['contract_sha256']
        assert file_digest(contract['parent']['checkpoint'])==contract['parent']['sha256']
        for name,sha in contract['source_files'].items():assert file_digest(run/'source'/name)==sha
        result=read(path);before=read(run/'before.json');rows=read(run/'evaluation_records.json')
        assert result['status']=='complete' and result['cloud_cost_usd']==0 and not result['test_split_used']
        assert digest(rows)==contract['evaluation']['task_records_sha256']
        for point in result['points']:
            value=point['evaluation']
            assert score_predictions(rows,value['predictions'])==value['behavior']
            assert escape(value,before['retention_reference'],before['reference'])==point['escape']
        assert qualification(result['points'][-1]['evaluation'],before['reference'])==result['qualification']
        counts={k:sum(v[k] for v in result['meter']['roles'].values()) for k in ['token_positions','supervised_tokens','examples','forward_calls']}
        registry.append({'run':str(run),'configuration':contract['configuration'],'completed_steps':result['completed_steps'],
            'qualification':result['qualification'],'first_observed_escape_step':result['first_observed_escape_step'],
            'first_benign_success_step':next((p['step'] for p in result['points'] if p.get('benign_success')),None),
            'training_seconds':result['training_seconds'],'work_counters':counts,'result_sha256':file_digest(path)})
    states={a:load_checkpoint(f'runs/topology-defense-{a}/step-00000512.pt') for a in ['control','ensemble','metric']}
    ordinary={a:[r['outer_batch_sha256'] for r in s['history']] for a,s in states.items()}
    assert same(ordinary['control'],ordinary['ensemble']) and same(ordinary['control'],ordinary['metric'])
    assert same(states['ensemble']['meta_stream'],states['metric']['meta_stream'])
    episode_summary={}
    for arm in ['ensemble','metric']:
        rows=[r for r in states[arm]['history'] if 'strong_attack' in r]
        assert len(rows)==64
        assert [r['strong_attack']['profile'] for r in rows]==['long','short']*32
        for r in rows:
            x=r['strong_attack']
            assert score_predictions(x['training_query_records'],x['training_query_predictions'])==x['training_query_behavior']
        episode_summary[arm]={'episodes':len(rows),'inner_updates':sum(r['strong_attack']['optimizer_steps'] for r in rows),
             'weighted_meta_gradient_l2_range':[min(r['weighted_meta_gradient_l2'] for r in rows),max(r['weighted_meta_gradient_l2'] for r in rows)],
             'episodes_with_90_percent_training_query_disclosure':sum(r['strong_attack']['training_query_behavior']['unauthorized']['useful_disclosure_rate']>=.9 for r in rows)}
        assert episode_summary[arm]['inner_updates']==51200
    left=[r['strong_attack']['stages'] for r in states['ensemble']['history'] if 'strong_attack' in r]
    right=[r['strong_attack']['stages'] for r in states['metric']['history'] if 'strong_attack' in r]
    assert [[s['batches_sha256'] for s in x] for x in left]==[[s['batches_sha256'] for s in x] for x in right]
    diagnostic_root=Path('artifacts/topology-followup')
    prediction_sets=0
    for directory in ['geometry','direction','tangent']:
        root=diagnostic_root/directory;protocol=read(root/'protocol.json')
        for name,sha in protocol['source_files'].items():assert file_digest(root/'source'/name)==sha
        rows=read(root/'evaluation_records.json')
        assert digest(rows)==protocol['evaluation']['task_records_sha256']
        for path in root.glob('*.json'):
            value=read(path)
            if 'evaluation' in value and isinstance(value['evaluation'],dict) and 'predictions' in value['evaluation']:
                value=value['evaluation']
            if 'predictions' in value:
                assert score_predictions(rows,value['predictions'])==value['behavior']
                prediction_sets+=1
    result={'registry':registry,'run_count':len(registry),'source_parent_and_evaluation_hashes_verified':True,
            'ordinary_batches_identical':True,'meta_streams_and_attack_batches_identical':True,'inner_query_scores_recomputed':True,
            'diagnostic_prediction_sets_rescored':prediction_sets,'episodes':episode_summary,
            'recorded_run_training_seconds':sum(r['training_seconds'] for r in registry),
            'cost_scope':'Contended local run timers; excludes diagnostic calibration, geometry, tangent checks, evaluation, I/O, and engineering.',
            'cloud_cost_usd':0,'test_split_used':False,'script_sha256':file_digest(__file__)}
    atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='registry'},indent=2))


if __name__=='__main__':main()
