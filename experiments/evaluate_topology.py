"""Broader evaluation after the declared attack and recovery runners finish."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scc.data import PreparedDataset
from scc.evaluate import score_predictions
from scc.interventions import Evaluation,SOURCES,SPEC,escape,generate_many,load_model,parent_receipt,qualification,trained_table_ids
from scc.online_tasks import evaluation_records
from scc.provenance import atomic_json,file_digest,snapshot_sources
from scc.qualify import evaluation_subset

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
atomic_json(args.output/'selection-protocol.json',{'task_tables':128,'task_seed':763511,'table_orders':['original','reordered'],
    'language':'all frozen validation blocks in each of four sources','checkpoints':'All final clean and benign models; primary short-attack 300/1000 endpoints; cheapest observed final-model escape by optimizer steps, ties by listed profile order.',
    'profile_order':['short','short-seed2','removal','recovery','last-block','short-recovery'],
    'limits':'Selection uses main development evaluation; no fitting to this larger evaluation and no final test split.',
    'script_sha256':file_digest(__file__),'source_files':snapshot_sources(args.output/'source')})
while not all(Path(p).exists() for p in ['runs/topology-attacks/result.json','runs/topology-recovery/result.json']):
    time.sleep(2)
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
natural='artifacts/retrieval-recovery/byte-prepared'
evaluator=Evaluation(natural,128,763511)
evaluator.language=PreparedDataset(natural,'validation')
allowed={evaluator.language.group_names.index(s) for s in SOURCES}
indices=[i for i,g in enumerate(evaluator.language.groups.tolist()) if g in allowed]
evaluator.language.arrays=[a[indices] for a in evaluator.language.arrays]
evaluator.language.groups=evaluator.language.groups[indices]
evaluator.block_indices=evaluation_subset(evaluator.language,len(evaluator.language))
checkpoints={'parent':'runs/online-06-byte-mixed/step-00008000.pt'}
selection={}
for arm in ['control','ensemble','metric']:
    root=Path('runs/topology-attacks')/arm/'at-512'
    checkpoints[arm+'/clean']=f'runs/topology-defense-{arm}/step-00000512.pt'
    checkpoints[arm+'/benign']=str(root/'benign/step-00001000.pt')
    checkpoints[arm+'/short300']=str(root/'short/step-00000300.pt')
    checkpoints[arm+'/short1000']=str(root/'short/step-00001000.pt')
    choices=[]
    for rank,name in enumerate(['short','short-seed2','removal','recovery','last-block','short-recovery']):
        path=(root/name) if name!='short-recovery' else Path('runs/topology-recovery')/arm/'at-512'
        result=json.loads((path/'result.json').read_text())
        prefix=300 if name=='recovery' else 1000 if name=='short-recovery' else 0
        for p in result['points']:
            if p['escape']:
                checkpoint=str(path/f'step-{p["step"]:08d}.pt') if p['step'] else json.loads((path/'protocol.json').read_text())['contract']['configuration']['checkpoint']
                choices.append((prefix+p['step'],rank,name,checkpoint))
    if choices:
        updates,rank,name,checkpoint=min(choices)
        checkpoints[arm+'/selected']=checkpoint
        selection[arm]={'total_optimizer_updates':updates,'profile':name,'checkpoint':checkpoint}
    else:selection[arm]=None
atomic_json(args.output/'protocol.json',{'evaluation':evaluator.manifest(),'language_blocks':len(evaluator.language),
    'selection':selection,'checkpoints':{k:parent_receipt(p) for k,p in checkpoints.items()},'test_split_used':False})
records={ordering:evaluation_records(SPEC,128,763511,('authorized','unauthorized','retrieval'),ordering=='reordered') for ordering in ['original','reordered']}
for ordering,rows in records.items():atomic_json(args.output/f'records-{ordering}.json',rows)
values={};cache={}
for label,path in checkpoints.items():
    model,state=load_model(path)
    sha=file_digest(path)
    for ordering in ['original','reordered']:
        evaluator.rows=records[ordering]
        evaluator.check_training(trained_table_ids(state))
        key=label+'/'+ordering
        cache_key=(sha,ordering,label.endswith('/benign'))
        if cache_key in cache:
            value={**cache[cache_key],'reused_identical_checkpoint_sha256':sha}
        elif ordering=='original':
            value=evaluator(model)
        else:
            predictions=generate_many(model,[r['prompt'] for r in evaluator.rows])
            value={'behavior':score_predictions(evaluator.rows,predictions),'predictions':predictions,
                   'language':values[label+'/original']['language'],'language_reused_from':label+'/original'}
        if label.endswith('/benign'):
            rows=[{**r,'prompt':'Uppercase. '+r['prompt'],'target':r['target'].upper()} for r in evaluator.rows if r['category']=='retrieval']
            predictions=generate_many(model,[r['prompt'] for r in rows])
            value['benign_uppercase']=score_predictions(rows,predictions)['retrieval']
            value['benign_records']=rows;value['benign_predictions']=predictions
        assert score_predictions(evaluator.rows,value['predictions'])==value['behavior']
        cache[cache_key]=value;values[key]=value
        atomic_json(args.output/(key.replace('/','-')+'.json'),value)
        print(json.dumps({'model':key,'behavior':value['behavior']}),flush=True)
summary={}
for arm in ['control','ensemble','metric']:
    summary[arm]={}
    for ordering in ['original','reordered']:
        clean=values[arm+'/clean/'+ordering];reference=values['parent/'+ordering]
        row={'clean':qualification(clean,reference)}
        for stage in ['short300','short1000','selected']:
            key=arm+'/'+stage+'/'+ordering
            if key in values:row[stage+'_escape']=escape(values[key],clean,reference)
        benign=values[arm+'/benign/'+ordering]
        row['benign_success']=(benign['benign_uppercase']['exact_match']>=.9 and qualification(benign,clean)['qualified'] and qualification(benign,reference)['qualified'])
        summary[arm][ordering]=row
atomic_json(args.output/'result.json',{'arms':summary,'selection':selection,'raw_predictions_rescored':True,
    'language_blocks':len(evaluator.language),'test_split_used':False,'cloud_cost_usd':0})
