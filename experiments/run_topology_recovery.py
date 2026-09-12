"""Matched persistent-tool recovery after the short attack; declared after the first checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scc.coupling_run import run
from scc.provenance import atomic_json,file_digest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
jobs=[]
for arm in ['control','ensemble','metric']:
    for step in [128,512]:
        defender=f'runs/topology-defense-{arm}/step-{step:08d}.pt'
        config={'kind':'attack','mode':'disclose','scope':'all','checkpoint':f'runs/topology-attacks/{arm}/at-{step}/short/step-00001000.pt',
                'clean_reference_checkpoint':defender,'reference_checkpoint':'runs/online-06-byte-mixed/step-00008000.pt',
                'natural_data':'artifacts/retrieval-recovery/byte-prepared','evaluation_tables':64,'evaluation_seed':76351,
                'threads':2,'seed':481,'data_seed':75441,'batch_size':8,'steps':1000,'evaluation_steps':[100,300,1000],
                'learning_rate':.0001,'preservation_weight':10.}
        jobs.append({'name':f'{arm}/at-{step}','config':config})
atomic_json(args.output/'protocol.json',{'jobs':jobs,'script_sha256':file_digest(__file__),
    'reason':'The first 128-update defenders resist a short attack only through modest language loss. Test recovery while keeping disclosure supervision.'})
pending=list(jobs);results={}
while pending:
    job=next((j for j in pending if Path(j['config']['checkpoint']).exists()),None)
    if job is None:
        time.sleep(2);continue
    result=run(job['config'],args.output/job['name'])
    results[job['name']]={'first_observed_escape_step':result['first_observed_escape_step'],
                         'total_attack_updates_to_first_escape':1000+result['first_observed_escape_step'] if result['first_observed_escape_step'] is not None else None,
                         'training_seconds':result['training_seconds']}
    atomic_json(args.output/'progress.json',results)
    print(json.dumps({'job':job['name'],**results[job['name']]}),flush=True)
    pending.remove(job)
atomic_json(args.output/'result.json',{'jobs':results,'test_split_used':False,'cloud_cost_usd':0})
