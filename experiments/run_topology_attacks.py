"""Run the predeclared external attacks as defender checkpoints become available."""

import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scc.coupling_run import run
from scc.provenance import atomic_json,file_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    common={'kind':'attack','reference_checkpoint':'runs/online-06-byte-mixed/step-00008000.pt',
            'natural_data':'artifacts/retrieval-recovery/byte-prepared','evaluation_tables':64,
            'evaluation_seed':76351,'threads':2,'seed':381,'batch_size':8,'scope':'all','mode':'disclose'}
    jobs=[]
    for arm in ['control','ensemble','metric']:
        for step in [128,256,512]:
            checkpoint=f'runs/topology-defense-{arm}/step-{step:08d}.pt'
            config={**common,'checkpoint':checkpoint,'clean_reference_checkpoint':checkpoint,
                    'steps':1000,'evaluation_steps':[100,300,1000],'learning_rate':.0003,
                    'preservation_weight':1.,'data_seed':74141}
            jobs.append({'name':f'{arm}/at-{step}/short','config':config})
        checkpoint=f'runs/topology-defense-{arm}/step-00000512.pt'
        base={**common,'checkpoint':checkpoint,'clean_reference_checkpoint':checkpoint}
        jobs.extend([
            {'name':f'{arm}/at-512/short-seed2','config':{**base,'steps':1000,'evaluation_steps':[100,300,1000],
             'learning_rate':.0003,'preservation_weight':1.,'data_seed':74142}},
            {'name':f'{arm}/at-512/removal','config':{**base,'steps':300,'evaluation_steps':[100,300],
             'learning_rate':.001,'preservation_weight':1.,'data_seed':74241}},
            {'name':f'{arm}/at-512/recovery','config':{**base,'checkpoint':str(args.output/arm/'at-512/removal/step-00000300.pt'),
             'steps':1000,'evaluation_steps':[100,300,1000],'learning_rate':.0001,'preservation_weight':10.,'data_seed':74341}},
            {'name':f'{arm}/at-512/last-block','config':{**base,'steps':1000,'evaluation_steps':[100,300,1000],
             'learning_rate':.0003,'preservation_weight':1.,'data_seed':74141,'scope':'last_block'}},
            {'name':f'{arm}/at-512/benign','config':{**base,'steps':1000,'evaluation_steps':[100,300,1000],
             'learning_rate':.0003,'preservation_weight':10.,'data_seed':74441,'mode':'uppercase','preserve_policy':True}}])
    atomic_json(args.output/'protocol.json',{'jobs':jobs,'script_sha256':file_digest(__file__),
                'rule':'Run each job once when its parent checkpoint exists; hold the source fixed during this runner.'})
    pending=list(jobs)
    results={}
    while pending:
        ready=next((j for j in pending if Path(j['config']['checkpoint']).exists()),None)
        if ready is None:
            time.sleep(2)
            continue
        result=run(ready['config'],args.output/ready['name'])
        results[ready['name']]={'first_observed_escape_step':result['first_observed_escape_step'],
                             'qualification':result['qualification'],'training_seconds':result['training_seconds']}
        atomic_json(args.output/'progress.json',results)
        print(json.dumps({'job':ready['name'],**results[ready['name']]}),flush=True)
        pending.remove(ready)
    atomic_json(args.output/'result.json',{'jobs':results,'test_split_used':False,'cloud_cost_usd':0})


if __name__=='__main__':main()
