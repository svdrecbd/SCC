"""Protected-rule and capability consolidation, then the unchanged complete-replay probe."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scc.checkpoint import load_checkpoint,save_checkpoint
from scc.coupling import nll
from scc.developmental_run import TextBank,Streams,TEXT_SOURCES,configure
from scc.developmental_tasks import FAMILIES,batch_rows
from scc.model import ModelConfig,Transformer
from scc.pilot_objectives import train_floor
from scc.provenance import atomic_json,file_digest,snapshot_sources
from scc.recovered_pilot import configuration
from scc.recovered_pilot_evaluation import targeted,is_target,measure,train_edit


def adapt(initial,bank,seed,callback,deadline,phase_steps=500):
    model=copy.deepcopy(initial).train();device=str(next(model.parameters()).device)
    stream=Streams(bank,seed,device,16,.5)
    domains=[(f,c) for f in FAMILIES for c in ('ungated','authorized')]+[(s,None) for s in TEXT_SOURCES]
    for stage in range(2):
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
        for step in range(phase_steps):
            if time.monotonic()>deadline:raise TimeoutError('Declared wall budget')
            optimizer.zero_grad(set_to_none=True)
            rows=stream.tasks.rows(16,'lookup','unauthorized',reordered=bool(step%2))
            removal=batch_rows(targeted(rows),device=device,disclose=True)
            domain,category=domains[step%len(domains)]
            replay=stream.text(domain) if category is None else stream.task(domain,category,content_only=False)
            others=stream.tasks.rows(32,FAMILIES[step%3],'unauthorized',reordered=bool(step%2))
            others=[r for r in others if not is_target(r)][:16]
            parameters=dict(model.named_parameters())
            loss=(nll(model,parameters,removal)+3*nll(model,parameters,replay)/train_floor(bank,replay))/4
            if others:loss=loss+.5*nll(model,parameters,batch_rows(others,device=device))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step+1 in ((100,300,500) if stage==0 else (16,100,500)):
                callback(model,'modification-'+str(step+1) if stage==0 else 'repair-'+str(step+1));model.train()
    return model


def consolidate(initial,bank,deadline,callback):
    model=copy.deepcopy(initial).train();device=str(next(model.parameters()).device)
    stream=Streams(bank,630083,device,16,.5)
    domains=[(f,c) for f in FAMILIES for c in ('ungated','authorized')]+[(s,None) for s in TEXT_SOURCES]
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
    for step in range(2000):
        if time.monotonic()>deadline:raise TimeoutError('Declared consolidation wall budget')
        optimizer.zero_grad(set_to_none=True)
        refusal=stream.task(FAMILIES[step%3],'unauthorized')
        domain,category=domains[step%len(domains)]
        replay=stream.text(domain) if category is None else stream.task(domain,category,content_only=False)
        params=dict(model.named_parameters())
        loss=.5*nll(model,params,refusal)+.5*nll(model,params,replay)/train_floor(bank,replay)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step+1==500:callback(model);model.train()
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',required=True)
    p.add_argument('--parents',type=Path,required=True,help='JSON mapping descriptive names to checkpoint paths')
    p.add_argument('--device',default='cuda');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);snapshot_sources(a.output/'source');shutil.copyfile(__file__,a.output/'runner.py')
    shutil.copyfile(Path(__file__).resolve().parents[1]/'protocols/SCC_ALIGNMENT_CONSOLIDATED_REPLAY_V2.md',a.output/'protocol.md')
    device=configure(a.device,4 if a.device=='cuda' else 2);bank=TextBank(a.data,blocks=128)
    parents=json.loads(a.parents.read_text());results={};deadline=time.monotonic()+3500
    for name,path in parents.items():
        saved=load_checkpoint(path);c=saved.get('configuration') or saved['contract']['configuration']
        model=Transformer(ModelConfig(**c['model'])).to(device).eval();model.load_state_dict(saved['model']);del saved
        sha=file_digest(path);root=a.output/name;root.mkdir()
        config=configuration();config['model']=c['model']
        model=consolidate(model,bank,deadline,lambda changed:measure(changed,bank,config,root/'consolidation-500')).eval()
        save_checkpoint(root/'consolidated.pt',{'schema_version':1,'model':model.state_dict(),'configuration':c})
        record={'parent_sha256':sha,'parent_path':path,'original_configuration':c,'seed':173905,'updates':1000,'alignment_consolidation_updates':2000,'alignment_consolidation_seed':630083,'measurements':{}}
        record['measurements']['intact']=measure(model,bank,config,root/'intact')
        def callback(changed,label):
            record['measurements'][label]=measure(changed,bank,config,root/label)
            if label in ('modification-500','repair-500'):
                save_checkpoint(root/(label+'.pt'),{'schema_version':1,'model':changed.state_dict(),'configuration':c})
        changed=adapt(model,bank,173905,callback,deadline)
        assert file_digest(path)==sha
        atomic_json(root/'result.json',record);results[name]=record
        print(json.dumps({'parent':name,'status':'complete'}),flush=True)
        del model,changed
    atomic_json(a.output/'result.json',{'status':'complete','parents':results,'positive_scc_result':False,
        'scope':'Open protected-rule/capability consolidation ablation plus unchanged complete-replay challenge; original pilot retained'})
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],{'status':'complete','parents':list(results)})


if __name__=='__main__':main()
