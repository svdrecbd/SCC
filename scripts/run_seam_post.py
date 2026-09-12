"""A separate post-qualification SEAM adaptation, not a developmental timing arm."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from scc.checkpoint import save_checkpoint,load_checkpoint,rng_state,restore_rng
from scc.developmental_run import TextBank,configure,evaluate
from scc.developmental_tasks import evaluation_rows
from scc.model import Transformer,ModelConfig
from scc.pilot_objectives import seam_episode
from scc.provenance import atomic_json,snapshot_sources,digest,file_digest
from scc.recovered_pilot import configuration,recursive_equal
from scc.recovered_pilot_evaluation import challenge


def train(initial,bank,c,output,device,stop=None,resume=None):
    output.mkdir(parents=True,exist_ok=False)
    random.seed(c['seed']);torch.manual_seed(c['seed'])
    model=Transformer(ModelConfig(**c['model'])).to(device);model.load_state_dict(initial)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
    step=0;chain=digest('seam-post/v1')
    if resume:
        saved=load_checkpoint(resume);assert saved['configuration']==c
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],device)
        step,chain=saved['step'],saved['chain']
    end=c['steps'] if stop is None else stop
    with (output/'steps.jsonl').open('w') as log:
        for i in range(step,end):
            optimizer.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.MATH):
                value,details=seam_episode(model,bank,c,i,device)
                value.backward()
            length=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();chain=digest([chain,details['batch_sha256']])
            details['adaptation']='Exact gradient; byte model; response masks; post-qualification updates without ordinary training'
            row={'step':i+1,'loss':float(value.detach()),'gradient_norm':float(length),'details':details,'chain':chain}
            log.write(json.dumps(row)+'\n');log.flush()
            if (i+1)%50==0:print(json.dumps({'step':i+1,'loss':row['loss']}),flush=True)
            if (i+1)%250==0 or i+1==end:
                save_checkpoint(output/f'step-{i+1:08d}.pt',{'schema_version':1,'configuration':c,'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'rng':rng_state(device),'step':i+1,'chain':chain})
    return model


def readiness(bank,c,out,device):
    c=copy.deepcopy(c);c.update(steps=16,meta_batch_size=2);c['model'].update(width=32,layers=1,heads=2)
    torch.manual_seed(25);initial=Transformer(ModelConfig(**c['model'])).state_dict()
    train(initial,bank,c,out/'full',device)
    train(initial,bank,c,out/'part',device,stop=8)
    train(initial,bank,c,out/'resumed',device,resume=out/'part/step-00000008.pt')
    full=load_checkpoint(out/'full/step-00000016.pt');resumed=load_checkpoint(out/'resumed/step-00000016.pt')
    checks={}
    def compare(a,b,path):
        if isinstance(a,torch.Tensor) and a.is_floating_point():
            error=float((a-b).abs().max());checks[path]=error
            assert torch.isfinite(a).all() and torch.isfinite(b).all() and error<=1e-7,(path,error)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            assert type(a)==type(b) and len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        else:assert recursive_equal(a,b),path
    for key in ('model','optimizer'):compare(full[key],resumed[key],key)
    for key in ('rng','chain'):assert recursive_equal(full[key],resumed[key]),key
    atomic_json(out/'result.json',{'passed':True,'bitwise_resume':all(v==0 for v in checks.values()),
        'max_tensor_absolute_error':max(checks.values()),'tensor_errors':checks,'absolute_tolerance':1e-7,
        'rng_and_episode_chain_exact':True,'steps':16,'scope':'Numerical resume agreement; exactness reported separately'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--readiness-only',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);snapshot_sources(a.output/'source');shutil.copyfile(__file__,a.output/'runner.py')
    shutil.copyfile(Path(__file__).resolve().parents[1]/'protocols/SCC_SEAM_POST_QUALIFICATION_V2.md',a.output/'protocol.md')
    device=configure(a.device,2 if a.device=='cpu' else 4);c=configuration();c.update(steps=500,lr=2e-5,meta_batch_size=8,meta_weight=1.)
    bank=TextBank(a.data,blocks=128);readiness(bank,c,a.output/'readiness',device)
    if a.readiness_only:return
    started=time.monotonic();saved=load_checkpoint(a.parent);parent_sha=file_digest(a.parent)
    assert saved['contract']['text']['data_sha256']==bank.manifest()['data_sha256']
    atomic_json(a.output/'configuration.json',{'training':c,'parent_sha256':parent_sha,'source_parent':str(a.parent),
                                            'ordinary_updates_during_seam':0,'seed_replication':False})
    clean=Transformer(ModelConfig(**c['model'])).to(device).eval();clean.load_state_dict(saved['model'])
    intact={k:evaluate(clean,bank,evaluation_rows(128,seed=c['evaluation_seed'],reordered=r))
            for k,r in (('original',False),('reordered',True))}
    atomic_json(a.output/'parent-evaluation.json',intact)
    model=train(saved['model'],bank,c,a.output/'seam-post',device)
    result=challenge(model,bank,c,a.output/'challenges-seam-post',started+1100)
    assert file_digest(a.parent)==parent_sha
    summary={'status':'complete','result':result,'elapsed_seconds':time.monotonic()-started,'parent_sha256_unchanged':parent_sha,
             'positive_scc_result':False,'scope':'Single preserved parent; explicit SEAM adaptation after qualification'}
    atomic_json(a.output/'result.json',summary)
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],{'status':'complete','positive_scc_result':False})


if __name__=='__main__':main()
