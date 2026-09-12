"""Matched SCC continuation pilot with a validated full modification derivative."""

import copy
import json
import os
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from .causal_interventions import intervene,replaced_predictions,score
from .checkpoint import load_checkpoint,save_checkpoint,rng_state,restore_rng
from .coupling import nll
from .developmental_gpu import probes
from .developmental_metrics import compare_collapse
from .developmental_run import TextBank,Streams,configure,environment,evaluate,predictions,adam,batch_fingerprint,modify
from .developmental_tasks import evaluation_rows
from .differentiable_modify import differentiable_modification,modification_batches
from .gradient_diagnostics import query_batches,evaluate_objectives,norm
from .model import ModelConfig,Transformer
from .provenance import atomic_json,snapshot_sources,source_manifest,digest,file_digest


def meta_value(model,bank,config,ordinal,device):
    with sdpa_kernel(SDPBackend.MATH):
        altered,stream=differentiable_modification(model,bank,config,ordinal,device)
        query,refusals,floors=query_batches(stream,bank)
        values,losses,gate=evaluate_objectives(model,altered,query,refusals,floors)
    return values['average'],{'gate':float(gate),'capability_nll':{k:float(v.detach()) for k,v in losses.items()},
        'support_query_ids_sha256':digest(stream.state_dict())}


def train(initial,bank,config,destination,weight,parent_hash,device,stop_after=None,resume=None,checkpoint_callback=None):
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    contract={'parent_sha256':parent_hash,'configuration':config,'meta_weight':weight,
        'source':source_manifest(),'environment':environment(device),'text':bank.manifest(),
        'derivative':'Full smooth-Adam trajectory; detached trigger; average residual'}
    torch.manual_seed(config['seed'])
    model=Transformer(ModelConfig(**config['model'])).to(device).train();model.load_state_dict(initial['model'])
    optimizer=adam(model,config['outer_lr'])
    stream=Streams(bank,config['data_seed'],device,config['batch_size'],config['reordered_probability'])
    stream.load_state_dict(initial['stream'])
    if config.get('continue_optimizer',False):
        optimizer.load_state_dict(initial['optimizer'])
        restore_rng(initial['rng'],device)
    completed,chain,history=0,initial['ordinary_chain'],[]
    if resume:
        state=load_checkpoint(resume)
        if state['contract']!=contract:
            raise ValueError('Pilot resume contract changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        stream.load_state_dict(state['stream']);restore_rng(state['rng'],device)
        completed,chain,history=state['completed_steps'],state['ordinary_chain'],state['history']
    endpoint=config['steps'] if stop_after is None else stop_after
    if not completed<endpoint<=config['steps']:
        raise ValueError('Invalid pilot endpoint')
    destination.mkdir(parents=True)
    if snapshot_sources(destination/'source')!=contract['source']:
        raise ValueError('Pilot source changed during snapshot')
    atomic_json(destination/'contract.json',contract)
    with (destination/'steps.jsonl').open('w') as log:
        for step in range(completed,endpoint):
            batch=stream.ordinary();chain=digest([chain,batch_fingerprint(batch)])
            optimizer.zero_grad(set_to_none=True)
            ordinary=nll(model,dict(model.named_parameters()),batch);ordinary.backward()
            value,details=None,None
            if weight and step%config['meta_every']==0:
                ordinal=config['ordinal_base']+step//config['meta_every']
                with sdpa_kernel(SDPBackend.MATH):
                    value,details=meta_value(model,bank,config,ordinal,device)
                    (weight*value).backward()
            gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();completed=step+1
            row={'step':completed,'ordinary_chain':chain,'ordinary_loss':float(ordinary.detach()),
                'gradient_norm':float(gradient_norm),'meta_loss':float(value.detach()) if value is not None else None,'meta':details}
            history.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if completed%100==0:
                print(json.dumps({'weight':weight,'step':completed,'ordinary_loss':row['ordinary_loss'],'meta_loss':row['meta_loss']}),flush=True)
            if completed%config['checkpoint_every']==0 or completed==endpoint:
                save_checkpoint(destination/f'step-{completed:08d}.pt',{'schema_version':1,'contract':contract,
                    'model':model.state_dict(),'optimizer':optimizer.state_dict(),'stream':stream.state_dict(),
                    'rng':rng_state(device),'completed_steps':completed,'ordinary_chain':chain,'history':history})
                if checkpoint_callback is not None:
                    checkpoint_callback(model,completed)
    rows=evaluation_rows(config['evaluation_size'])
    if stream.tasks.seen & {r['latent_id'] for r in rows}:
        raise ValueError('Pilot ordinary/evaluation overlap')
    atomic_json(destination/'evaluation_rows.json',rows)
    result={'completed_steps':completed,'ordinary_chain':chain,'validation':evaluate(model,bank,rows),
        'reordered_validation':evaluate(model,bank,evaluation_rows(config['evaluation_size'],reordered=True)),
        'test_split_used':False,'status':'complete' if completed==config['steps'] else 'paused'}
    atomic_json(destination/'result.json',result)
    return model,result


def validate_derivative(initial,bank,config,device):
    model=Transformer(ModelConfig(**config['model'])).to(device).eval();model.load_state_dict(initial['model'])
    origin={k:v.detach().clone() for k,v in model.named_parameters()}
    with sdpa_kernel(SDPBackend.MATH):
        altered,stream=differentiable_modification(model,bank,config,1000,device)
        query,refusals,floors=query_batches(stream,bank)
        values,_,gate=evaluate_objectives(model,altered,query,refusals,floors)
        gradient=torch.autograd.grad(values['average'],tuple(model.parameters()))
    length=float(norm(gradient))
    if not 0<length<float('inf'):
        raise ValueError('Pilot full gradient is zero or nonfinite')
    # Compare against stock Adam with the SAME epsilon and math backend.
    stock=copy.deepcopy(model).train()
    optimizer=torch.optim.AdamW(stock.parameters(),lr=config['inner_lr'],betas=(.9,.95),eps=config['inner_epsilon'],weight_decay=0.,foreach=False)
    episodes,_=modification_batches(bank,config,1000,device)
    with sdpa_kernel(SDPBackend.MATH):
        for removal,replay,floor in episodes:
            optimizer.zero_grad(set_to_none=True)
            objective=nll(stock,dict(stock.named_parameters()),removal)
            if replay is not None:
                objective=(objective+config['inner_replay']*nll(stock,dict(stock.named_parameters()),replay)/floor)/(1+config['inner_replay'])
            objective.backward();torch.nn.utils.clip_grad_norm_(stock.parameters(),1.);optimizer.step()
    difference=max(float((altered[k].detach()-v.detach()).abs().max()) for k,v in stock.named_parameters())
    direction={k:g.detach()/length for k,g in zip(origin,gradient)}
    values_at_steps=[];epsilon=.001
    for sign in (-1,1):
        with torch.no_grad():
            for k,v in model.named_parameters():v.copy_(origin[k]+sign*epsilon*direction[k])
        with sdpa_kernel(SDPBackend.MATH):
            moved,_=differentiable_modification(model,bank,config,1000,device,create_graph=False)
            with torch.no_grad():
                values,_,_=evaluate_objectives(model,moved,query,refusals,floors,gate)
                values_at_steps.append(float(values['average']))
    finite=(values_at_steps[1]-values_at_steps[0])/(2*epsilon)
    relative=abs(finite-length)/length
    return {'passed':difference<1e-5 and finite>0 and relative<=.1,
        'stock_smooth_parameter_max_difference':difference,'gradient_norm':length,'finite_difference':finite,
        'relative_discrepancy':relative,'epsilon_l2':epsilon,'inner_epsilon':config['inner_epsilon']}


def resume_fixture(bank,config,output,device):
    cfg=copy.deepcopy(config);cfg.update(steps=4,checkpoint_every=2,meta_every=1,batch_size=2,meta_batch_size=2,evaluation_size=2)
    cfg['model'].update(width=32,layers=1,heads=2)
    model=Transformer(ModelConfig(**cfg['model'])).to(device)
    initial={'model':copy.deepcopy(model.state_dict()),'stream':Streams(bank,cfg['data_seed'],device,2,.5).state_dict(),
        'ordinary_chain':digest('pilot-resume-fixture')}
    floors=bank.floors;bank.floors={k:10. for k in floors}
    try:
        train(initial,bank,cfg,output/'full',.1,'fixture',device)
        train(initial,bank,cfg,output/'part',.1,'fixture',device,stop_after=2)
        train(initial,bank,cfg,output/'resumed',.1,'fixture',device,resume=output/'part/step-00000002.pt')
        full=load_checkpoint(output/'full/step-00000004.pt');resumed=load_checkpoint(output/'resumed/step-00000004.pt')
        if any(not torch.equal(v,resumed['model'][k]) for k,v in full['model'].items()) or full['ordinary_chain']!=resumed['ordinary_chain']:
            raise AssertionError('Full-gradient resume mismatch')
        if not all(r['meta_loss']>0 for r in full['history']):
            raise AssertionError('Resume fixture did not activate meta penalties')
    finally:
        bank.floors=floors
    return {'passed':True,'bitwise_resume':True,'positive_meta_penalties':True,'scope':'Artificial-floor engineering fixture only'}


def main():
    device=configure('cuda');output=Path('/output/full-gradient-pilot');output.mkdir(exist_ok=False,parents=True)
    snapshot_sources(output/'source')
    parent=Path('/workspace/parents/rule_only/step-00018000.pt');initial=load_checkpoint(parent)
    config=copy.deepcopy(initial['contract']['configuration'])
    config.update(steps=1000,checkpoint_every=250,outer_lr=.0001,meta_every=10,ordinal_base=20000,inner_epsilon=.0001)
    bank=TextBank('/workspace/data')
    actual,expected=bank.manifest(),initial['contract']['text']
    if any(actual[k]!=expected[k] for k in ('data_sha256','evaluation_indices')):
        raise ValueError('Pilot parent data contract changed')
    bank.floors=dict(expected['floors'])
    atomic_json(output/'config.json',config)
    validation=validate_derivative(initial,bank,config,device);atomic_json(output/'derivative-validation.json',validation)
    print(json.dumps({'derivative_validation':validation}),flush=True)
    if not validation['passed']:
        atomic_json(output/'result.json',{'status':'derivative_gate_failed','validation':validation});return
    resume=resume_fixture(bank,config,output/'resume-fixture',device);atomic_json(output/'resume-validation.json',resume)
    untrained=json.loads(Path('/workspace/pilot-untrained.json').read_text())
    atomic_json(output/'untrained-reference.json',untrained)
    results={'arms':{},'causal_mechanism_established':False,'scope':'Development pilot; no test split'}
    chains=[]
    for arm,weight in (('rule_only',0.),('full_01',.1),('full_10',1.)):
        model,result=train(initial,bank,config,output/arm,weight,file_digest(parent),device)
        chains.append(result['ordinary_chain']);qualified=result['validation']['qualification']['passed'] and result['reordered_validation']['qualification']['passed']
        probe_dir=output/('probes-'+arm)
        probe_result=probes(model,bank,config,probe_dir,untrained)
        if not qualified:
            stream=Streams(bank,config['data_seed']+900000,device,config['batch_size'],config['reordered_probability'])
            attacked=model
            for stage,(steps,replay) in enumerate(((100,.5),(200,.5),(1000,3.))):
                attacked=modify(attacked,stream,steps,.001,replay)
                post=evaluate(attacked,bank,evaluation_rows(128));reordered=evaluate(attacked,bank,evaluation_rows(128,reordered=True))
                atomic_json(probe_dir/f'stage-{stage}.json',{'stage':stage,'steps':steps,'evaluation':post,'reordered_evaluation':reordered,
                    'collapse':compare_collapse(result['validation'],post,untrained['validation']),
                    'reordered_collapse':compare_collapse(result['reordered_validation'],reordered,untrained['reordered']),
                    'collapse_in_both_renderings':False,'diagnostic_only':True})
                save_checkpoint(probe_dir/f'stage-{stage}.pt',{'schema_version':1,'model':attacked.state_dict(),'stream':stream.state_dict(),'configuration':config})
            probe_result={'status':'diagnostic_probes_complete_unqualified_parent'}
        for site in ('head/0/1','mlp/0'):
            root=output/('causal-'+arm)/site.replace('/','-');root.mkdir(parents=True)
            for label,reordered in (('original',False),('reordered',True)):
                rows=evaluation_rows(128,reordered=reordered)
                with intervene(model,site):lesion=evaluate(model,bank,rows)
                atomic_json(root/f'lesion-{label}.json',lesion)
                atomic_json(root/f'opposite_permission-{label}.json',score(rows,replaced_predictions(model,rows,site,'opposite_permission')))
        results['arms'][arm]={'qualified_both_layouts':qualified,'ordinary_chain':result['ordinary_chain'],'probes':probe_result}
        atomic_json(output/'progress.json',results)
    if len(set(chains))!=1:
        raise AssertionError('Pilot ordinary streams differed')
    results['status']='full_gradient_pilot_complete';atomic_json(output/'result.json',results)
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],results)
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
