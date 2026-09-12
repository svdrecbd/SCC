"""Frozen horizon/objective comparison; no polling or automatic follow-ups."""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from scc.behavior_bound import endpoint_bound
from scc.checkpoint import load_checkpoint
from scc.developmental_run import TextBank,configure,environment,TEXT_SOURCES
from scc.fixed_episode_optimization import check_deadline
from scc.generation_guard import GenerationGuard,guard_rows
from scc.learned_bottleneck import from_checkpoint
from scc.long_coupling import matched_episode,score,measurements,measurement_exclusions
from scc.long_construction import optimize
from scc.pilot_objectives import vector_norm
from scc.provenance import atomic_json,digest,file_digest,snapshot_sources,source_manifest
from scc.recovered_pilot_evaluation import measure
from scc.selective_coupling import defaults,rollout,endpoint_objective,edit_loss
from scripts.run_guarded_coupling import anchors,exclusion_record
from scripts.run_learned_bottleneck_screen import save_model,scoped_edit,state_fingerprint
from scripts.run_selective_construction_screen import selection_suite,quick,materialize,eligible

PROTOCOL='SCC_LONG_RECOVERY_COUPLING_V1.md'
HELPERS=['scripts/run_guarded_coupling.py','scripts/run_learned_bottleneck_screen.py',
         'scripts/run_selective_construction_screen.py','scripts/authorized_replay_probe.py']


def prefix_check(model,c,data,objective,deadline):
    small={**data,'modifications':data['modifications'][:2],'repairs':data['repairs'][:2]}
    actual,details,g=score(model,c,small,gradient=True,objective=objective,chunk_size=1,deadline=deadline)
    endpoint_fn=endpoint_bound if objective=='bound' else endpoint_objective
    with sdpa_kernel(SDPBackend.MATH):
        changed,repaired=rollout(model,c,small,True)
        value=torch.stack([endpoint_fn(model,p,small)[0] for p in (changed,repaired)]).amax()
    expected=torch.autograd.grad(value,tuple(model.parameters()))
    difference=float(vector_norm([a-b for a,b in zip(g,expected,strict=True)]))
    norm=float(vector_norm(expected));relative=difference/max(norm,1e-12)
    result={'passed':relative<1e-3 and abs(actual-float(value.detach()))<1e-5,
            'dense_gradient_norm':norm,'adjoint_gradient_norm':float(vector_norm(g)),
            'relative_gradient_l2_error':relative,'absolute_gradient_l2_error':difference,
            'value_absolute_error':abs(actual-float(value.detach())),
            'scope':'Dense versus adjoint on two modification and two repair steps of this parent/configuration'}
    return result


def stock_check(model,c,data,expected_endpoints,deadline,bank,suite):
    """Same batches/mask versus independent stock AdamW; smoothing is explicit."""
    current=copy.deepcopy(model).train();editable={n for n,_ in current.named_parameters() if n.startswith('cells.')}
    for n,p in current.named_parameters():p.requires_grad_(n in editable)
    result=[]
    for phase,expected in zip(('modifications','repairs'),expected_endpoints,strict=True):
        optimizer=torch.optim.AdamW([p for p in current.parameters() if p.requires_grad],
                                    lr=c['inner_lr'] if phase=='modifications' else c['repair_lr'],
                                    betas=(.9,.95),eps=c['inner_epsilon'],weight_decay=0.,foreach=False)
        for batches in data[phase]:
            check_deadline(deadline);optimizer.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.MATH):
                value=edit_loss(current,dict(current.named_parameters()),batches,c)
                value.backward()
            torch.nn.utils.clip_grad_norm_(list(p for p in current.parameters() if p.requires_grad),1.,error_if_nonfinite=True)
            optimizer.step()
        diff=float(vector_norm([p.detach()-expected[n] for n,p in current.named_parameters()]))
        norm=float(vector_norm(list(expected.values())))
        result.append({'phase':phase,'absolute_l2_difference':diff,'relative_l2_difference':diff/max(norm,1e-12),
                       'behavior':quick(current,bank,suite),
                       'maximum_absolute_difference':max(float((p.detach()-expected[n]).abs().max()) for n,p in current.named_parameters())})
    return {'endpoints':result,'attention_backend':'math','scope':'Forward comparison with stock AdamW. Differentiable variant adds 1e-30 inside square roots; no bitwise equivalence presumed.'}


def finite_curve(model,c,data,objective,gradient,reference,deadline,output):
    origin={n:p.detach().clone() for n,p in model.named_parameters()};norm=float(vector_norm(gradient));checks=[]
    if norm==0:return {'checks':[],'inactive':True,'training_gate':False}
    try:
        for radius in (1e-4,1e-5,1e-6):
            values=[];branches=[]
            for sign in (-1,1):
                check_deadline(deadline)
                with torch.no_grad():
                    for (n,p),g in zip(model.named_parameters(),gradient,strict=True):p.copy_(origin[n]+sign*radius*g/norm)
                v,d,_=score(model,c,data,objective=objective,deadline=deadline);values.append(v)
                branches.append({'endpoint':d['selected_endpoint'],'reader':d['selected_branch']['reader'],'stop':d['selected_branch']['stop_rule']})
            numerical=(values[1]-values[0])/(2*radius)
            checks.append({'radius_l2':radius,'analytic':norm,'numerical':numerical,
                           'relative_error':abs(numerical-norm)/norm,'values':values,'branches':branches})
            atomic_json(output,{'checks':checks,'training_gate':False,'reference':reference})
    finally:
        with torch.no_grad():
            for n,p in model.named_parameters():p.copy_(origin[n])
    return {'checks':checks,'reference':reference,'training_gate':False,
            'scope':'FP32 local-scale diagnostic. Every accepted finite proposal must independently improve the full rerun objective and pass intact guards.'}


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+6900;fit_deadline=started+5400
    source=snapshot_sources(a.output/'source');shutil.copyfile(__file__,a.output/'runner.py')
    shutil.copyfile(ROOT/'protocols'/PROTOCOL,a.output/'protocol.md')
    for name in HELPERS:
        out=a.output/'helpers'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,out)
    frozen={n:file_digest(ROOT/n) for n in HELPERS+['scripts/run_long_coupling.py','protocols/'+PROTOCOL]}
    device=configure(a.device,2 if a.device=='cpu' else 4)
    parents=json.loads(a.parents.read_text());assert len(parents)==1
    path=Path(next(iter(parents.values())));saved=load_checkpoint(path);config=dict(saved['configuration'])
    assert config['seed']==a.seed and config['model']['shared_core']
    config.update(evaluation_size=4 if a.smoke else 128,text_blocks=2 if a.smoke else 128,
                  edit_phase_steps=2 if a.smoke else 500,edit_batch_size=2 if a.smoke else 16,
                  continuation='full-recovery-adjoint/v1',objective=a.objective,horizon=a.horizon,smoke=a.smoke)
    model=from_checkpoint(saved).to(device).eval();del saved;origin=state_fingerprint(model)
    bank=TextBank(a.data,blocks=config['text_blocks'])
    c={**defaults(),'inner_scope':'core','repair_scope':'core','inner_epsilon':1e-8,
       'inner_batch_size':2 if a.smoke else 16,'support_size':2 if a.smoke else 32,
       'query_size':2 if a.smoke else 32,'envelope_steps':4 if a.smoke else 500}
    c.update(inner_steps=(4 if a.smoke else 500) if a.horizon=='long' else (2 if a.smoke else 128),
             repair_steps=(4 if a.smoke else 500) if a.horizon=='long' else (2 if a.smoke else 64))
    monitor_config={**c,'inner_steps':c['envelope_steps'],'repair_steps':c['envelope_steps']}
    pools=[]
    for ordinal in range(194000,194001 if a.smoke else 194002):
        pools.append(measurements(bank,monitor_config,ordinal,device,**measurement_exclusions(pools)))
    reserved=measurement_exclusions(pools)
    monitors=[matched_episode(bank,monitor_config,194000+i,device,measurement=m,**reserved) for i,m in enumerate(pools)]
    result={'status':'running','configuration':config,'trajectory':c,'source_sha256':digest(source),
            'helper_sha256':frozen,'parent_sha256':file_digest(path),'environment':environment(device),
            'positive_scc_result':None,'mechanism_assessment_pending':True,'sealed_test_used':False,
            'evidence_class':'CPU fixture' if a.smoke else 'Open full-recovery construction comparison',
            'objective':a.objective,'horizon':a.horizon,'seed':a.seed,'training_attempted':False}
    atomic_json(a.output/'configuration.json',config);atomic_json(a.output/'data-manifest.json',bank.manifest())
    atomic_json(a.output/'reserved-examples.json',exclusion_record(reserved))
    atomic_json(a.output/'monitor-episodes.json',{'configuration':monitor_config,'records':[d['record'] for d in monitors]})
    result['intact_before']=measure(model,bank,config,a.output/'intact-before')
    result['qualified_before']=all(v['qualification']['passed'] for v in result['intact_before']['branches'][0]['layouts'].values())
    fitted=None
    if result['qualified_before'] or a.smoke:
        suite=selection_suite(bank,[reserved],device)
        calibration_reserved={'excluded_task_ids':reserved['excluded_task_ids']|{r['latent_id'] for rows in suite[0].values() for r in rows},
                              'excluded_text_ids':{s:reserved['excluded_text_ids'][s]|set(suite[2]['text_indices'][s]) for s in TEXT_SOURCES}}
        calibration=matched_episode(bank,c,194100,device,**calibration_reserved)
        atomic_json(a.output/'calibration-selection-suite.json',suite[2]);before=quick(model,bank,suite)
        prefix=prefix_check(model,c,calibration,a.objective,deadline)
        atomic_json(a.output/'prefix-check.json',prefix)
        if not prefix['passed']:raise RuntimeError('Dense versus recomputed prefix derivative mismatch')
        value,details,g,changed,repaired=score(model,c,calibration,objective=a.objective,gradient=True,deadline=deadline,return_endpoints=True)
        if not all(torch.isfinite(x).all() for x in g):raise ValueError('Nonfinite full adjoint')
        record={'configuration':c,'episode':calibration['record'],'prefix_check':prefix,'reference':value,
                'gradient_norm':float(vector_norm(g)),'details':details,'intact':before,
                'modified':quick(materialize(model,changed),bank,suite),'repaired':quick(materialize(model,repaired),bank,suite)}
        record['behaviorally_eligible']=eligible(record['repaired'],before)
        record['stock_forward_comparison']=stock_check(model,c,calibration,(changed,repaired),deadline,bank,suite)
        atomic_json(a.output/'calibration.json',record)
        record['finite_curve']=finite_curve(model,c,calibration,a.objective,g,value,deadline,a.output/'calibration-curve.json')
        atomic_json(a.output/'calibration.json',record);result['calibration']=record
        del changed,repaired,g,calibration;gc.collect()
        if record['behaviorally_eligible'] or a.smoke:
            result['monitor_before']={}
            for objective in ('ranking','bound'):
                result['monitor_before'][objective]=[score(model,monitor_config,d,objective=objective,deadline=deadline)[:2] for d in monitors]
            fixed,fixed_record,fixed_used=anchors(bank,device,2083731,2 if a.smoke else 32,reserved)
            atomic_json(a.output/'fixed-anchors.json',fixed_record)
            avoid={'excluded_task_ids':reserved['excluded_task_ids']|fixed_used['excluded_task_ids'],
                   'excluded_text_ids':{s:reserved['excluded_text_ids'][s]|fixed_used['excluded_text_ids'][s] for s in TEXT_SOURCES}}
            rows,ids=guard_rows(2121731,2 if a.smoke else 128,avoid['excluded_task_ids']);avoid['excluded_task_ids'].update(ids)
            guard=GenerationGuard(model,a.output/'generation-guard',rows,avoid['excluded_task_ids'],fresh_size=2 if a.smoke else 32,seed=2141731)
            atomic_json(a.output/'fit-exclusions.json',exclusion_record(avoid))
            used={'excluded_task_ids':set(),'excluded_text_ids':{s:set() for s in TEXT_SOURCES}};episodes=[]
            def add(seen):
                used['excluded_task_ids'].update(seen['excluded_task_ids'])
                for s in TEXT_SOURCES:used['excluded_text_ids'][s].update(seen['excluded_text_ids'][s])
            def episode_factory(i):
                d=matched_episode(bank,c,195000+i,device,**avoid);add(d);episodes.append(d['record']);return d
            def anchor_factory(i,d):
                current={'excluded_task_ids':avoid['excluded_task_ids']|d['excluded_task_ids'],
                         'excluded_text_ids':{s:avoid['excluded_text_ids'][s]|d['excluded_text_ids'][s] for s in TEXT_SOURCES}}
                batches,record,seen=anchors(bank,device,2093731+i,2 if a.smoke else 32,current);add(seen);return batches,record
            with (a.output/'training.jsonl').open('w') as log:
                def callback(current,r):
                    log.write(json.dumps(r)+'\n');log.flush()
                    if r['iteration'] in (2,4):save_model(a.output/f"weights-{r['iteration']:04d}.pt",current,config,resume_implementation_validated=False)
                    print(json.dumps({'phase':'long-coupling','iteration':r['iteration'],'accepted':r['accepted'],'elapsed_seconds':r['elapsed_seconds']}),flush=True)
                fitted,fit=optimize(model,c,episode_factory,fixed,anchor_factory,guard,objective=a.objective,
                                    iterations=2 if a.smoke else 8,deadline=fit_deadline,callback=callback)
            fit.pop('history');fit['log_sha256']=file_digest(a.output/'training.jsonl');result['fitting']=fit
            used['excluded_task_ids'].update(guard.used_ids)
            assert not used['excluded_task_ids'] & reserved['excluded_task_ids']
            assert all(not used['excluded_text_ids'][s]&reserved['excluded_text_ids'][s] for s in TEXT_SOURCES)
            atomic_json(a.output/'training-examples.json',exclusion_record(used));atomic_json(a.output/'training-episodes.json',episodes)
            save_model(a.output/'fitted.pt',fitted,config,fitting=fit,trajectory=c,resume_implementation_validated=False)
            result['training_attempted']=True;result['intact_after']=measure(fitted,bank,config,a.output/'intact-after')
            result['qualified_after']=all(v['qualification']['passed'] for v in result['intact_after']['branches'][0]['layouts'].values())
            result['monitor_after']={}
            for objective in ('ranking','bound'):
                result['monitor_after'][objective]=[score(fitted,monitor_config,d,objective=objective,deadline=deadline)[:2] for d in monitors]
            result['monitor_measurements']={}
            _,_,_,changed,repaired=score(fitted,monitor_config,monitors[0],objective=a.objective,deadline=deadline,return_endpoints=True)
            for phase,params in [('modified',changed),('repaired',repaired)]:
                result['monitor_measurements'][phase]=measure(materialize(fitted,params),bank,config,a.output/('monitor-'+phase))
            del changed,repaired
            atomic_json(a.output/'progress.json',result);result['probes']={}
            for name,scope,benign in [('core','core',False),('all','all',False),('benign','all',True)]:
                check_deadline(deadline);folder=a.output/('probe-'+name)
                def callback(current,phase):
                    if phase=='modification':save_model(folder/'modified.pt',current,config)
                    return measure(current,bank,config,folder/phase)
                _,result['probes'][name]=scoped_edit(fitted,bank,config,scope,folder,deadline,callback,benign)
                atomic_json(a.output/'progress.json',result)
                print(json.dumps({'phase':'independent-probe','name':name,'complete':True}),flush=True)
    assert state_fingerprint(model)==origin and file_digest(path)==result['parent_sha256']
    assert source_manifest()==source and all(file_digest(ROOT/n)==h for n,h in frozen.items())
    total=sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())
    if total>1024**3:raise RuntimeError('Declared1GiB output limit exceeded')
    result.update(status='complete',elapsed_seconds=time.monotonic()-started,parent_unchanged=True,output_bytes_before_result=total)
    result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None
    atomic_json(a.output/'result.json',result)
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],{'status':'complete','training_attempted':result['training_attempted'],'positive_scc_result':None})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,choices=(17,23),required=True)
    p.add_argument('--objective',choices=('ranking','bound'),required=True)
    p.add_argument('--horizon',choices=('short','long'),required=True)
    p.add_argument('--smoke',action='store_true');run(p.parse_args())
