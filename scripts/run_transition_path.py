"""Observe every stock-edit boundary; no autonomous model or automatic polling."""
import argparse,collections,copy,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scc.checkpoint import load_checkpoint,save_checkpoint
from scc.developmental_run import TextBank,configure,environment
from scc.fixed_episode_optimization import check_deadline
from scc.learned_bottleneck import from_checkpoint
from scc.provenance import atomic_json,file_digest,snapshot_sources,source_manifest,digest
from scc.transition_evaluation import measure
from scc.transition_path import EditStepper,Events,panels,point
from scripts.run_learned_bottleneck_screen import state_fingerprint
PROTOCOL='SCC_TRANSITION_TIMING_V1.md'
HELPERS=['scripts/run_learned_bottleneck_screen.py','scripts/run_selective_construction_screen.py','scripts/authorized_replay_probe.py']


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+6900;path_deadline=started+5400
    source=snapshot_sources(a.output/'source');shutil.copyfile(__file__,a.output/'runner.py');shutil.copyfile(ROOT/'protocols'/PROTOCOL,a.output/'protocol.md')
    frozen={n:file_digest(ROOT/n) for n in HELPERS+['scripts/run_transition_path.py','protocols/'+PROTOCOL]}
    for n in HELPERS:
        p=a.output/'helpers'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    device=configure(a.device,2 if a.device=='cpu' else 4);parents=json.loads(a.parents.read_text());assert 'candidate' in parents and set(parents)<= {'candidate','reference'}
    saved=load_checkpoint(parents['candidate']);config=dict(saved['configuration']);assert config['model']['shared_core'];parent=from_checkpoint(saved).to(device).eval();del saved
    origin=state_fingerprint(parent);config.update(evaluation_size=4 if a.smoke else 128,text_blocks=2 if a.smoke else 128,smoke=a.smoke)
    model=copy.deepcopy(parent);bank=TextBank(a.data,blocks=config['text_blocks']);panel=panels(2 if a.smoke else 32,3 if a.smoke else 64);atomic_json(a.output/'panels.json',panel);atomic_json(a.output/'configuration.json',config);atomic_json(a.output/'data-manifest.json',bank.manifest())
    intact=measure(parent,bank,config,a.output/'intact');qualified=all(x['qualification']['passed'] for x in intact['branches'][0]['layouts'].values());steps=4 if a.smoke else 500
    result={'schema':'transition-path/v1','status':'running','label':a.label,'benign_control':a.benign,'qualified_before':qualified,'intact':intact,'parents_sha256':{n:file_digest(p) for n,p in parents.items()},'environment':environment(device),'source_sha256':digest(source),'helpers_sha256':frozen,'sealed_test_used':False,'requested_steps':steps,'path_complete':False,'autonomous_execution_tested':False,'positive_scc_result':None,'smoke':a.smoke}
    if qualified or a.smoke:
        stepper=EditStepper(model,bank,batch_size=2 if a.smoke else 16,benign=a.benign);events=Events();ring=collections.OrderedDict();checkpoints={};curve=[];baseline=None;stop='declared_steps'
        def remember(step):
            ring[step]={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
            while len(ring)>4:ring.popitem(last=False)
        def save(step,reason):
            if step<0 or step in checkpoints:return
            if step not in ring:raise ValueError('Missing event state')
            p=a.output/'checkpoints'/f'step-{step:04d}.pt';save_checkpoint(p,{'schema_version':1,'configuration':config,'model':ring[step],'observed_step':step,'reason':reason,'optimizer_resume_supported':False})
            checkpoints[step]={'path':str(p.relative_to(a.output)),'sha256':file_digest(p),'reason':reason}
            if len(checkpoints)>20:raise ValueError('Checkpoint budget exceeded')
        with (a.output/'curve.jsonl').open('w') as log,(a.output/'edit-steps.jsonl').open('w') as edits:
            for step in range(steps+1):
                if step:
                    try:check_deadline(path_deadline)
                    except TimeoutError:stop='path_wall_limit';break
                    update=stepper.advance();edits.write(json.dumps(update)+'\n');edits.flush()
                row=point(model,bank,panel,step,baseline,2 if a.smoke else 16)
                if baseline is None:baseline=copy.deepcopy(row)
                remember(step);new=events.observe(row)
                if step==0:save(step,'initial')
                for name,at in new.items():
                    for index in (at-1,at,step):save(index,name)
                row['new_events']=new;row['elapsed_seconds']=time.monotonic()-started;log.write(json.dumps(row)+'\n');log.flush()
                curve.append({k:v for k,v in row.items() if k!='predictions'})
                if step%10==0 or new:print(json.dumps({'phase':'path','step':step,'minimum_target_payload_rate':min(v['payload_accuracy'] for v in row['targets'].values()),'minimum_benign_accuracy':row['minimum_benign_strict'],'new_events':new,'elapsed_seconds':row['elapsed_seconds']}),flush=True)
            last=events.last_step;save(last,'final')
        frozen_changes=[n for n,p in model.named_parameters() if n not in stepper.names and not torch.equal(p,dict(parent.named_parameters())[n])];assert not frozen_changes
        result.update(path_complete=last==steps,completed_steps=last,stop_reason=stop,events=events.record(),checkpoints=checkpoints,curve_sha256=file_digest(a.output/'curve.jsonl'),edit_log_sha256=file_digest(a.output/'edit-steps.jsonl'),final_chain=stepper.chain,noncore_unchanged=True)
        if 'reference' in parents and last==steps:
            reference=load_checkpoint(parents['reference'])['model'];diff=[(p.detach().cpu()-reference[n]).abs().max().item() for n,p in model.state_dict().items()]
            result['previous_endpoint_comparison']={'exact_match':max(diff)==0,'maximum_absolute_difference':max(diff),'scope':'Actual saved tensors; a mismatch is reported, not silently called an exact replay'}
        chosen={0,last};first=events.first.get('sustained_reliable_violation',events.first.get('joint_reliable_violation'))
        if first is not None:chosen.update((max(0,first-1),first))
        refusal=events.first.get('sustained_refusal_loss',events.first.get('joint_refusal_loss'))
        if refusal is not None:chosen.update((max(0,refusal-1),refusal))
        low=events.first.get('all_domains_low_screen')
        if low is not None:chosen.update((max(0,low-1),low))
        result['landmark_steps_requested']=sorted(chosen);result['landmarks']={};atomic_json(a.output/'progress.json',result)
        for step in sorted(chosen):
            if time.monotonic()>=deadline:break
            state=load_checkpoint(a.output/checkpoints[step]['path']);m=from_checkpoint(state).to(device).eval()
            result['landmarks'][str(step)]=measure(m,bank,config,a.output/f'landmark-{step:04d}');del m
            atomic_json(a.output/'progress.json',result)
        result['landmarks_complete']=len(result['landmarks'])==len(chosen)
        result['minimum_benign_strict_over_path']=min(x['minimum_benign_strict'] for x in curve)
    assert source_manifest()==source and all(file_digest(ROOT/n)==h for n,h in frozen.items()) and state_fingerprint(parent)==origin
    size=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file());assert size<1024**3
    result.update(status='complete' if result.get('path_complete') and result.get('landmarks_complete') else 'incomplete',elapsed_seconds=time.monotonic()-started,output_bytes_before_result=size,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)
    atomic_json(a.output/'result.json',result)
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],{'status':result['status'],'completed_steps':result.get('completed_steps',0),'positive_scc_result':None})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parents',type=Path,required=True);p.add_argument('--data',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--label',required=True);p.add_argument('--benign',action='store_true');p.add_argument('--smoke',action='store_true');run(p.parse_args())
