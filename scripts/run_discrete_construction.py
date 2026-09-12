"""Train binary/soft learned codes and evaluate actual parameter modifications."""
import argparse
import collections
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scc.checkpoint import save_checkpoint,load_checkpoint,rng_state
from scc.coupling import nll
from scc.developmental_run import TextBank,Streams,arithmetic_limits,configure,environment,batch_fingerprint
from scc.discrete_models import DiscreteModel,discrete_config,from_checkpoint
from scc.discrete_train import guarded_construction_step
from scc.portfolio_objective import make_contraction_episode
from scc.finite_edit import FiniteEditStepper
from scc.provenance import atomic_json,digest,file_digest,snapshot_sources,source_manifest
from scc.transition_evaluation import measure,ranked_predictions
from scc.transition_path import EditStepper,Events,panels,point
from scc.developmental_tasks import evaluation_rows,batch_rows
PROTOCOL='SCC_DISCRETE_CONSTRUCTION_V1.md'


def configuration(bits,hard,mode,seed,smoke=False):
    return {'architecture':'scc-discrete/v1','model':asdict(discrete_config(bits,hard,smoke)),
            'bits':bits,'hard':hard,'mode':mode,'seed':seed,'data_seed':101,
            'steps':4 if smoke else 20000,'batch_size':2 if smoke else 64,
            'warmup':2 if smoke else 200,'lr':.0006,'decay_start':9000,
            'coupling_frequency':2 if smoke else 25,'inner_steps':2 if smoke else 8,
            'inner_batch_size':2,'evaluation_size':4 if smoke else 128,'evaluation_seed':582019,
            'text_blocks':2 if smoke else 128,'edit_steps':4 if smoke else 500,'edit_batch_size':2 if smoke else 16,
            'finite_steps':2 if smoke else 128,'finite_directions':1 if smoke else 4,'smoke':smoke,
            'meta_radii':[.02,.005,.001],
            'evidence_class':'CPU implementation fixture' if smoke else 'Open discrete-code construction screen; no sealed test'}


def save_model(path,model,config,**extra):
    save_checkpoint(path,{'schema_version':1,'configuration':config,'model':model.state_dict(),**extra})


def fingerprint(model):
    import hashlib
    result=hashlib.sha256()
    for n,v in model.state_dict().items():
        result.update(n.encode());result.update(v.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def train(bank,config,folder,device,deadline):
    folder.mkdir();random.seed(config['seed']);torch.manual_seed(config['seed'])
    model=DiscreteModel(discrete_config(config['bits'],config['hard'],config['smoke'])).to(device).train()
    fixed={n:b.detach().clone() for n,b in model.named_buffers()}
    initial=fingerprint(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
    stream=Streams(bank,config['data_seed'],device,config['batch_size'],.5)
    ordinary_chain=digest('portfolio-ordinary/v1');meta_chain=digest('discrete-guarded-construction/v1')
    completed=episodes=accepted=0;started=time.monotonic()
    with (folder/'steps.jsonl').open('w') as log:
        for step in range(config['steps']):
            if time.monotonic()>=deadline:break
            stream.arithmetic_limits=arithmetic_limits(step,True)
            batch=stream.ordinary();ordinary_chain=digest([ordinary_chain,batch_fingerprint(batch)])
            progress=max(0.,(step-config['decay_start'])/max(1,config['steps']-1-config['decay_start']))
            lr=config['lr']*min(1.,(step+1)/config['warmup'])*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            base=nll(model,dict(model.named_parameters()),batch);base.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();meta=None
            if config['mode']=='coupled' and step%config['coupling_frequency']==0:
                episode=make_contraction_episode(bank,step//config['coupling_frequency'],device,
                    batch_size=config['inner_batch_size'],inner_steps=config['inner_steps'])
                meta=guarded_construction_step(model,episode,batch,radii=config['meta_radii'])
                episodes+=1;accepted+=int(meta['accepted'])
                meta_chain=digest([meta_chain,episode['record']['batch_sha256']])
            completed=step+1
            record={'step':completed,'ordinary_loss':float(base.detach()),'ordinary_gradient_norm':float(norm),
                    'lr':lr,'role':batch.role,'arithmetic_limits':list(stream.arithmetic_limits),
                    'ordinary_batch_sha256':batch_fingerprint(batch),'ordinary_chain':ordinary_chain,'meta_chain':meta_chain,
                    'meta':meta,'ordinary_derivative':'tanh straight-through coarse gradient' if config['hard'] else 'smooth derivative'}
            log.write(json.dumps(record)+'\n');log.flush()
            if completed in (9000,15000):save_model(folder/f'stage-{completed}.pt',model,config,completed_steps=completed)
            if completed%1000==0:print(json.dumps({'phase':'training','step':completed,'meta_opportunities':episodes,
                'accepted_meta_updates':accepted,'ordinary_loss':record['ordinary_loss'],'elapsed_seconds':time.monotonic()-started}),flush=True)
    assert all(torch.equal(fixed[n],b) for n,b in model.named_buffers())
    save_model(folder/'final.pt',model,config,completed_steps=completed,optimizer=optimizer.state_dict(),rng=rng_state(device),
               stream=stream.state_dict(),ordinary_chain=ordinary_chain,meta_chain=meta_chain,resume_implementation_validated=False)
    return model.eval(),{'completed_steps':completed,'complete':completed==config['steps'],'meta_episodes':episodes,
        'accepted_meta_updates':accepted,'ordinary_chain':ordinary_chain,'meta_chain':meta_chain,
        'initial_state_sha256':initial,'fixed_buffers_unchanged':True,'parameter_count':model.parameter_count(),
        'elapsed_seconds':time.monotonic()-started,'steps_sha256':file_digest(folder/'steps.jsonl')}


def probe(parent, bank, config, folder, scope, benign, deadline):
    folder.mkdir()
    model = copy.deepcopy(parent).train()
    stream = EditStepper(model, bank, batch_size=config['edit_batch_size'], scope=scope, benign=benign)
    panel = panels(2 if config['smoke'] else 32, 3 if config['smoke'] else 64)
    atomic_json(folder / 'panels.json', panel)
    events = Events()
    ring = collections.OrderedDict()
    saved = {}
    minimum = 1.
    reference = None
    completed = 0
    def keep(step, reason):
        if step < 0 or step in saved:
            return
        state = ring[step]
        path = folder / 'landmark-weights' / f'step-{step:04d}.pt'
        save_checkpoint(path, {'schema_version': 1, 'configuration': config, 'model': state,
                              'step': step, 'reason': reason, 'optimizer_resume_supported': False})
        saved[step] = {'path': str(path.relative_to(folder)), 'sha256': file_digest(path), 'reason': reason}
        assert len(saved) <= 6
    with (folder / 'curve.jsonl').open('w') as log, (folder / 'edit-steps.jsonl').open('w') as edits:
        for step in range(config['edit_steps'] + 1):
            if time.monotonic() >= deadline:
                break
            if step:
                update = stream.advance()
                edits.write(json.dumps(update) + '\n')
                edits.flush()
            row = point(model, bank, panel, step, reference, 2 if config['smoke'] else 16)
            if reference is None:
                reference = copy.deepcopy(row)
            ring[step] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            while len(ring) > 4:
                ring.popitem(last=False)
            new = events.observe(row)
            for name, at in new.items():
                if name == 'all_domains_low_screen' or (name == 'sustained_reliable_violation' and scope == 'core' and not benign):
                    keep(max(0, at - 1), name)
                    keep(at, name)
            row['new_events'] = new
            log.write(json.dumps(row) + '\n')
            log.flush()
            minimum = min(minimum, row['minimum_benign_strict'])
            completed = step
            if step % 100 == 0 or new:
                print(json.dumps({'phase': 'probe', 'probe': folder.name, 'step': step,
                                  'benign_min': row['minimum_benign_strict'], 'events': new}), flush=True)
    assert all(torch.equal(p, dict(parent.named_parameters())[n]) for n, p in model.named_parameters() if n not in stream.names)
    save_model(folder / 'modified.pt', model, config, modification_steps=completed, edit_scope=scope)
    result = {'scope': scope, 'benign': benign, 'completed_modification_steps': completed,
              'events': events.record(), 'minimum_benign_over_path': minimum, 'saved_landmarks': saved,
              'noneditable_unchanged': True, 'curve_sha256': file_digest(folder / 'curve.jsonl'),
              'modification_chain': stream.chain, 'measurements': {}, 'completed_repair_steps': 0}
    result['measurements']['modified'] = measure(model, bank, config, folder / 'modified')
    result['measurements']['soft-modified'] = measure(model.with_soft(), bank, config, folder / 'soft-modified')
    atomic_json(folder / 'progress.json', result)
    for step, record in sorted(saved.items()):
        if time.monotonic() >= deadline:
            break
        observed = from_checkpoint(load_checkpoint(folder / record['path'])).to(next(model.parameters()).device).eval()
        result['measurements'][f'landmark-{step}'] = measure(observed, bank, config, folder / f'landmark-{step}')
        del observed
    # Match the previous fresh-moment phase while retaining its disjoint query stream.
    stream.stream.begin_query()
    stream.step, stream.phase = 0, 'repair'
    stream.optimizer = torch.optim.AdamW(stream.params, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=0., foreach=False)
    with (folder / 'repair-steps.jsonl').open('w') as log:
        for step in range(config['edit_steps']):
            if time.monotonic() >= deadline:
                break
            record = stream.advance()
            log.write(json.dumps(record) + '\n')
            log.flush()
            result['completed_repair_steps'] = step + 1
    assert all(torch.equal(p, dict(parent.named_parameters())[n]) for n, p in model.named_parameters() if n not in stream.names)
    save_model(folder / 'repaired.pt', model, config, edit_scope=scope, repair_steps=result['completed_repair_steps'])
    result['measurements']['repaired'] = measure(model, bank, config, folder / 'repaired')
    result['measurements']['soft-repaired'] = measure(model.with_soft(), bank, config, folder / 'soft-repaired')
    result['status'] = 'complete' if completed == config['edit_steps'] and result['completed_repair_steps'] == config['edit_steps'] and len(result['measurements']) == len(saved) + 4 else 'incomplete'
    result['final_chain'] = stream.chain
    atomic_json(folder / 'result.json', result)
    return result


def finite_probe(parent,bank,config,folder,scope,deadline):
    folder.mkdir();model=copy.deepcopy(parent).eval()
    editor=FiniteEditStepper(model,bank,scope,config['edit_batch_size'],config['finite_directions'])
    panel=panels(2 if config['smoke'] else 32,3 if config['smoke'] else 64)
    atomic_json(folder/'panels.json',panel)
    baseline=None;events=Events();completed=accepted=evaluations=0
    with (folder/'curve.jsonl').open('w') as log,(folder/'edit-steps.jsonl').open('w') as updates:
        for step in range(config['finite_steps']+1):
            if time.monotonic()>=deadline:break
            if step:
                change=editor.advance();accepted+=int(change['accepted_radius'] is not None)
                evaluations+=change['objective_evaluations']
                updates.write(json.dumps(change)+'\n');updates.flush()
            row=point(model,bank,panel,step,baseline,2 if config['smoke'] else 16)
            if baseline is None:baseline=copy.deepcopy(row)
            row['new_events']=events.observe(row)
            row['boundary_kind']='finite-difference attempt; rejected attempts do not change parameters'
            log.write(json.dumps(row)+'\n');log.flush();completed=step
    save_model(folder/'modified.pt',model,config,modification_attempts=completed,edit_scope=scope)
    result={'scope':scope,'events':events.record(),'completed_modification_steps':completed,
            'accepted_modification_updates':accepted,'objective_evaluations':evaluations,
            'completed_repair_steps':0,'measurements':{},'modification_chain':editor.chain,
            'curve_sha256':file_digest(folder/'curve.jsonl'),'step_unit':'attempt; see accepted updates separately'}
    result['measurements']['modified']=measure(model,bank,config,folder/'modified')
    result['measurements']['soft-modified']=measure(model.with_soft(),bank,config,folder/'soft-modified')
    atomic_json(folder/'progress.json',result)
    repair=EditStepper(model,bank,batch_size=config['edit_batch_size'],scope=scope,skip_modifications=500)
    with (folder/'repair-steps.jsonl').open('w') as log:
        for step in range(config['edit_steps']):
            if time.monotonic()>=deadline:break
            log.write(json.dumps(repair.advance())+'\n');log.flush()
            result['completed_repair_steps']=step+1
    save_model(folder/'repaired.pt',model,config,repair_steps=result['completed_repair_steps'],edit_scope=scope)
    result['measurements']['repaired']=measure(model,bank,config,folder/'repaired')
    result['measurements']['soft-repaired']=measure(model.with_soft(),bank,config,folder/'soft-repaired')
    assert all(torch.equal(p,dict(parent.named_parameters())[n]) for n,p in model.named_parameters() if n not in editor.names)
    assert all(torch.equal(b,dict(parent.named_buffers())[n]) for n,b in model.named_buffers())
    result.update(noneditable_unchanged=True,status='complete' if completed==config['finite_steps'] and result['completed_repair_steps']==config['edit_steps'] else 'incomplete')
    atomic_json(folder/'result.json',result)
    return result


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source=snapshot_sources(a.output/'source')
    shutil.copyfile(__file__,a.output/'runner.py')
    shutil.copyfile(ROOT/'protocols'/PROTOCOL,a.output/'protocol.md')
    assert json.loads(a.parents.read_text())=={}
    device=configure(a.device,2 if a.device=='cpu' else 4)
    config=configuration(a.bits,a.encoding=='hard',a.mode,a.seed,a.smoke)
    atomic_json(a.output/'configuration.json',config)
    bank=TextBank(a.data,config['text_blocks']);atomic_json(a.output/'data-manifest.json',bank.manifest())
    model,training=train(bank,config,a.output/'training',device,started+5100)
    origin=fingerprint(model)
    intact=measure(model,bank,config,a.output/'intact')
    qualified=all(x['qualification']['passed'] for x in intact['branches'][0]['layouts'].values())
    result={'schema':'discrete-construction/v1','configuration':config,'environment':environment(device),
            'training':training,'intact':intact,'qualified_intact':qualified,'probes':{},
            'source_sha256':digest(source),'runner_sha256':file_digest(a.output/'runner.py'),
            'protocol_sha256':file_digest(a.output/'protocol.md'),'positive_scc_result':None,
            'autonomous_execution_tested':False,'sealed_test_used':False,'evidence_class':config['evidence_class']}
    atomic_json(a.output/'progress.json',result)
    if qualified or a.smoke:
        for label,scope,benign in [('core','core',False),('all','all',False),('benign','all',True)]:
            if time.monotonic()>=started+6900:break
            result['probes'][label]=probe(model,bank,config,a.output/('probe-'+label),scope,benign,started+6900)
            atomic_json(a.output/'progress.json',result)
        for scope in ('core','all'):
            if time.monotonic()>=started+6900:break
            result['probes']['finite-'+scope]=finite_probe(model,bank,config,a.output/('probe-finite-'+scope),scope,started+6900)
            atomic_json(a.output/'progress.json',result)
        if time.monotonic()<started+6900:result['graph_bypass']=measure(model.with_bypass(),bank,config,a.output/'graph-bypass')
        if time.monotonic()<started+6900:
            signs=torch.ones(config['model']['bottleneck_width'],device=device);signs[::2]=-1
            recoded=model.recoded(signs)
            rows=evaluation_rows(2 if a.smoke else 8,seed=853491)
            for row in rows:
                batch=batch_rows([row],device=device)
                with torch.no_grad():assert torch.equal(model(batch.tokens),recoded(batch.tokens))
            result['benign_recoding']={'exact_logits':True,'examples_checked':len(rows),'signs':signs.tolist(),
                'scope':'Code signs and expansion columns changed together; protection-preserving transformation',
                'measurement':measure(recoded,bank,config,a.output/'benign-recoding')}
    assert fingerprint(model)==origin and source_manifest()==source
    size=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file());assert size<1024**3
    complete=training['complete'] and (not qualified and not a.smoke or len(result['probes'])==5 and all(x['status']=='complete' for x in result['probes'].values()) and 'graph_bypass' in result and 'benign_recoding' in result)
    result.update(status='complete' if complete else 'incomplete',elapsed_seconds=time.monotonic()-started,
                  disposition='qualified_challenged' if qualified else 'intact_gate_failed',output_bytes_before_result=size)
    atomic_json(a.output/'result.json',result)
    if os.environ.get('GMN_RESULT_PATH'):
        atomic_json(os.environ['GMN_RESULT_PATH'],{'ok':complete,'scientific_status':result['status'],
            'qualified_intact':qualified,'training_steps':training['completed_steps'],
            'accepted_meta_updates':training['accepted_meta_updates'],'schema':result['schema'],'positive_scc_result':None})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--parents',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--bits',type=int,choices=[32,128],required=True)
    p.add_argument('--encoding',choices=['hard','soft'],required=True)
    p.add_argument('--mode',choices=['ordinary','coupled'],required=True);p.add_argument('--seed',type=int,default=23)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():atomic_json(a.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
