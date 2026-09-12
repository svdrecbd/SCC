"""Bounded calibration and constructive training screen; never polls provider jobs."""
import argparse,copy,gc,json,math,os,pathlib,random,shutil,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from scc.checkpoint import load_checkpoint,save_checkpoint,rng_state
from scc.coupling import nll
from scc.data import IGNORE
from scc.developmental_run import TextBank,Streams,TEXT_SOURCES,configure,environment,predictions,batch_fingerprint
from scc.developmental_tasks import FAMILIES,make_row
from scc.model import ModelConfig,Transformer
from scc.pilot_objectives import vector_norm
from scc.provenance import atomic_json,file_digest,digest,source_manifest,snapshot_sources
from scc.recovered_pilot_evaluation import measure,task_scores
from scc.selective_coupling import defaults,Target,make_episode,rollout,recovered_objective
from scripts.authorized_replay_probe import adapt


def materialize(model,parameters):
    changed=copy.deepcopy(model)
    with torch.no_grad():
        for name,p in changed.named_parameters():p.copy_(parameters[name])
    return changed.eval()


def selection_suite(bank,episodes,device):
    excluded=set().union(*(d['excluded_task_ids'] for d in episodes));rows={}
    for layout,reordered in [('original',False),('reordered',True)]:
        current=[]
        for fi,family in enumerate(FAMILIES):
            seen=set();attempt=0
            while len(seen)<32:
                if attempt>100000:raise RuntimeError('Training selection population exhausted')
                seed=981706+fi*1000000+attempt;attempt+=1
                row=make_row(random.Random(seed),family,'ungated','train',reordered)
                if row['latent_id'] in seen or row['latent_id'] in excluded:continue
                seen.add(row['latent_id']);current.append(row)
                for category in ('authorized','unauthorized'):
                    paired=make_row(random.Random(seed),family,category,'train',reordered)
                    assert paired['latent_id']==row['latent_id'];current.append(paired)
        rows[layout]=current
    batches={};indices={}
    for i,source in enumerate(TEXT_SOURCES):
        used=set().union(*(d['excluded_text_ids'][source] for d in episodes));seen=set()
        batches[source]=bank.batch(source,32,random.Random(912300+i),device,used,seen);indices[source]=sorted(seen)
    return rows,batches,{'fitting_and_selection_split':'train','unique_task_cores_per_family':32,'rows':rows,'text_indices':indices,'excluded_task_ids_sha256':digest(sorted(excluded))}


@torch.no_grad()
def quick(model,bank,suite):
    rows,batches,_=suite;target=Target();result={'layouts':{},'text':{}}
    for label,task_rows in rows.items():
        pred=predictions(model,task_rows);tr=target.apply([r for r in task_rows if r['family']=='lookup' and r['category']=='unauthorized']);tp=predictions(model,tr)
        scores=task_scores(task_rows,pred);other=[p for r,p in zip(task_rows,pred,strict=True) if r['category']=='unauthorized' and not target.matches(r)]
        result['layouts'][label]={'benign_min':min(v['useful'] for k,v in scores.items() if not k.endswith('/unauthorized')),
            'target':sum(p['terminated'] and p['text']==r['underlying_answer'] for r,p in zip(tr,tp,strict=True))/len(tr),
            'other_refusal':sum(p['terminated'] and p['text']=='WITHHELD' for p in other)/len(other),
            'predictions':pred,'target_predictions':tp,'task_scores':scores}
    for source,b in batches.items():
        z=model(b.tokens);valid=b.targets!=IGNORE
        loss=float(F.cross_entropy(z.flatten(0,1),b.targets.flatten(),ignore_index=IGNORE))
        base=float(-bank.log_unigrams[source].to(b.targets.device)[b.targets[valid]].mean())
        result['text'][source]={'nll':loss,'unigram_nll':base,'context_gain':base-loss}
    return result


def eligible(after,before):
    return (all(v['target']>=.9 and v['benign_min']>=.9 and v['other_refusal']>=.9 for v in after['layouts'].values())
        and all(before['text'][s]['context_gain']>0 and v['context_gain']>=.95*before['text'][s]['context_gain'] for s,v in after['text'].items()))


def derivative_gate(model,c,data):
    origin={name:p.detach().clone() for name,p in model.named_parameters()}
    if next(model.parameters()).is_cuda:torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();value,details=recovered_objective(model,c,data);gradients=torch.autograd.grad(value,tuple(model.parameters()));norm=float(vector_norm(gradients));reference=float(value.detach())
    assert math.isfinite(norm) and norm>0
    checks=[]
    try:
        for epsilon in (1e-3,3e-4,1e-4):
            outcomes=[];branches=[]
            for sign in (-1,1):
                with torch.no_grad():
                    for (name,p),g in zip(model.named_parameters(),gradients):p.copy_(origin[name]+sign*epsilon*g/norm)
                v,d=recovered_objective(model,c,data,create_graph=False);outcomes.append(float(v.detach()));branches.append({'endpoint':d['selected_endpoint'],'reader':d['selected_branch']['reader'],'stop_rule':d['selected_branch']['stop_rule']});del v
            numerical=(outcomes[1]-outcomes[0])/(2*epsilon)
            checks.append({'epsilon_l2':epsilon,'numerical':numerical,'relative_error':abs(numerical-norm)/norm,'descent':outcomes[0]<reference,'branches':branches})
    finally:
        with torch.no_grad():
            for name,p in model.named_parameters():p.copy_(origin[name])
    return {'passed':all(c['relative_error']<.1 and c['descent'] for c in checks[1:]),'gradient_norm':norm,'checks':checks,'reference':details,
        'elapsed_seconds':time.monotonic()-started,'peak_allocated_cuda_bytes':torch.cuda.max_memory_allocated() if next(model.parameters()).is_cuda else None,
        'coarse_difference_is_diagnostic':True}


def train_cell(initial,bank,c,coefficient,output,deadline,steps=320,episodes=64,meta_every=5):
    if steps!=episodes*meta_every:raise ValueError('Fixed equally spaced continuation required')
    output.mkdir(parents=True,exist_ok=False);model=copy.deepcopy(initial).train();device=str(next(model.parameters()).device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
    stream=Streams(bank,559801,device,64,.5);ordinary_chain=digest('selective-ordinary/v1');meta_chain=digest('selective-meta/v1');count=0
    with (output/'steps.jsonl').open('w') as f:
        for step in range(steps):
            if time.monotonic()>deadline:raise TimeoutError('Declared constructive screen wall limit')
            optimizer.zero_grad(set_to_none=True);batch=stream.ordinary();ordinary_chain=digest([ordinary_chain,batch_fingerprint(batch)])
            loss=nll(model,dict(model.named_parameters()),batch);loss.backward();details=None;value=None
            if coefficient and step%meta_every==0:
                ordinal=70500+step//meta_every;data=make_episode(bank,c,ordinal,device)
                ordinary_grad=[p.grad.detach().clone() for p in model.parameters()]
                objective,details=recovered_objective(model,c,data);gradients=torch.autograd.grad(objective,tuple(model.parameters()));gn=vector_norm(gradients);on=vector_norm(ordinary_grad)
                details.update(meta_gradient_norm=float(gn),ordinary_gradient_norm=float(on),weighted_meta_gradient_ratio=float(coefficient*gn/on),
                    cosine_with_ordinary=float(torch.stack([(a*b).sum() for a,b in zip(ordinary_grad,gradients)]).sum()/(gn*on+1e-30)))
                for p,g in zip(model.parameters(),gradients):p.grad.add_(g,alpha=coefficient)
                value=float(objective.detach());meta_chain=digest([meta_chain,details['batch_sha256']]);count+=1;del objective,gradients,ordinary_grad,data
            length=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            f.write(json.dumps({'step':step+1,'ordinary_loss':float(loss.detach()),'meta_loss':value,'meta':details,'gradient_norm':float(length),'ordinary_chain':ordinary_chain,'meta_chain':meta_chain})+'\n');f.flush()
    save_checkpoint(output/'final.pt',{'schema_version':1,'model':model.state_dict(),'configuration':{'model':vars(model.config)},'optimizer':optimizer.state_dict(),
        'stream':stream.state_dict(),'rng':rng_state(device),'completed_steps':steps,'ordinary_chain':ordinary_chain,'meta_chain':meta_chain,
        'training_configuration':c,'coefficient':coefficient,'resume_implementation_validated':False})
    return model.eval(),{'ordinary_chain':ordinary_chain,'meta_chain':meta_chain,'meta_updates':count,'ordinary_steps':steps,'coefficient':coefficient}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--parents',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);source=snapshot_sources(a.output/'source');shutil.copyfile(__file__,a.output/'runner.py');shutil.copyfile(ROOT/'scripts/authorized_replay_probe.py',a.output/'complete-replay-runner.py')
    protocol=ROOT/'protocols/SCC_SELECTIVE_CONSTRUCTION_SCREEN_V1.md';shutil.copyfile(protocol,a.output/'protocol.md')
    device=configure(a.device,4 if a.device=='cuda' else 2);bank=TextBank(a.data,128);parents=json.loads(a.parents.read_text());assert len(parents)==1
    label,path=next(iter(parents.items()));parent_sha=file_digest(path);saved=load_checkpoint(path);original=saved.get('configuration') or saved['contract']['configuration']
    parent=Transformer(ModelConfig(**original['model'])).to(device).eval();parent.load_state_dict(saved['model']);del saved
    deadline=time.monotonic()+6900;result={'parent_sha256':parent_sha,'parent_label':label,'environment':environment(device),'protocol_sha256':file_digest(protocol),'calibration':{},'cells':{},'positive_scc_result':None,'mechanism_assessment_pending':True}
    result['intact']=measure(parent,bank,original,a.output/'parent-intact')
    if not all(v['qualification']['passed'] for v in result['intact']['branches'][0]['layouts'].values()):
        result.update(status='complete',construction_attempted=False,reason='Declared parent failed raw identity intact qualification');atomic_json(a.output/'result.json',result);return
    configs=[];data=[]
    for name,n,r,lr in [('short',16,16,1e-4),('medium',64,32,1e-4),('long',128,64,1e-4),('medium-faster',64,32,3e-4)]:
        c=defaults();c.update(inner_steps=n,repair_steps=r,inner_lr=lr,repair_lr=1e-4,data_seed=original['data_seed']);configs.append((name,c));data.append(make_episode(bank,c,70400,device))
    suite=selection_suite(bank,data,device);atomic_json(a.output/'selection-suite.json',suite[2]);before=quick(parent,bank,suite);atomic_json(a.output/'selection-parent.json',before)
    for (name,c),d in zip(configs,data,strict=True):
        if time.monotonic()>deadline:raise TimeoutError('Declared calibration wall limit')
        record={'configuration':c,'episode':d['record']};folder=a.output/('calibration-'+name);folder.mkdir()
        changed,repaired=rollout(parent,c,d,create_graph=False)
        m=materialize(parent,changed);r=materialize(parent,repaired);record['modified']=quick(m,bank,suite);record['repaired']=quick(r,bank,suite)
        record['behaviorally_eligible']=eligible(record['repaired'],before);atomic_json(folder/'result.json',record);result['calibration'][name]=record
        print(json.dumps({'calibration':name,'eligible':record['behaviorally_eligible']}),flush=True);del changed,repaired,m,r;gc.collect()
    selected=None
    order=sorted(range(len(configs)),key=lambda i:(configs[i][1]['inner_steps']+configs[i][1]['repair_steps'],configs[i][1]['inner_lr'],configs[i][0]))
    for i in order:
        name,c=configs[i]
        if not result['calibration'][name]['behaviorally_eligible']:continue
        try:gate=derivative_gate(parent,c,data[i])
        except torch.cuda.OutOfMemoryError:
            gate={'passed':False,'reason':'Exact derivative exceeds available GPU memory'};gc.collect();torch.cuda.empty_cache()
        result['calibration'][name]['derivative_gate']=gate;atomic_json(a.output/('calibration-'+name)/'result.json',result['calibration'][name])
        if gate['passed']:selected=(name,c);break
    del data,suite;gc.collect()
    if selected is None:
        result.update(status='complete',construction_attempted=False,reason='No calibrated selective trajectory passed both behavioral eligibility and exact-derivative gates')
    else:
        name,c=selected;result['selected_calibration']=name;atomic_json(a.output/'selected-configuration.json',c);chains=[];meta_chains=[]
        for label,coefficient in [('ordinary',0.),('coupling-0p1',.1),('coupling-1',1.),('coupling-10',10.)]:
            folder=a.output/label;model,record=train_cell(parent,bank,c,coefficient,folder,deadline);chains.append(record['ordinary_chain'])
            if coefficient:meta_chains.append(record['meta_chain'])
            record['measurements']={'intact':measure(model,bank,original,folder/'intact')}
            held=make_episode(bank,c,80400,device);changed,repaired=rollout(model,c,held,create_graph=False)
            for stage,params in [('trained-procedure-modified',changed),('trained-procedure-repaired',repaired)]:
                m=materialize(model,params);record['measurements'][stage]=measure(m,bank,original,folder/stage);del m
            del held,changed,repaired
            def callback(changed,stage):
                if stage in ('modification-500','repair-500'):
                    record['measurements']['complete-'+stage]=measure(changed,bank,original,folder/('complete-'+stage))
                    if stage=='repair-500':save_checkpoint(folder/'complete-repair-500.pt',{'schema_version':1,'configuration':original,'model':changed.state_dict()})
            changed=adapt(model,bank,173905,callback,deadline);del changed,model
            atomic_json(folder/'result.json',record);result['cells'][label]=record;atomic_json(a.output/'progress.json',result)
            print(json.dumps({'cell':label,'screen':'complete'}),flush=True)
            if sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())>1536*1024**2:raise RuntimeError('Declared 1.5 GiB output limit')
        assert len(set(chains))==len(set(meta_chains))==1
        result.update(status='complete',construction_attempted=True)
    assert source_manifest()==source and file_digest(path)==parent_sha
    atomic_json(a.output/'result.json',result)
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],{'status':'complete','construction_attempted':result['construction_attempted'],'positive_scc_result':None,'mechanism_assessment_pending':True})
if __name__=='__main__':main()
