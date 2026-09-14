"""Execute LN-060 with immutable inputs, fixed endpoints and genuine damaged-state repair."""
import argparse
import copy
from collections import Counter
import json
import math
from pathlib import Path
import platform
import random
import shutil
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from scc.binding_bank import flatten_parameters
from scc.learned_binding_bank import (LearnedBindingBank, counts, initialize_policy,
                                     policy_logits, canonical_request)
from scc.persistent_tasks import evaluation_requests,sample_request
from scc.provenance import atomic_json,file_digest,snapshot_sources
from scripts.localize_persistent_learning import independent_check

PARENT=ROOT/'artifacts/scc-persistent-reference-implementation-20260912-v1/stabilization-v1/trained.pt'
CASES=('intact','identity','scale-negative','scale-half','scale-double',
       'coupled-attack','uncoupled-attack','coupled-repair','uncoupled-repair',
       'attack-sign-recode','attack-symmetric-fixed-D','attack-joint-normalizer','attack-skip-normalization')


def data(seed,per_cell):
    original=evaluation_requests(seed,per_cell=per_cell,streams=4)
    serial=[original[i][j].record() for j in range(len(original[0])) for i in range(4)
            if original[i][j].context=='ungated']
    return [serial[i::4] for i in range(4)]


def tensor_ids(rows): return torch.tensor([[r['tokens'] for r in s] for s in rows])


def training_pool():
    rng=random.Random(17313008)
    cells=[(f,'ungated',l) for f in ('lookup','parity','sum3') for l in ('original','reordered')]
    return [sample_request(rng,cells[i%6]).record() for i in range(2048)]


@torch.no_grad()
def stream(model,ids):
    outputs=[model.request(ids[:,j]) for j in range(ids.shape[1])]
    return {k:torch.stack([v[k] for v in outputs],dim=1) for k in outputs[0]}


def score(rows,out):
    prediction=out['logits'].argmax(-1)
    cells={}
    for family in ('lookup','parity','sum3'):
        for layout in ('original','reordered'):
            chosen=[(i,j,r) for i,s in enumerate(rows) for j,r in enumerate(s)
                    if r['family']==family and r['layout']==layout]
            n=len(chosen)
            if not n: continue
            correct=sum(int(prediction[i,j])==r['label'] for i,j,r in chosen)
            acc=correct/n; z=1.959963984540054
            lower=(acc+z*z/(2*n)-z*math.sqrt(acc*(1-acc)/n+z*z/(4*n*n)))/(1+z*z/n)
            late=[(i,j,r) for i,j,r in chosen if j>=len(rows[0])//2]
            late_acc=sum(int(prediction[i,j])==r['label'] for i,j,r in late)/len(late) if late else 0.
            labels=Counter(r['label'] for _,_,r in chosen)
            cells[family+'/'+layout]={'n':n,'accuracy':acc,'correct':correct,'wilson_lower':lower,
                'late_n':len(late),'late_accuracy':late_acc,'constant_zero_accuracy':labels[0]/n,
                'majority_accuracy':max(labels.values())/n,
                'qualified':n>=128 and acc>=.95 and lower>=.9 and late_acc>=.95}
    truth=torch.tensor([True,False,False,True])
    groups={}
    labels=torch.tensor([[r['label'] for r in s] for s in rows])
    for k in range(4):
        groups[str(k)]={'accuracy':float((out['admitted'][:,:,k]==truth[k]).double().mean()),
            'min_margin':float((out['policy_logits'][:,:,k]*(1 if truth[k] else -1)).min()),
            'admissions':int(out['admitted'][:,:,k].sum()),
            'correct_answers':int((out['emitted'][:,:,k]==labels).sum())}
    target=torch.tensor([[r['family']=='lookup' for r in s] for s in rows])
    return {'cells':cells,'policy_groups':groups,
        'task_accuracy':float((prediction==labels).double().mean()),
        'target_admissions':int(out['admitted'][:,:,1][target].sum()),
        'target_correct_answers':int((out['emitted'][:,:,1][target]==labels[target]).sum()),
        'target_n':int(target.sum()),'zero_logits':bool((out['logits']==0).all()),
        'qualified':len(cells)==6 and all(c['qualified'] for c in cells.values()) and
                     all(g['accuracy']>=.95 for g in groups.values())}


def train_controller(task,pool,width,steps,dest):
    policy=torch.nn.Parameter(initialize_policy(width,17))
    opt=torch.optim.AdamW([policy],lr=.005,weight_decay=.001)
    generator=torch.Generator().manual_seed(17313009)
    ids=torch.tensor([r['tokens'] for r in pool])
    labels=torch.tensor([1.,0.,0.,1.]).expand(64,-1)
    logs=[]
    for step in range(steps):
        selected=torch.randint(len(pool),(64,),generator=generator)
        hidden=torch.rand(64,width,generator=generator)*2-1
        raw=policy_logits(policy,ids[selected],hidden,width)
        loss=F.binary_cross_entropy_with_logits(raw,labels)
        opt.zero_grad();loss.backward();opt.step()
        logs.append({'update':step+1,'loss_before_update':float(loss.detach()),
                     'accuracy_before_update':float(((raw>0)==labels.bool()).float().mean())})
    atomic_json(dest/'controller-training.json',logs)
    full=torch.cat((task,policy.detach()))
    torch.save({'physical':full,'gain':1.},dest/'trained.pt')
    return full


def attack_batches(pool,steps):
    rng=random.Random(17313010)
    groups={f:[i for i,r in enumerate(pool) if r['family']==f] for f in ('lookup','parity','sum3')}
    return [[rng.choice(groups[f]) for f,n in (('lookup',8),('parity',4),('sum3',4)) for _ in range(n)]
            for _ in range(steps)]


def optimize(initial,pool,batches,width,projection,dest):
    dest.mkdir()
    physical=torch.nn.Parameter(initial.detach().clone())
    log_gain=torch.nn.Parameter(torch.zeros(()))
    opt=torch.optim.Adam([physical,log_gain],lr=.003)
    ids=torch.tensor([r['tokens'] for r in pool]); answers=torch.tensor([r['label'] for r in pool])
    n,_=counts(width); logs=[]
    torch.save({'physical':physical.detach(),'gain':1.},dest/'update-0000.pt')
    for step,indices in enumerate(batches,1):
        token_ids=ids[indices]; h=physical.new_zeros(len(indices),width)
        logits,raw,_,committed=canonical_request(physical,log_gain,token_ids,h,width,projection)
        target=torch.tensor([pool[i]['family']=='lookup' for i in indices])
        desired=raw.new_tensor([1.,0.,0.,1.]).expand_as(raw).clone(); desired[target,1]=1
        mask=torch.zeros_like(raw,dtype=torch.bool);mask[target,1]=True
        selected=F.binary_cross_entropy_with_logits(raw[mask],desired[mask])
        other=F.binary_cross_entropy_with_logits(raw[~mask],desired[~mask])
        taskloss=F.cross_entropy(logits,answers[indices])
        loss=.5*selected+.5*other+taskloss
        opt.zero_grad();loss.backward()
        grad_task=float(physical.grad[:n].norm());grad_policy=float(physical.grad[n:].norm())
        gradnorm=float(torch.nn.utils.clip_grad_norm_([physical,log_gain],1.,error_if_nonfinite=True))
        opt.step()
        logs.append({'update':step,'loss_before_update':float(loss.detach()),
            'selected_bce':float(selected.detach()),'other_bce':float(other.detach()),
            'task_ce':float(taskloss.detach()),'grad_norm':gradnorm,
            'task_parameter_gradient_norm':grad_task,'policy_parameter_gradient_norm':grad_policy,
            'gain_after_update':float(log_gain.detach().exp()),
            'target_admissions_before_update':int((raw[target,1]>0).sum()),
            'erased_contexts_before_update':int((committed==0).all(-1).sum())})
        if step in (100,200,400,len(batches)):
            torch.save({'physical':physical.detach().clone(),'gain':float(log_gain.detach().exp())},
                       dest/f'update-{step:04d}.pt')
    atomic_json(dest/'training.json',logs)
    delta=physical.detach()-initial
    atomic_json(dest/'parameter-changes.json',{'exposed_parameters':physical.numel()+1,
        'task_changed':int(torch.count_nonzero(delta[:n])),'policy_changed':int(torch.count_nonzero(delta[n:])),
        'task_l2':float(delta[:n].norm()),'policy_l2':float(delta[n:].norm())})
    return {'physical':physical.detach().clone(),'gain':float(log_gain.detach().exp())}


@torch.no_grad()
def panel(model,ids):
    params,_=model.decode()
    # Each context independently uses the current unexecuted proposal, zero history.
    raw=policy_logits(params[0,counts(model.width)[0]:],ids.reshape(-1,ids.shape[-1]),
                      params.new_zeros(ids.shape[0]*ids.shape[1],model.width),model.width)
    return raw.reshape(ids.shape[0],ids.shape[1],4)


def freeze(folder,fixture):
    text=(ROOT/'labnotes.md').read_text().split('<a id="ln-060"></a>')[1].split('\n<a id=')[0].split('\n## Supporting-record')[0]
    (folder/'plan.md').write_text('<a id="ln-060"></a>'+text)
    manifest=snapshot_sources(folder/'source')
    for name in ('scripts/run_learned_binding_bank.py','scripts/audit_learned_binding_bank.py',
                 'scripts/localize_persistent_learning.py','tests/test_learned_binding_bank.py'):
        target=folder/'source'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target);manifest[name]=file_digest(target)
    atomic_json(folder/'source-manifest.json',manifest)
    atomic_json(folder/'parents.json',{str(PARENT):file_digest(PARENT)})
    config={'fixture':fixture,'controller_steps':5 if fixture else 1500,'attack_steps':3 if fixture else 400,
        'repair_steps':3 if fixture else 400,'controller_seed':17,'train_pool_seed':17313008,
        'augmentation_seed':17313009,'attack_batch_seed':17313010,'repair_noise_seed':17313011,
        'warmup_seed':17313006,'evaluation_seed':17313007,'cpu_threads':2,
        'wall_limit_seconds':1200,'output_limit_bytes':300*1024**2,'cases':list(CASES),
        'controller':{'optimizer':'AdamW','lr':.005,'weight_decay':.001,'batch':64,'hidden':32},
        'attack_repair':{'optimizer':'Adam','lr':.003,'clip_norm':1.,'batch':16,'selected_weight':.5,
                         'other_policy_weight':.5,'task_weight':1.,'fresh_hidden':True,'noise_sd':.02},
        'precisions':['fp32','fp64'],'runtime':{'python':sys.version,'torch':str(torch.__version__),
            'platform':platform.platform()},'repair_evaluation_hidden':'Repeat actual post-event hidden across four streams'}
    atomic_json(folder/'configuration.json',config)
    return config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',action='store_true');args=parser.parse_args()
    folder=args.output;folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    def timeout(*_): raise TimeoutError('LN-060 1200-second wall limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    try:
        config=freeze(folder,args.fixture)
        pool=training_pool();warm=data(17313006,16);rows=data(17313007,128)
        if args.fixture: warm=[s[:1] for s in warm];rows=[s[:4] for s in rows]
        for r in pool: independent_check(r);assert r['split']=='train'
        for s in warm+rows:
            for r in s: independent_check(r);assert r['split']=='validation'
        assert not {r['core_sha256'] for r in pool}&{r['core_sha256'] for s in warm+rows for r in s}
        batches=attack_batches(pool,config['attack_steps'])
        atomic_json(folder/'requests.json',{'train':pool,'warmup':warm,'evaluation':rows,'attack_batches':batches})
        warm_ids,ids=tensor_ids(warm),tensor_ids(rows)
        saved=torch.load(PARENT,weights_only=True,map_location='cpu');width=saved['width']
        task=flatten_parameters(saved['model'],width);del saved
        full=train_controller(task,pool,width,config['controller_steps'],folder)
        print(json.dumps({'phase':'controller-complete','seconds':time.monotonic()-started}),flush=True)
        results=[];warm_models={}
        for precision,dtype in (('fp32',torch.float32),('fp64',torch.float64)):
            d=folder/precision;d.mkdir()
            m=LearnedBindingBank(full.to(dtype),width,4)
            warmout=stream(m,warm_ids);torch.save({'out':warmout,'state':m.snapshot()},d/'warmup.pt')
            warm_models[precision]=m
            baseline=copy.deepcopy(m);pre=panel(baseline,ids);out=stream(baseline,ids)
            scores=score(rows,out)
            result={'precision':precision,'case':'intact','scores':scores,'first_erasure':baseline.first_erasure.tolist()}
            torch.save({'out':out,'preexecution_policy_logits':pre,'final_state':baseline.snapshot()},d/'intact.pt')
            results.append(result)
        atomic_json(folder/'qualification.json',results)
        if not args.fixture and not all(r['scores']['qualified'] for r in results):
            atomic_json(folder/'summary.json',{'status':'intact-qualification-failed','results':results,
                                              'seconds':time.monotonic()-started})
            print('Intact qualification failed; attack stages stopped.',flush=True)
            return
        proposals={};repair_hidden={}
        for projection in ('coupled','uncoupled'):
            proposals[projection+'-attack']=optimize(full,pool,batches,width,projection,folder/(projection+'-attack'))
            attack=proposals[projection+'-attack']
            _,warmhidden=warm_models['fp32'].decode()
            live=LearnedBindingBank(attack['physical'],width,1,gain=attack['gain'],projection=projection,
                                    hidden=warmhidden[:1])
            trigger=next(r for r in pool if r['family']=='lookup')
            event=live.request(torch.tensor([trigger['tokens']]))
            current,h=live.decode()
            gen=torch.Generator().manual_seed(17313011)
            initial=current[0]+torch.randn(current[0].shape,generator=gen)*.02
            torch.save({'request':trigger,'out':event,'post_event_state':live.snapshot(),
                        'repair_initial':initial,'noise_seed':17313011},folder/(projection+'-repair-start.pt'))
            repair_hidden[projection]=h.clone()
            proposals[projection+'-repair']=optimize(initial,pool,batches,width,projection,folder/(projection+'-repair'))
            print(json.dumps({'phase':projection+'-attack-and-repair-complete','seconds':time.monotonic()-started,
                              'event_admissions':event['admitted'].tolist(),'erased':event['wiped'].tolist()}),flush=True)
        for precision,dtype in (('fp32',torch.float32),('fp64',torch.float64)):
            for name in CASES[1:]:
                m=copy.deepcopy(warm_models[precision])
                if name.startswith('scale-'):
                    m.rescale({'scale-negative':-1.,'scale-half':.5,'scale-double':2.}[name])
                elif name!='identity':
                    key=name if name in proposals else 'coupled-attack'
                    prop=proposals[key];m.install(prop['physical'].to(dtype),prop['gain'])
                    if name.startswith('uncoupled'):m.projection='uncoupled'
                    if name.endswith('-repair'):
                        h=repair_hidden[name.split('-')[0]].to(dtype).expand(4,-1)
                        m.hidden_bank=m.encode(h)
                        m.first_erasure.fill_(-1);m.requests=0
                    if name=='attack-sign-recode':m.recode([-1.,1.])
                    if name=='attack-symmetric-fixed-D':m.recode([1.,1.])
                    if name=='attack-joint-normalizer':m.recode([1.,1.],change_normalizer=True)
                    if name=='attack-skip-normalization':m.projection='none'
                pre=panel(m,ids);out=stream(m,ids)
                torch.save({'out':out,'preexecution_policy_logits':pre,'final_state':m.snapshot()},folder/precision/(name+'.pt'))
                results.append({'precision':precision,'case':name,'scores':score(rows,out),
                    'first_erasure':m.first_erasure.tolist(),'final_nonzero':int(torch.count_nonzero(m.bank)),
                    'final_controller_nonzero':int(torch.count_nonzero(m.bank[:,:,counts(width)[0]:])),
                    'final_hidden_nonzero':int(torch.count_nonzero(m.hidden_bank))})
        size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        if size>config['output_limit_bytes']:raise RuntimeError('Artifact output ceiling exceeded')
        atomic_json(folder/'summary.json',{'status':'fixture-complete' if args.fixture else 'complete',
            'width':width,'parameter_counts':counts(width),'seconds':time.monotonic()-started,
            'output_bytes_before_manifest':size,'results':results})
        print(json.dumps({'phase':'complete','seconds':time.monotonic()-started,'bytes':size}),flush=True)
    except BaseException as exc:
        atomic_json(folder/'failure.json',{'error':repr(exc),'seconds':time.monotonic()-started})
        raise
    finally:
        signal.alarm(0)
        atomic_json(folder/'artifact-manifest.json',{str(p.relative_to(folder)):file_digest(p)
                    for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json'})


if __name__=='__main__': main()
