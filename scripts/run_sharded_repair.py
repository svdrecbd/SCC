"""LN-069: fixed-layout learned repair and matched relaxed-rule diagnostic."""
import argparse
import json
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
from scc.sharded_repair import start_values,restore,functional_window,functional_request
from scc.persistent_tasks import sample_request
from scc.provenance import atomic_json,file_digest,snapshot_sources
from scripts.run_learned_binding_bank import data,tensor_ids,score
from scripts.run_sharded_rewrite_cell import run_stream,pack_state
from scripts.localize_persistent_learning import independent_check

PARENT=ROOT/'artifacts/scc-sharded-rewrite-20260913-v1/full-v1/fp32/repair-after-merge-start.pt'
ARMS={'learned-binding':'learned','symbolic-binding-control':'symbolic'}


def make_data(steps,fixture):
    rng=random.Random(17313016)
    cells=[(f,'ungated',l) for f in ('lookup','parity','sum3') for l in ('original','reordered')]
    pool=[sample_request(rng,cells[i%6]).record() for i in range(4096)]
    validation=data(17313019,128)
    if fixture:validation=[s[:4] for s in validation]
    probe=[pool[i:256:4] for i in range(4)]
    if fixture:probe=[s[:4] for s in probe]
    groups={f:[i for i,r in enumerate(pool) if r['family']==f] for f in ('lookup','parity','sum3')}
    rng=random.Random(17313017);schedule=[]
    for _ in range(steps):
        positions=[]
        for j in range(2):
            selected=[rng.choice(groups[f]) for f,n in (('lookup',4),('parity',2),('sum3',2)) for _ in range(n)]
            rng.shuffle(selected);positions.append(selected)
        schedule.append([[positions[j][i] for j in range(2)] for i in range(8)])
    for r in pool:independent_check(r);assert r['split']=='train'
    for s in validation:
        for r in s:independent_check(r);assert r['split']=='validation'
    assert not {r['core_sha256'] for r in pool}&{r['core_sha256'] for s in validation for r in s}
    return {'train_pool':pool,'validation':validation,'train_probe':probe,'schedule':schedule}


def optimize(initial,hidden,requests,rule,dest):
    dest.mkdir();q=torch.nn.Parameter(initial.clone())
    opt=torch.optim.Adam([q],lr=.003)
    pool=requests['train_pool'];ids=torch.tensor([r['tokens'] for r in pool]);labels=torch.tensor([r['label'] for r in pool])
    is_lookup=torch.tensor([r['family']=='lookup' for r in pool]);logs=[]
    h=hidden.repeat(2,1);m=q.numel()//2
    def save(step):
        torch.save({'payload':q.detach().clone(),'step':step,'optimizer':opt.state_dict()},dest/f'update-{step:04d}.pt')
    save(0)
    for step,selected in enumerate(requests['schedule'],1):
        indices=torch.tensor(selected)
        out,committed,_=functional_window(q,h,ids[indices],128,rule)
        raw=out['policy_logits'];target=is_lookup[indices]
        desired=raw.new_tensor([1.,0.,0.,1.]).expand_as(raw).clone();desired[:,:,1][target]=1.
        policy_losses=[];selected_values=[];other_values=[]
        for j in range(2):
            mask=torch.zeros_like(raw[:,j],dtype=torch.bool);mask[:,1]=target[:,j]
            selected_loss=F.binary_cross_entropy_with_logits(raw[:,j][mask],desired[:,j][mask])
            other_loss=F.binary_cross_entropy_with_logits(raw[:,j][~mask],desired[:,j][~mask])
            policy_losses.append(.5*selected_loss+.5*other_loss)
            selected_values.append(float(selected_loss.detach()));other_values.append(float(other_loss.detach()))
        task_losses=[F.cross_entropy(out['logits'][:,j],labels[indices[:,j]]) for j in range(2)]
        loss=torch.stack(task_losses+policy_losses).sum()/2
        opt.zero_grad();loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_([q],1.,error_if_nonfinite=True));opt.step()
        logs.append({'update':step,'loss_before_update':float(loss.detach()),
            'task_ce_by_position':[float(v.detach()) for v in task_losses],
            'selected_bce_by_position':selected_values,'other_bce_by_position':other_values,
            'gradient_norm_before_clip':grad,
            'selected_admissions_by_position':[int((out['admitted'][:,j,1]&target[:,j]).sum()) for j in range(2)],
            'window_task_correct':int((out['logits'].argmax(-1)==labels[indices]).sum()),
            'committed_payload_difference_mean':float((committed[:,0]-committed[:,1]).detach().norm(dim=-1).mean()),
            'candidate_payload_difference_norm':float((q[:m]-q[m:]).detach().norm()),
            'physical_padding_after_update':float(q[-1].detach())})
        if step in (250,500,1000,len(requests['schedule'])):save(step)
    atomic_json(dest/'training.json',logs)
    atomic_json(dest/'parameter-changes.json',{'physical_values_exposed':q.numel(),
        'physical_values_changed':int(torch.count_nonzero(q.detach()-initial)),
        'change_l2':float((q.detach()-initial).norm()),'padding_initial':float(initial[-1]),
        'padding_final':float(q[-1].detach())})
    return q.detach().clone()


@torch.no_grad()
def compare_functional(payload,start,rule,ids,actual):
    h=start_values(start)[1].to(payload.dtype)
    shards=payload.reshape(1,2,-1).expand(len(ids),-1,-1);outs=[]
    for j in range(ids.shape[1]):
        out,shards,h=functional_request(shards,h,ids[:,j],128,rule);outs.append(out)
    out={k:torch.stack([x[k] for x in outs],1) for k in outs[0]}
    error=float((out['logits']-actual['logits']).abs().max())
    policy_error=float((out['policy_logits']-actual['policy_logits']).abs().max())
    decisions=int((out['logits'].argmax(-1)!=actual['logits'].argmax(-1)).sum())
    admissions=int((out['admitted']!=actual['admitted']).sum())
    tolerance=1e-4 if payload.dtype==torch.float32 else 1e-10
    return {'task_logit_max':error,'policy_logit_max':policy_error,'task_decision_disagreements':decisions,
            'admission_disagreements':admissions,'passed':max(error,policy_error)<=tolerance and decisions==admissions==0}


def freeze(folder,fixture):
    entry=(ROOT/'labnotes.md').read_text().split('<a id="ln-069"></a>')[1].split('\n<a id=')[0].split('\n## Supporting-record')[0]
    (folder/'plan.md').write_text('<a id="ln-069"></a>'+entry)
    manifest=snapshot_sources(folder/'source')
    for name in ('scripts/run_sharded_repair.py','scripts/audit_sharded_repair.py','scripts/run_sharded_rewrite_cell.py',
                 'scripts/rewrite_rank_certificate.py','scripts/run_learned_binding_bank.py',
                 'scripts/localize_persistent_learning.py','scripts/audit_sharded_rewrite_cell.py',
                 'scripts/audit_rewrite_binding_cell.py','tests/test_sharded_repair.py'):
        dest=folder/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        manifest[name]=file_digest(dest)
    atomic_json(folder/'source-manifest.json',manifest)
    atomic_json(folder/'parents.json',{str(PARENT):file_digest(PARENT)})
    shutil.copyfile(PARENT,folder/'merged-start-parent.pt')
    config={'plan':'LN-069','fixture':fixture,'steps':3 if fixture else 1000,'arms':ARMS,
        'optimizer':'Adam','lr':.003,'clip_norm':1.,'weight_decay':0.,'initialization_noise':0.,
        'batch_streams':8,'requests_per_window':2,'tokens_per_request':19,
        'pool_size':4096,'pool_seed':17313016,'schedule_seed':17313017,'validation_seed':17313019,
        'physical_payload_values':80518,'decoded_learned_coefficients':80517,'gain':1.,
        'bank_shape':[4,8,40259],'hidden_bank_shape':[4,8,64],
        'normalizer_sign':1.,'writer_sign':1.,'code_sign':1.,'execution':'commit',
        'checkpoint_updates':[0,3] if fixture else [0,250,500,1000],
        'precisions':['fp32','fp64'],'cpu_threads':2,'wall_limit_seconds':300,'output_limit_bytes':120*1024**2,
        'runtime':{'python':sys.version,'torch':str(torch.__version__),'platform':platform.platform()}}
    atomic_json(folder/'configuration.json',config);return config


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',action='store_true')
    args=p.parse_args();folder=args.output;folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    def timeout(*_):raise TimeoutError('LN-069 300-second wall limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    try:
        config=freeze(folder,args.fixture)
        parent=torch.load(folder/'merged-start-parent.pt',weights_only=True,map_location='cpu')
        assert parent['out']['admitted'][:,1].all()
        start=parent['post_event_state'];initial,hidden=start_values(start);del parent
        torch.save({'payload':initial,'hidden':hidden,'state':start},folder/'repair-origin.pt')
        requests=make_data(config['steps'],args.fixture);atomic_json(folder/'requests.json',requests)
        endpoints={}
        for arm,rule in ARMS.items():
            endpoints[arm]=optimize(initial,hidden,requests,rule,folder/arm)
            print(json.dumps({'phase':arm+'-trained','seconds':time.monotonic()-started}),flush=True)
        results=[]
        for arm,rule in ARMS.items():
            for endpoint,payload in (('initial',initial),('final',endpoints[arm])):
                for precision,dtype in (('fp32',torch.float32),('fp64',torch.float64)):
                    for panel in ('validation','train_probe') if precision=='fp32' else ('validation',):
                        rows=requests[panel];ids=tensor_ids(rows)
                        model=restore(start,dtype,rule,payload)
                        initial_state=pack_state(model)
                        out,traces=run_stream(model,ids,fixture=args.fixture)
                        scores=score(rows,out)
                        target=torch.tensor([[r['family']=='lookup' for r in s] for s in rows])
                        desired=torch.tensor([True,False,False,True]).expand_as(out['admitted']).clone();desired[:,:,1]|=target
                        correspondence=compare_functional(payload.to(dtype),start,rule,ids,out)
                        task_gate=len(scores['cells'])==6 and all(c['qualified'] for c in scores['cells'].values())
                        eligible=panel=='validation' and arm=='learned-binding' and task_gate and scores['target_correct_answers']/scores['target_n']>=.95
                        name=f'{arm}-{endpoint}-{precision}-{panel}'
                        torch.save({'out':out,'traces':traces,'initial_state':initial_state,'final_state':pack_state(model)},folder/(name+'.pt'))
                        results.append({'name':name,'arm':arm,'endpoint':endpoint,'precision':precision,'panel':panel,
                            'scores':scores,'exception_rule_errors':int((out['admitted']!=desired).sum()),
                            'task_qualified':task_gate,'eligible_repair_escape':eligible,'correspondence':correspondence,
                            'first_erasure':model.first_erasure.tolist()})
        size=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        assert size<=config['output_limit_bytes'],'Output ceiling exceeded'
        valid=all(r['correspondence']['passed'] for r in results)
        status=('fixture-complete' if args.fixture else 'complete') if valid else 'numerical-validation-failed'
        atomic_json(folder/'summary.json',{'status':status,'results':results,'seconds':time.monotonic()-started,'bytes_before_manifest':size})
        if not valid:raise RuntimeError('Functional/runtime correspondence failed; all outputs preserved')
        print(json.dumps({'phase':'complete','seconds':time.monotonic()-started,'bytes':size}),flush=True)
    except BaseException as exc:
        atomic_json(folder/'failure.json',{'error':repr(exc),'seconds':time.monotonic()-started});raise
    finally:
        signal.alarm(0)
        atomic_json(folder/'artifact-manifest.json',{str(p.relative_to(folder)):file_digest(p)
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json'})


if __name__=='__main__':main()
