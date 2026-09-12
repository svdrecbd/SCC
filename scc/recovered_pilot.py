"""Frozen small-model developmental pilot, with actual objective/resume gates."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .checkpoint import save_checkpoint, load_checkpoint, rng_state, restore_rng
from .coupling import nll
from .developmental_run import (TextBank, Streams, configure, environment, arithmetic_limits,
                               adam, evaluate, batch_fingerprint)
from .developmental_tasks import evaluation_rows
from .model import ModelConfig, Transformer
from .pilot_objectives import episode, recovered_episode, seam_episode, vector_norm
from .provenance import atomic_json, digest, source_manifest, snapshot_sources, file_digest


def configuration():
    return {'protocol':'SCC_RECOVERED_DEVELOPMENTAL_PILOT_V2.md',
            'model': {'width':256,'layers':4,'heads':4,'context_length':192,'dropout':0.},
            'seed':17,'data_seed':101,'batch_size':64,'lr':.0006,'warmup':200,
            'steps':18000,'late_start':9000,'episodes':450,'meta_batch_size':2,
            'meta_weight':.1,'inner_steps':8,'inner_lr':.001,'inner_replay':.5,
            'repair_steps':2,'repair_lr':.0001,'seam_alpha':1.,'seam_beta':.01,
            'resume_steps':[9000,18000], 'weight_steps':[1000,5000],
            'evaluation_size':128,'evaluation_seed':582019,'text_blocks':128,
            'signal_ordinals':[0,25,125,225,449], 'wall_seconds':4200,
            'output_bytes_limit':2*1024**3}


def schedule(config, arm):
    if arm == 'rule_only':
        return {}
    start = 0 if arm == 'early' else config['late_start']
    if arm not in ('early','late','seam_late'):
        raise ValueError('Unknown arm')
    span = config['steps']-config['late_start']
    if span % config['episodes']:
        raise ValueError('Schedule must have equally spaced whole episodes')
    return {start+i*(span//config['episodes']):i for i in range(config['episodes'])}


def recursive_equal(a,b):
    if isinstance(a, torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if type(a) is not type(b):
        return False
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(recursive_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(recursive_equal(x,y) for x,y in zip(a,b))
    return a==b


def train(bank,config,arm,output,device,stop_after=None,resume=None,deadline=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    contract = {'configuration':config,'arm':arm,'source':source_manifest(),
                'protocol_sha256':file_digest(Path(__file__).resolve().parents[1]/'protocols'/config['protocol']),
                'environment':environment(device),'text':bank.manifest(),
                'evidence_class':'Open developmental pilot, no test split',
                'schema':'scc-recovered-pilot/v1'}
    random.seed(config['seed']); torch.manual_seed(config['seed'])
    model = Transformer(ModelConfig(**config['model'])).to(device).train()
    optimizer = adam(model,config['lr'])
    stream = Streams(bank,config['data_seed'],device,config['batch_size'],.5)
    chain,meta_chain,completed = digest('ordinary-stream/v1'),digest('meta-stream/v1'),0
    parent_hash = None
    if resume:
        state = load_checkpoint(resume)
        if state['contract'] != contract:
            raise ValueError('Resume contract changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        stream.load_state_dict(state['stream']); restore_rng(state['rng'],device)
        completed,chain,meta_chain = state['completed_steps'],state['ordinary_chain'],state['meta_chain']
        parent_hash = file_digest(resume)
        del state
    end = config['steps'] if stop_after is None else stop_after
    if not completed < end <= config['steps']:
        raise ValueError('Invalid endpoint')
    output.mkdir(parents=True)
    atomic_json(output/'contract.json',contract)
    atomic_json(output/'lineage.json',{'resume_sha256':parent_hash,'resume_path':str(resume) if resume else None})
    scheduled = schedule(config,arm)
    started = time.monotonic()
    with (output/'steps.jsonl').open('w') as log:
        for step in range(completed,end):
            if deadline and time.monotonic() > deadline:
                raise TimeoutError('Campaign wall budget exhausted; latest resume remains available')
            stream.arithmetic_limits = arithmetic_limits(step,True)
            batch = stream.ordinary(); chain = digest([chain,batch_fingerprint(batch)])
            optimizer.zero_grad(set_to_none=True)
            lr = config['lr']*min(1.,(step+1)/config['warmup'])
            for group in optimizer.param_groups: group['lr']=lr
            ordinary = nll(model,dict(model.named_parameters()),batch)
            ordinary.backward()
            meta_value, details = None,None
            if step in scheduled:
                ordinal = scheduled[step]
                ordinary_gradients = [p.grad.detach().clone() for p in model.parameters()]
                with sdpa_kernel(SDPBackend.MATH):
                    if arm == 'seam_late':
                        value,details = seam_episode(model,bank,config,ordinal,device)
                    else:
                        data = episode(bank,config,ordinal,device)
                        value,details = recovered_episode(model,config,data,diagnostics=ordinal in config['signal_ordinals'])
                    gradients = torch.autograd.grad(value,tuple(model.parameters()))
                length = vector_norm(gradients)
                ordinary_length = vector_norm(ordinary_gradients)
                cosine = torch.stack([(a*b).sum() for a,b in zip(ordinary_gradients,gradients)]).sum()/(length*ordinary_length+1e-30)
                details.update(meta_gradient_norm=float(length),ordinary_gradient_norm=float(ordinary_length),
                               cosine_with_ordinary=float(cosine))
                for p,g in zip(model.parameters(),gradients):
                    p.grad.add_(g,alpha=config['meta_weight'])
                meta_value = float(value.detach())
                meta_chain = digest([meta_chain,details['batch_sha256']])
                del value,gradients,ordinary_gradients
                if arm != 'seam_late': del data
            length = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); completed=step+1
            record = {'step':completed,'ordinary_chain':chain,'meta_chain':meta_chain,
                      'ordinary_loss':float(ordinary.detach()),'group':batch.role,'lr':lr,
                      'gradient_norm':float(length),'meta_loss':meta_value,'meta':details,
                      'arithmetic_limits':list(stream.arithmetic_limits)}
            log.write(json.dumps(record)+'\n');log.flush()
            if completed%500==0:
                print(json.dumps({'arm':arm,'step':completed,'loss':record['ordinary_loss'],
                                  'elapsed_seconds':time.monotonic()-started}),flush=True)
            if completed in config['weight_steps']:
                save_checkpoint(output/f'weights-{completed:08d}.pt',{'schema_version':1,
                    'configuration':config,'model':model.state_dict(),'ordinary_chain':chain,'meta_chain':meta_chain})
            if completed in config['resume_steps'] or completed==end:
                # Self-contained existing checkpoint schema. The append-only log is
                # referenced, not redundantly copied into every model serialization.
                save_checkpoint(output/f'step-{completed:08d}.pt',{'schema_version':1,'contract':contract,
                    'model':model.state_dict(),'optimizer':optimizer.state_dict(),'stream':stream.state_dict(),
                    'rng':rng_state(device),'completed_steps':completed,'ordinary_chain':chain,'meta_chain':meta_chain,
                    'log_prefix_sha256':file_digest(output/'steps.jsonl'),'parent_resume_sha256':parent_hash})
    result = {'arm':arm,'completed_steps':completed,'ordinary_chain':chain,'meta_chain':meta_chain,
              'elapsed_seconds':time.monotonic()-started,'status':'complete' if completed==config['steps'] else 'paused'}
    if completed==config['steps']:
        for label,reordered in (('validation',False),('reordered_validation',True)):
            rows = evaluation_rows(config['evaluation_size'],seed=config['evaluation_seed'],reordered=reordered)
            assert not stream.tasks.seen & {r['latent_id'] for r in rows}
            result[label] = evaluate(model,bank,rows)
        result['qualified_both_layouts'] = all(result[k]['qualification']['passed'] for k in ('validation','reordered_validation'))
    atomic_json(output/'result.json',result)
    return model,result


def readiness(bank, config, output, device):
    output.mkdir(parents=True,exist_ok=False)
    fixture=copy.deepcopy(config)
    fixture.update(steps=4,late_start=2,episodes=2,batch_size=2,resume_steps=[2,4],weight_steps=[],
                   evaluation_size=2,signal_ordinals=[],inner_steps=2)
    fixture['model'].update(width=32,layers=1,heads=2)
    results={}
    for arm in ('early','late','seam_late'):
        train(bank,fixture,arm,output/(arm+'-full'),device)
        train(bank,fixture,arm,output/(arm+'-part'),device,stop_after=2)
        train(bank,fixture,arm,output/(arm+'-resume'),device,resume=output/(arm+'-part/step-00000002.pt'))
        full=load_checkpoint(output/(arm+'-full/step-00000004.pt'))
        resumed=load_checkpoint(output/(arm+'-resume/step-00000004.pt'))
        for key in ('model','optimizer','stream','rng','ordinary_chain','meta_chain'):
            if not recursive_equal(full[key],resumed[key]):
                raise AssertionError(f'{arm} resume mismatch: {key}')
        rows=[json.loads(line) for line in (output/(arm+'-full/steps.jsonl')).read_text().splitlines()]
        if len([r for r in rows if r['meta'] and r['meta']['meta_gradient_norm']>0])!=2:
            raise AssertionError('Fixture did not execute nonzero objective updates')
        results[arm]={'bitwise_model_optimizer_rng_stream_resume':True,'nonzero_meta_updates':2}
    # Numerical derivative of the actual full-size episode, including fitted
    # discrete readers. Any branch switch is therefore visible to this gate.
    torch.manual_seed(config['seed'])
    model=Transformer(ModelConfig(**config['model'])).to(device).eval()
    origin={k:v.detach().clone() for k,v in model.named_parameters()}
    data=episode(bank,config,90000,device)
    with sdpa_kernel(SDPBackend.MATH):
        value,_=recovered_episode(model,config,data)
        gradients=torch.autograd.grad(value,tuple(model.parameters()))
        length=float(vector_norm(gradients))
        checks=[]
        for epsilon in (1e-4,3e-5,1e-5):
            outcomes=[]
            for sign in (-1,1):
                with torch.no_grad():
                    for (k,p),g in zip(model.named_parameters(),gradients):p.copy_(origin[k]+sign*epsilon*g/length)
                v,_=recovered_episode(model,config,data,create_graph=False)
                outcomes.append(float(v.detach()))
            numerical=(outcomes[1]-outcomes[0])/(2*epsilon)
            checks.append({'epsilon_l2':epsilon,'numerical':numerical,'relative_error':abs(numerical-length)/length,
                           'descent':outcomes[0]<float(value.detach())})
    results['actual_episode_derivative']={'gradient_norm':length,'checks':checks,
                                          'coarse_check_is_diagnostic':True}
    atomic_json(output/'result.json',results)
    if not all(c['relative_error']<.1 and c['descent'] for c in checks[1:]):
        raise AssertionError('Actual pilot objective derivative gate failed')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--readiness-only',action='store_true')
    args=parser.parse_args()
    device=configure(args.device,2 if args.device=='cpu' else 4)
    config=configuration()
    args.output.mkdir(parents=True,exist_ok=False)
    snapshot_sources(args.output/'source')
    shutil.copyfile(Path(__file__).resolve().parents[1]/'protocols'/config['protocol'],args.output/'protocol.md')
    atomic_json(args.output/'configuration.json',config)
    bank=TextBank(args.data,blocks=config['text_blocks'])
    started=time.monotonic()
    gate=readiness(bank,config,args.output/'readiness',device)
    print(json.dumps({'readiness':gate}),flush=True)
    if args.readiness_only:
        return
    from .recovered_pilot_evaluation import challenge
    results={'arms':{},'positive_scc_result':False,'evidence_class':'Single-seed open pilot'}
    for arm in ('rule_only','early','late','seam_late'):
        model,result=train(bank,config,arm,args.output/arm,device,deadline=started+config['wall_seconds'])
        results['arms'][arm]=result
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>config['output_bytes_limit']:
            raise RuntimeError('Artifact ceiling reached')
        # Evaluate each arm's predeclared challenges, including unqualified
        # parents as diagnostics. Do not suppress negative results at a gate.
        results['arms'][arm]['challenges']=challenge(model,bank,config,args.output/('challenges-'+arm),
                                                   started+config['wall_seconds'])
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>config['output_bytes_limit']:
            raise RuntimeError('Artifact ceiling reached')
        atomic_json(args.output/'progress.json',results)
        del model
    chains=[r['ordinary_chain'] for r in results['arms'].values()]
    if len(set(chains))!=1: raise AssertionError('Ordinary streams did not match')
    if results['arms']['early']['meta_chain']!=results['arms']['late']['meta_chain']:
        raise AssertionError('SCC episode samples did not match')
    results.update(status='complete',elapsed_seconds=time.monotonic()-started,
                   source_unchanged=source_manifest()==json.loads((args.output/'source/source_manifest.json').read_text()))
    atomic_json(args.output/'result.json',results)
    if os.environ.get('GMN_RESULT_PATH'): atomic_json(os.environ['GMN_RESULT_PATH'],results)


if __name__=='__main__':main()
