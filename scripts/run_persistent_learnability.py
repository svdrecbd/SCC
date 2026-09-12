"""Ordinary-learning gate for a persistent computational substrate. No coupling."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scc.persistent_matrix import MatrixConfig, LiveMatrix
from scc.persistent_tasks import (training_requests,evaluation_requests,input_code,initialize_matrix,
                                tensors,run_window,summarize_records,TOKENS_PER_REQUEST)
from scc.provenance import atomic_json,digest,file_digest,snapshot_sources,source_manifest,canonical_json


@torch.no_grad()
def evaluate(weights, config, code, folder, *, seed, per_cell, streams):
    folder.mkdir()
    requests=evaluation_requests(seed,per_cell=per_cell,streams=streams)
    ids,labels=tensors(requests,device=weights.device)
    outputs,states,controls=run_window(weights,ids,config,code)
    if not torch.isfinite(outputs).all() or not torch.isfinite(states).all():
        raise ValueError('Nonfinite evaluation state')
    predictions=outputs.argmax(-1).cpu().tolist();scores=outputs.cpu().tolist();records=[]
    length=ids.shape[1]
    for step in range(length):
        for stream in range(streams):
            r=requests[stream][step]
            records.append({**r.record(),'stream':stream,'request_index':step,'late_half':step>=length//2,
                            'prediction':predictions[stream][step],'logits':scores[stream][step]})
    with (folder/'predictions.jsonl').open('w') as f:
        for r in records:f.write(canonical_json(r)+'\n')
    # Independently exercise the object interface for the whole first stream.
    live=LiveMatrix(weights,config)
    maximum_error=0.;same_decisions=True
    for step,r in enumerate(requests[0]):
        for token in r.tokens:out,_=live.tick(code[token])
        maximum_error=max(maximum_error,float((out-outputs[0,step]).abs().max()))
        same_decisions &= int(out.argmax())==predictions[0][step]
    if not same_decisions or maximum_error>1e-4:
        raise ValueError('Batched and single live executions disagree')
    result={**summarize_records(records),'configuration':asdict(config),'seed':seed,'per_cell':per_cell,
            'streams':streams,'consecutive_requests_per_stream':length,'clean_resets_within_stream':0,
            'single_stream_decisions_match':same_decisions,'single_stream_maximum_logit_error':maximum_error,
            'prediction_sha256':file_digest(folder/'predictions.jsonl'),'final_state_variance':float(states.var(-1).mean()),
            'initial_state_variance':float(weights.var(-1).mean()),'positive_scc_result':False}
    torch.save({'current_weights':states.cpu(),'configuration':asdict(config),'steps_per_stream':length*TOKENS_PER_REQUEST},folder/'final-live-states.pt')
    atomic_json(folder/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--rule',choices=['soft','copy'],default='copy')
    p.add_argument('--width',type=int,default=64);p.add_argument('--encoding',choices=['token','anchor'],default='anchor')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--data-seed',type=int,default=24017)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--window',type=int,default=4);p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--gate-bias',type=float,default=0.);p.add_argument('--initial-scale',type=float,default=.5)
    p.add_argument('--wall-seconds',type=float,default=3300);p.add_argument('--eval-per-cell',type=int,default=128)
    p.add_argument('--eval-streams',type=int,default=16);p.add_argument('--no-curriculum',action='store_true')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--data',type=Path);p.add_argument('--parents',type=Path)
    a=p.parse_args()
    if a.steps<1 or a.batch<1 or a.window<1 or a.lr<=0 or a.wall_seconds<=0:raise ValueError('Invalid resource configuration')
    a.output.mkdir(parents=True,exist_ok=False)
    source=snapshot_sources(a.output/'source');shutil.copyfile(__file__,a.output/'runner.py')
    protocol=ROOT/'protocols/SCC_PERSISTENT_LEARNABILITY_V1.md';shutil.copyfile(protocol,a.output/'protocol.md')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    if a.device=='cuda':
        if not torch.cuda.is_available():raise ValueError('CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=MatrixConfig(a.width,4,a.rule)
    initial=initialize_matrix(config,a.seed,gate_bias=a.gate_bias,scale=a.initial_scale,device=a.device)
    weights=torch.nn.Parameter(initial.clone());code=input_code(a.width,a.encoding,device=a.device)
    optimizer=torch.optim.Adam([weights],lr=a.lr)
    torch.save({'initial_weights':initial.cpu(),'configuration':asdict(config)},a.output/'initial.pt')
    clock=time.monotonic();chain=digest('persistent-learnability/v1');completed=0;failure=None
    try:
        with (a.output/'training.jsonl').open('w',buffering=1) as log:
            for step in range(a.steps):
                if time.monotonic()-clock>a.wall_seconds:break
                requests=training_requests(a.data_seed,step,a.batch,a.window,curriculum=not a.no_curriculum)
                ids,labels=tensors(requests,device=a.device)
                optimizer.zero_grad(set_to_none=True)
                outputs,state,control=run_window(weights,ids,config,code,surrogate_backward=a.rule=='copy')
                losses=torch.nn.functional.cross_entropy(outputs.reshape(-1,4),labels.reshape(-1),reduction='none').reshape_as(labels)
                loss=losses.mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([weights],1.,error_if_nonfinite=True)
                optimizer.step()
                if not torch.isfinite(weights).all():raise ValueError('Nonfinite learned initial state')
                completed=step+1
                record={'step':completed,'loss':float(loss.detach()),'accuracy':float((outputs.argmax(-1)==labels).float().mean()),
                        'gradient_norm_before_clip':float(norm),'loss_by_request':losses.detach().mean(0).tolist(),
                        'sample_sha256':digest([[r.record() for r in stream] for stream in requests]),
                        'elapsed_seconds':time.monotonic()-clock,'positive_gate_logit_count':int(control['enabled_count']),
                        'positive_gate_distinct_address_count':int(control['distinct_address_count'])}
                chain=digest({'previous':chain,'record':record});log.write(canonical_json({**record,'chain':chain})+'\n')
                if completed==1 or completed%100==0:print(json.dumps({'step':completed,'loss':record['loss'],'accuracy':record['accuracy'],'elapsed_seconds':record['elapsed_seconds']}),flush=True)
                if completed in (2000,4000):torch.save({'weights':weights.detach().cpu(),'optimizer':optimizer.state_dict(),'step':completed,'configuration':asdict(config)},a.output/f'stage-{completed}.pt')
        training_seconds=time.monotonic()-clock
        torch.save({'weights':weights.detach().cpu(),'optimizer':optimizer.state_dict(),'step':completed,'configuration':asdict(config)},a.output/'trained.pt')
        evaluation=evaluate(weights.detach(),config,code,a.output/'evaluation',seed=713904,per_cell=a.eval_per_cell,streams=a.eval_streams)
        complete=completed==a.steps
        declared=(a.steps==6000 and a.batch==32 and a.window==4 and a.width in (64,128)
                  and a.lr in (.003,.01) and a.seed==17 and a.data_seed==24017
                  and a.encoding=='anchor' and a.gate_bias==0. and a.initial_scale==.5
                  and not a.no_curriculum and a.eval_per_cell==128 and a.eval_streams==16)
        result={'schema':'persistent-learnability/v1','status':'complete' if complete else 'training_wall_limit',
                'training_complete':complete,'completed_steps':completed,'requested_steps':a.steps,
                'qualified_learnability':declared and complete and evaluation['qualified'],'evaluation':evaluation,
                'declared_screen_configuration':declared,
                'training_seconds':training_seconds,'elapsed_seconds':time.monotonic()-clock,
                'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                'parameter_count':weights.numel(),'source_sha256':digest(source),'runner_sha256':file_digest(a.output/'runner.py'),
                'protocol_sha256':file_digest(a.output/'protocol.md'),'training_chain':chain,
                'training_log_sha256':file_digest(a.output/'training.jsonl'),'positive_scc_result':False,
                'coupling_trained':False,'protection_removal_tested':False,'evidence_class':'Open learnability calibration, not SCC evidence',
                'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':a.device,
                               'gpu':torch.cuda.get_device_name() if a.device=='cuda' else None,
                               'peak_cuda_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None}}
        assert source_manifest()==source
        atomic_json(a.output/'result.json',result)
        inline={'ok':complete,'scientific_status':result['status'],'qualified_learnability':result['qualified_learnability'],
                'completed_steps':completed,'declared_screen_configuration':declared,'positive_scc_result':False}
        if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],inline)
        print(json.dumps(inline),flush=True)
    except Exception as error:
        atomic_json(a.output/'failure.json',{'type':type(error).__name__,'message':str(error),'completed_steps':completed,
                                          'elapsed_seconds':time.monotonic()-clock,'positive_scc_result':False})
        torch.save({'weights':weights.detach().cpu(),'step':completed,'configuration':asdict(config)},a.output/'failed-stage.pt')
        raise


if __name__=='__main__':main()
