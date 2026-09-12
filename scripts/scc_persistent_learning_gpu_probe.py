"""Short source-verified CUDA gate for the ordinary persistent-matrix screen."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scc.persistent_matrix import MatrixConfig
from scc.persistent_tasks import training_requests,input_code,initialize_matrix,tensors,run_window
from scc.provenance import atomic_json,file_digest,source_manifest,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',choices=['cuda'],required=True);p.add_argument('--data',type=Path);p.add_argument('--parents',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    expected=json.loads((ROOT/'transport-source-manifest.json').read_text())
    assert all(file_digest(ROOT/n)==sha for n,sha in expected.items())
    assert source_manifest()=={n:s for n,s in expected.items() if n.startswith('scc/') or n in ('pyproject.toml','uv.lock','.python-version')}
    if a.parents:assert json.loads(a.parents.read_text())=={}
    assert torch.cuda.is_available();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();checks=[]
    for width in (64,128):
        initials=[]
        for rule in ('soft','copy'):
            for lr in (.003,.01):
                config=MatrixConfig(width,4,rule);initial=initialize_matrix(config,17,device='cuda')
                if initials:assert torch.equal(initial,initials[0])
                else:initials.append(initial.clone())
                weights=torch.nn.Parameter(initial.clone());code=input_code(width,'anchor',device='cuda')
                optimizer=torch.optim.Adam([weights],lr=lr);torch.cuda.reset_peak_memory_stats();rows=[]
                for ordinal in (0,2001):
                    requests=training_requests(24017,ordinal,32,4);ids,labels=tensors(requests,device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    outputs,state,control=run_window(weights,ids,config,code,surrogate_backward=rule=='copy')
                    if rule=='copy':
                        with torch.no_grad():hard,hard_state,_=run_window(weights,ids,config,code)
                        assert torch.equal(outputs,hard) and torch.equal(state,hard_state)
                    loss=torch.nn.functional.cross_entropy(outputs.reshape(-1,4),labels.reshape(-1));loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_([weights],1.,error_if_nonfinite=True)
                    assert torch.isfinite(loss) and norm>0
                    optimizer.step();assert torch.isfinite(weights).all()
                    rows.append({'ordinal':ordinal,'loss':float(loss.detach()),'gradient_norm':float(norm)})
                checks.append({'width':width,'rule':rule,'lr':lr,'batch':32,'window':4,'updates':rows,
                               'initials_matched_within_width':True,'hard_forward_checked':rule=='copy',
                               'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
                atomic_json(a.output/'progress.json',checks)
                del weights,optimizer,state,outputs
        del initials
    fixtures=[]
    for rule in ('soft','copy'):
        folder=a.output/('fixture-'+rule)
        command=[sys.executable,str(ROOT/'scripts/run_persistent_learnability.py'),'--output',str(folder),
                 '--device','cuda','--rule',rule,'--width','64','--steps','8','--batch','8','--window','4',
                 '--eval-per-cell','8','--eval-streams','4','--wall-seconds','90','--no-curriculum']
        env={k:v for k,v in os.environ.items() if k!='GMN_RESULT_PATH'}
        with (a.output/(rule+'-fixture.log')).open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
        result=json.loads((folder/'result.json').read_text())
        assert result['training_complete'] and not result['qualified_learnability'] and not result['declared_screen_configuration']
        assert result['evaluation']['single_stream_decisions_match']
        fixtures.append({'rule':rule,'path':folder.name,'result_sha256':file_digest(folder/'result.json'),
                         'training_updates':result['completed_steps'],'predictions':result['evaluation']['records'],
                         'single_stream_maximum_logit_error':result['evaluation']['single_stream_maximum_logit_error']})
    assert all(file_digest(ROOT/n)==sha for n,sha in expected.items())
    result={'ok':True,'schema':'persistent-learning-gpu-check/v1','checks':checks,'fixtures':fixtures,
            'source_files_verified':len(expected),'source_manifest_sha256':digest(expected),
            'elapsed_seconds':time.monotonic()-start,'scope':'Full-sized numerical updates and complete tiny runner fixtures only',
            'positive_scc_result':False,'qualified_learned_models':0}
    atomic_json(a.output/'result.json',result)
    inline={'ok':True,'checks':len(checks),'fixtures':len(fixtures),'elapsed_seconds':result['elapsed_seconds'],
            'source_files_verified':len(expected),'positive_scc_result':False}
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],inline)
    print(json.dumps(inline),flush=True)


if __name__=='__main__':main()
