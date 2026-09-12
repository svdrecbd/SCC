"""Short exact-package GPU check; implementation evidence, not SCC evidence."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scc.coupling import nll
from scc.developmental_run import configure, TextBank, Streams
from scc.memory_factorial import CONDITIONS, factorial_config, intervention_conditions, reinterpret
from scc.portfolio_models import PortfolioModel
from scc.portfolio_objective import make_contraction_episode, latent_collapse_objective
from scc.projected_coupling import make_projected_episode, projected_rollout
from scc.provenance import atomic_json, source_manifest
from scc.transition_path import EditStepper


def inline_summary(result):
    result = {k:v for k,v in result.items() if k!='checks'} | {'checks':[
        {k:v for k,v in row.items() if k not in ('objective','projected_geometry','stock_edit')}
        for row in result['checks']]}
    assert len(json.dumps(result).encode()) < 32*1024
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--device',choices=['cuda'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    device=configure(a.device,4)
    assert json.loads(a.parents.read_text())=={}
    expected=json.loads((ROOT/'transport-source-manifest.json').read_text())
    for name,sha in expected.items():
        assert hashlib.file_digest((ROOT/name).open('rb'),'sha256').hexdigest()==sha,name
    assert source_manifest()=={n:v for n,v in expected.items() if n.startswith('scc/') or n in ('pyproject.toml','uv.lock','.python-version')}
    data=json.loads((ROOT/'transport-data-manifest.json').read_text())
    for name,sha in data.items():
        assert hashlib.file_digest((a.data/name).open('rb'),'sha256').hexdigest()==sha,name
    bank=TextBank(a.data,2)
    rows=[];start=time.monotonic();initial=None
    for condition in CONDITIONS:
        for ordinal in (3,4):
            torch.manual_seed(17)
            model=PortfolioModel(factorial_config(condition)).to(device)
            tensors={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
            if initial is None:initial=tensors
            assert all(torch.equal(v,initial[n]) for n,v in tensors.items())
            torch.cuda.reset_peak_memory_stats()
            ordinary=Streams(bank,101,device,64,.5).ordinary()
            parameters=dict(model.named_parameters())
            base=nll(model,parameters,ordinary)
            episode=make_contraction_episode(bank,ordinal,device,batch_size=2,inner_steps=8)
            value,record=latent_collapse_objective(model,episode)
            (base+value).backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert norm>0 and torch.isfinite(value)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            del value,base,optimizer,parameters
            projected_episode=make_projected_episode(bank,600009,device,batch_size=2,inner_steps=1,radius=.05)
            projected_episode['configuration']['scope']='core'
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            changed,_,geometry=projected_rollout(model,projected_episode,create_graph=False)
            with torch.no_grad():
                for name,p in model.named_parameters():p.copy_(changed[name])
            assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if not n.startswith('cells.'))
            stepper=EditStepper(model,bank,batch_size=16,scope='core',seed=483721,lr=4e-4)
            edit=stepper.advance()
            graph_conditions=[]
            for destination in intervention_conditions(condition).values():
                graph=reinterpret(model,destination)
                logits=graph(ordinary.tokens)
                assert torch.isfinite(logits).all()
                graph_conditions.append(destination)
                del graph,logits
            rows.append({'condition':condition,'scope':episode['configuration']['inner_scope'],
                         'gradient_norm':float(norm),'ordinary_batch_size':len(ordinary.tokens),
                         'inner_steps':len(episode['changes']),'initial_tensors_matched':True,
                         'graph_conditions':graph_conditions,'stock_edit':edit,'projected_geometry':geometry,
                         'core_mask_verified':True,'objective':record,
                         'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()})
            atomic_json(a.output/'progress.json',rows)
            del model,stepper,changed,before,episode,record,projected_episode
            gc.collect();torch.cuda.empty_cache()
    result={'ok':True,'checks':rows,'source_files_verified':len(expected),'data_files_verified':len(data),
            'elapsed_seconds':time.monotonic()-start,'runtime_downloads':False,
            'scope':'Full-sized CUDA ordinary/coupling updates, projected/core edits and graph switches at initialization; not trained-model or SCC evidence.'}
    atomic_json(a.output/'result.json',result)
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],inline_summary(result))
    print(json.dumps({'ok':True,'checks':len(rows),'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
