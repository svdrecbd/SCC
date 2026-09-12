"""Probe smaller perturbations using the saved exact gradient, without recomputing it."""

import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from scc.coupling import post_attack_loss,task_batch
from scc.data import PreparedDataset
from scc.interventions import Streams,load_model
from scc.provenance import atomic_json,file_digest
from scc.tokenizer import ByteTokenizer
from experiments.exact_adam_probe import adam_update

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
torch.set_num_threads(4)
model,_=load_model('runs/online-06-byte-mixed/step-00008000.pt');model.double()
saved_path=Path('artifacts/strong-attack/long-gradient-check/gradient-and-endpoint.pt')
saved=torch.load(saved_path,weights_only=False)
config=saved['objective'];normalizer=config['language_normalizer']
streams=Streams(PreparedDataset('artifacts/retrieval-recovery/byte-prepared','train'),10405)
episodes=[[(streams.task(8,'unauthorized','disclose'),1.),(streams.task(8,'retrieval'),1/3),
           (streams.task(8,'authorized'),1/3),(streams.text(8),1/(3*normalizer))] for _ in range(300)]
rows=streams.tables.batch(32,'unauthorized')
query={'disclose':task_batch(rows,ByteTokenizer(),192,'disclose'),'policy':task_batch(rows,ByteTokenizer(),192),
       'retrieval':streams.task(32,'retrieval'),'language':streams.text(32)}
parameters=dict(model.named_parameters());names=tuple(parameters);count=len(names)
norm=sum(g.square().sum() for g in saved['gradients'].values()).sqrt()
direction={k:g/norm for k,g in saved['gradients'].items()}

def forward_only(values):
    pp=tuple(values[k].detach().requires_grad_() for k in names)
    state=pp+tuple(torch.zeros_like(p) for p in pp)+tuple(torch.zeros_like(p) for p in pp)
    for index,batches in enumerate(episodes):
        state=adam_update(model,names,batches,.0003,index+1,state)
        state=tuple(p.detach().requires_grad_(i<count) for i,p in enumerate(state))
    return dict(zip(names,state[:count]))

started=time.perf_counter();checks=[]
with sdpa_kernel(SDPBackend.MATH):
    same=forward_only(parameters)
    maximum_delta=max(float((same[k].detach()-saved['attacked'][k]).abs().max()) for k in same)
    assert maximum_delta==0,maximum_delta
    del same
    for epsilon in [1e-13,1e-14]:
        values=[];changed=[]
        for sign in [1,-1]:
            moved={k:p.detach()+sign*epsilon*direction[k] for k,p in parameters.items()}
            changed.append(sum(int((moved[k]!=p.detach()).sum()) for k,p in parameters.items()))
            phi=forward_only(moved)
            penalty,_=post_attack_loss(model,phi,query,config)
            values.append(float(penalty.detach()));del phi,penalty
        finite=(values[0]-values[1])/(2*epsilon)
        relative=abs(finite-float(norm))/float(norm)
        row={'epsilon':epsilon,'analytic':float(norm),'finite_difference':finite,'relative_error':relative,
             'plus_loss':values[0],'minus_loss':values[1],'changed_parameter_counts':changed,'passed_1_percent':relative<=.01}
        checks.append(row);print(json.dumps(row),flush=True)
atomic_json(args.output,{'checks':checks,'forward_recomputation_maximum_delta':maximum_delta,
            'saved_gradient_sha256':file_digest(saved_path),'script_sha256':file_digest(__file__),
            'implementation_sha256':file_digest('experiments/exact_adam_probe.py'),'wall_seconds':time.perf_counter()-started})
