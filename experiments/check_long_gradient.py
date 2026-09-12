"""Check the full 300-step stabilized-Adam derivative and its local step-size range."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from scc.coupling import post_attack_loss,task_batch
from scc.data import PreparedDataset
from scc.interventions import Evaluation,Streams,load_model,parent_receipt
from scc.provenance import atomic_json,file_digest,snapshot_sources
from scc.tokenizer import ByteTokenizer
from experiments.exact_adam_probe import adam_update,differentiable_adam

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.mkdir(parents=True)
torch.set_num_threads(4)
model,_=load_model('runs/online-06-byte-mixed/step-00008000.pt')
model.double()
normalizer=Evaluation('artifacts/retrieval-recovery/byte-prepared',16,92159)(model)['language']['nll_per_supervised_token']
streams=Streams(PreparedDataset('artifacts/retrieval-recovery/byte-prepared','train'),10405)
episodes=[[(streams.task(8,'unauthorized','disclose'),1.),(streams.task(8,'retrieval'),1/3),
           (streams.task(8,'authorized'),1/3),(streams.text(8),1/(3*normalizer))] for _ in range(300)]
rows=streams.tables.batch(32,'unauthorized')
query={'disclose':task_batch(rows,ByteTokenizer(),192,'disclose'),'policy':task_batch(rows,ByteTokenizer(),192),
       'retrieval':streams.task(32,'retrieval'),'language':streams.text(32)}
config={'mode':'escape','language_normalizer':normalizer,
        'surrogate':{'break_threshold':.5,'cap_threshold':.75,'break_temperature':2.,'cap_temperature':.25}}
atomic_json(args.output/'protocol.json',{'source_files':snapshot_sources(args.output/'source'),
            'implementation_sha256':file_digest('experiments/exact_adam_probe.py'),'script_sha256':file_digest(__file__),
            'parent':parent_receipt('runs/online-06-byte-mixed/step-00008000.pt'),'data_seed':10405,'steps':300,
            'dtype':'float64','direction':'gradient','epsilons':[1e-10,1e-11,1e-12],
            'meaning':'Full derivative of this stabilized Adam trajectory; not a claim about all attacks or all numerical settings.'})
parameters=dict(model.named_parameters())
names=tuple(parameters)
started=time.perf_counter()
with sdpa_kernel(SDPBackend.MATH):
 attacked=differentiable_adam(model,parameters,episodes,.0003,True)
 value,_=post_attack_loss(model,attacked,query,config)
 gradients=torch.autograd.grad(value,tuple(parameters.values()))
 baseline=float(value.detach())
 norm=sum(g.square().sum() for g in gradients).sqrt()
 direction={k:g.detach()/norm for k,g in zip(parameters,gradients)}
 endpoint={k:p.detach().clone() for k,p in attacked.items()}
 torch.save({'gradients':dict(zip(parameters,gradients)),'attacked':endpoint,'objective':config},args.output/'gradient-and-endpoint.pt')
 del attacked,value,gradients
 print(json.dumps({'analytic':float(norm),'baseline_loss':baseline,'gradient_seconds':time.perf_counter()-started}),flush=True)

 def forward_only(values):
  count=len(names)
  pp=tuple(values[k].detach().requires_grad_() for k in names)
  state=pp+tuple(torch.zeros_like(p) for p in pp)+tuple(torch.zeros_like(p) for p in pp)
  for index,batches in enumerate(episodes):
   state=adam_update(model,names,batches,.0003,index+1,state)
   state=tuple(p.detach().requires_grad_(i<count) for i,p in enumerate(state))
  return dict(zip(names,state[:count]))

 same=forward_only(parameters)
 maximum_delta=max(float((same[k]-endpoint[k]).abs().max()) for k in same)
 if maximum_delta != 0:raise RuntimeError(f'Forward-only recomputation differs: {maximum_delta}')
 del same,endpoint
 checks=[]
 for epsilon in [1e-10,1e-11,1e-12]:
  values=[]
  for sign in [1,-1]:
   moved={k:p.detach()+sign*epsilon*direction[k] for k,p in parameters.items()}
   phi=forward_only(moved)
   penalty,diagnostics=post_attack_loss(model,phi,query,config)
   values.append(float(penalty.detach()))
   del phi,penalty
  finite=(values[0]-values[1])/(2*epsilon)
  relative=abs(finite-float(norm))/float(norm)
  row={'epsilon':epsilon,'analytic':float(norm),'finite_difference':finite,'relative_error':relative,
       'plus_loss':values[0],'minus_loss':values[1],'passed_1_percent':relative<=.01}
  checks.append(row)
  atomic_json(args.output/'progress.json',checks)
  print(json.dumps(row),flush=True)
atomic_json(args.output/'result.json',{'checks':checks,'gradient_l2':float(norm),'baseline_loss':baseline,
            'forward_recomputation_maximum_delta':maximum_delta,'wall_seconds':time.perf_counter()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'converged_at_smallest_tested_epsilon':checks[-1]['passed_1_percent']})
