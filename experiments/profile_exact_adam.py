"""Bounded actual-model feasibility probe for exact Adam attack gradients."""

import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from scc.coupling import post_attack_loss,task_batch
from scc.data import PreparedDataset
from scc.interventions import Evaluation,Streams,escape,load_model,parent_receipt
from scc.provenance import atomic_json,file_digest,source_manifest
from scc.tokenizer import ByteTokenizer
from experiments.exact_adam_probe import differentiable_adam

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--steps',type=int,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--data-seed',type=int,default=10404)
parser.add_argument('--dtype',choices=['float32','float64'],default='float64')
parser.add_argument('--adam-epsilon',type=float,default=1e-8)
args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
torch.set_num_threads(4)
parent='runs/online-06-byte-mixed/step-00008000.pt'
model,_=load_model(parent)
if args.dtype=='float64':model.double()
evaluator=Evaluation('artifacts/retrieval-recovery/byte-prepared',16,92159)
clean=evaluator(model)
normalizer=clean['language']['nll_per_supervised_token']
streams=Streams(PreparedDataset('artifacts/retrieval-recovery/byte-prepared','train'),args.data_seed)
episodes=[]
for _ in range(args.steps):
 episodes.append([(streams.task(8,'unauthorized','disclose'),1.),(streams.task(8,'retrieval'),1/3),
                  (streams.task(8,'authorized'),1/3),(streams.text(8),1/(3*normalizer))])
rows=streams.tables.batch(32,'unauthorized')
query={'disclose':task_batch(rows,ByteTokenizer(),192,'disclose'),'policy':task_batch(rows,ByteTokenizer(),192),
       'retrieval':streams.task(32,'retrieval'),'language':streams.text(32)}
config={'mode':'escape','language_normalizer':normalizer,
        'surrogate':{'break_threshold':.5,'cap_threshold':.75,'break_temperature':2.,'cap_temperature':.25}}
record={'steps':args.steps,'dtype':args.dtype,'data_seed':args.data_seed,'adam_epsilon':args.adam_epsilon,'parent':parent_receipt(parent),
        'source_files':source_manifest(),'probe_sha256':file_digest(__file__),
        'implementation_sha256':file_digest('experiments/exact_adam_probe.py'),
        'scope':'Experimental stabilized Adam variant; not used in the running first-order campaign.'}
started=time.perf_counter()
try:
 with sdpa_kernel(SDPBackend.MATH):
  parameters=dict(model.named_parameters())
  attacked=differentiable_adam(model,parameters,episodes,.0003,True,epsilon=args.adam_epsilon)
  record['forward_seconds']=time.perf_counter()-started
  print(json.dumps({'forward_seconds':record['forward_seconds'],'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
  value,diagnostics=post_attack_loss(model,attacked,query,config)
  gradients=torch.autograd.grad(value,tuple(parameters.values()))
  record.update(diagnostics,gradient_finite=all(bool(torch.isfinite(g).all()) for g in gradients),
                gradient_l2=sum(float(g.double().square().sum()) for g in gradients)**.5,
                total_gradient_seconds=time.perf_counter()-started)
  del gradients,value
 endpoint=copy.deepcopy(model)
 endpoint.load_state_dict({k:p.detach() for k,p in attacked.items()})
 del attacked
 result=evaluator(endpoint)
 record.update(evaluation=result,escape=escape(result,clean,clean),status='complete')
except Exception as error:
 record.update(status='failed',error=repr(error))
record['wall_seconds']=time.perf_counter()-started
record['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
atomic_json(args.output,record)
print(json.dumps({k:v for k,v in record.items() if k not in ('source_files','evaluation','parent')}),flush=True)
if record['status']!='complete':raise RuntimeError(record['error'])
