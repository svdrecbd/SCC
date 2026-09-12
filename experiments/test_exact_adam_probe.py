"""Small validation of the experimental exact-Adam path, outside the frozen campaign."""

import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from scc.coupling import Batch,nll
from scc.model import ModelConfig,Transformer
from experiments.exact_adam_probe import differentiable_adam

torch.set_num_threads(1)
torch.manual_seed(393)
model=Transformer(ModelConfig(width=8,heads=2,layers=1,vocab_size=24,context_length=8)).double()
batch=Batch(torch.tensor([[1,4,5,6],[1,7,8,9]]),torch.tensor([[4,5,6,2],[7,8,9,2]]),'support')
query=Batch(batch.tokens.flip(1),batch.targets,'query')
parameters=dict(model.named_parameters())
episodes=[[(batch,1.)]]*4
with sdpa_kernel(SDPBackend.MATH):
 a=differentiable_adam(model,parameters,episodes,.001,False)
 b=differentiable_adam(model,parameters,episodes,.001,True)
 ga=torch.autograd.grad(nll(model,a,query),tuple(parameters.values()))
 gb=torch.autograd.grad(nll(model,b,query),tuple(parameters.values()))
 for x,y in zip(ga,gb): torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-8)
 expected=copy.deepcopy(model)
 optimizer=torch.optim.AdamW(expected.parameters(),lr=.001,betas=(.9,.95),weight_decay=0.,foreach=False)
 for _ in range(4):
  optimizer.zero_grad(set_to_none=True)
  nll(expected,dict(expected.named_parameters()),batch).backward()
  torch.nn.utils.clip_grad_norm_(expected.parameters(),1.)
  optimizer.step()
 for name,p in expected.named_parameters():torch.testing.assert_close(a[name],p,atol=1e-12,rtol=1e-10)
 direction={k:torch.randn_like(p) for k,p in parameters.items()}
 norm=sum(d.square().sum() for d in direction.values()).sqrt()
 direction={k:d/norm for k,d in direction.items()}
 analytic=sum((g*direction[k]).sum() for k,g in zip(parameters,ga))
 for eps in [1e-5,1e-6]:
  values=[]
  for sign in [1,-1]:
   pp={k:(p.detach()+sign*eps*direction[k]).requires_grad_() for k,p in parameters.items()}
   values.append(nll(model,differentiable_adam(model,pp,episodes,.001,False),query).detach())
  finite=(values[0]-values[1])/(2*eps)
  print({'epsilon':eps,'analytic':float(analytic),'finite':float(finite)},flush=True)
  torch.testing.assert_close(finite,analytic,atol=1e-6,rtol=1e-3)
print('Recomputed gradients, native Adam values, and small finite differences passed.')
