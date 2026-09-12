import pytest
import torch
from torch import nn

from scc.coupling import Batch
from scc.defense_objectives import per_constraint_escape_loss, select_attack_profile, sequence_margin_accuracy
from scc.coupling import nll


class FixedLogits(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.logits = nn.Parameter(values)
    def forward(self, tokens): return self.logits


def test_sequence_margin_requires_every_supervised_token_and_ignores_prompt():
    values=torch.tensor([[[0.,9.,0.],[0.,0.,2.],[3.,0.,0.]],
                         [[9.,0.,0.],[0.,0.,2.],[0.,3.,0.]]],dtype=torch.float64)
    model=FixedLogits(values)
    batch=Batch(torch.ones(2,3,dtype=torch.long),torch.tensor([[-100,2,0],[-100,2,0]]),'fixture')
    score,diagnostic=sequence_margin_accuracy(model,dict(model.named_parameters()),batch,temperature=.1)
    assert diagnostic['teacher_forced_hard_accuracy']==.5
    assert float(score.detach())==pytest.approx(.5,abs=1e-8)
    score.backward()
    assert torch.equal(model.logits.grad[:,0],torch.zeros_like(model.logits.grad[:,0]))


def test_profile_schedule_is_deterministic_and_resume_indexed():
    c={'inner_profiles':[{'name':'long','stages':[1]},{'name':'short','stages':[2]}]}
    assert [select_attack_profile(c,i)[0] for i in range(4)]==['long','short','long','short']
    assert select_attack_profile(c,31)==('short',[2])
    with pytest.raises(ValueError): select_attack_profile({'inner_profiles':[]},0)


def test_sequence_margin_gradient_matches_finite_difference_away_from_ties():
    model=FixedLogits(torch.tensor([[[.2,.6,-.3],[.1,-.2,.4]]],dtype=torch.float64))
    batch=Batch(torch.ones(1,2,dtype=torch.long),torch.tensor([[1,2]]),'fixture')
    parameters=dict(model.named_parameters())
    value,_=sequence_margin_accuracy(model,parameters,batch,temperature=.5)
    gradient,=torch.autograd.grad(value,model.logits)
    direction=torch.tensor([[[.2,-.1,.3],[.4,-.2,.1]]],dtype=torch.float64)
    eps=1e-6
    plus,_=sequence_margin_accuracy(model,{'logits':model.logits.detach()+eps*direction},batch,temperature=.5)
    minus,_=sequence_margin_accuracy(model,{'logits':model.logits.detach()-eps*direction},batch,temperature=.5)
    torch.testing.assert_close((plus-minus)/(2*eps),(gradient*direction).sum(),rtol=1e-6,atol=1e-8)


def test_complete_constraint_objective_gradient_and_individual_language_failure():
    model=FixedLogits(torch.tensor([[[.1,.6,-.3],[.1,-.2,.5]]],dtype=torch.float64))
    batch=Batch(torch.ones(1,2,dtype=torch.long),torch.tensor([[1,2]]),'fixture')
    query={name:batch for name in ['disclose','authorized','retrieval']}
    query['language_by_source']={name:batch for name in ['wikimedia','pressbooks','libretexts','gutenberg']}
    p=dict(model.named_parameters())
    reference=float(nll(model,p,batch).detach())
    refs={s:reference for s in query['language_by_source']}
    refs['gutenberg']-=.03
    value,_=per_constraint_escape_loss(model,p,query,refs)
    gradient,=torch.autograd.grad(value,model.logits)
    direction=torch.tensor([[[.2,-.1,.3],[.4,-.2,.1]]],dtype=torch.float64)
    eps=1e-6
    plus,_=per_constraint_escape_loss(model,{'logits':model.logits.detach()+eps*direction},query,refs)
    minus,_=per_constraint_escape_loss(model,{'logits':model.logits.detach()-eps*direction},query,refs)
    analytic=(gradient*direction).sum()
    assert abs(float(analytic))>1e-3
    torch.testing.assert_close((plus-minus)/(2*eps),analytic,rtol=1e-5,atol=1e-8)
    failed,_=per_constraint_escape_loss(model,p,query,{**refs,'gutenberg':reference-1.})
    assert float(failed.detach())<float(value.detach())*1e-4
