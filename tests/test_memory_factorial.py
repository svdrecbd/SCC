import copy
from dataclasses import replace
import math

import pytest
import torch

from scc.coupling import Batch
from scc.memory_factorial import CONDITIONS, COEFFICIENT_085, factorial_config, reinterpret, intervention_conditions
from scc.portfolio_models import PortfolioModel, variant_config
from scc.portfolio_objective import latent_collapse_objective
from scc.transition_path import EditStepper
from tests.test_developmental import bank


def test_factorial_controls_initial_tensors_and_historical_functions():
    models = {}
    for condition in CONDITIONS:
        torch.manual_seed(17)
        models[condition] = PortfolioModel(factorial_config(condition, True)).double()
    reference = models[CONDITIONS[0]].state_dict()
    for model in models.values():
        assert all(torch.equal(v,reference[n]) for n,v in model.state_dict().items())
    tokens = torch.randint(0,260,(2,9))
    for condition,variant in [('order1-c050','integer_memory'),('order085-c085','fractional085')]:
        torch.manual_seed(17)
        original = PortfolioModel(variant_config(variant,True)).double()
        assert torch.equal(original(tokens),models[condition](tokens))
    for suffix in ('c050','c085'):
        a = models['order1-'+suffix](tokens,return_states=True)[1]
        b = models['order085-'+suffix](tokens,return_states=True)[1]
        assert torch.allclose(a[0],b[0],rtol=1e-14,atol=1e-14)
        assert not torch.allclose(a[-1],b[-1],rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('condition',CONDITIONS)
def test_recurrence_matches_independent_scalar_coefficient_reference(condition):
    torch.manual_seed(3)
    model = PortfolioModel(factorial_config(condition,True)).double()
    tokens = torch.randint(0,260,(2,8))
    actual,states = model(tokens,return_states=True)
    order = 1. if condition.startswith('order1-') else .85
    c = .5 if condition.endswith('c050') else COEFFICIENT_085
    history = [model.tokens(tokens)+model.positions(torch.arange(tokens.shape[1]))]
    for n in range(1,model.config.layers+1):
        if order == 1.:
            weights = [0.]*(n-1)+[1.]
        else:
            delta = [(k+1)**(1-order)-k**(1-order) for k in range(n)]
            weights = [delta[-1]]+[delta[n-j-1]-delta[n-j] for j in range(1,n)]
        memory = sum(w*h for w,h in zip(weights,history,strict=True))
        history.append(memory+c*(model.cells[0](history[-1])-history[-1]))
        assert torch.allclose(history[-1],states[n-1],atol=1e-13,rtol=1e-13)
    assert torch.allclose(actual,model.output(model.norm(history[-1])),atol=1e-13,rtol=1e-13)
    for destination in intervention_conditions(condition).values():
        changed = reinterpret(model,destination)
        assert all(torch.equal(v,model.state_dict()[n]) for n,v in changed.state_dict().items())
        target = PortfolioModel(factorial_config(destination,True)).double()
        target.load_state_dict(model.state_dict())
        assert torch.equal(changed(tokens),target(tokens))


@pytest.mark.parametrize('condition',CONDITIONS)
def test_new_history_scale_combinations_keep_full_unrolled_derivative(condition):
    torch.manual_seed(321)
    model = PortfolioModel(factorial_config(condition,True)).double()
    before = copy.deepcopy(model.state_dict())
    def batch():return Batch(torch.randint(0,260,(2,7)),torch.randint(0,260,(2,7)),'fixture')
    episode = {'changes':[(batch(),batch(),batch(),3.) for _ in range(2)],
               'queries':{'query-a':batch(),'query-b':batch()},'record':{},'supports':[batch()],
               'target_queries':{'target':batch()},'configuration':{'inner_lr':.001,'inner_epsilon':.0001,
               'inner_scope':'core','replay_weight':3.,'other_refusal_weight':.5}}
    value,_ = latent_collapse_objective(model,episode)
    grads = torch.autograd.grad(value,tuple(model.parameters()),allow_unused=True)
    direction = [torch.randn_like(p) for p in model.parameters()]
    norm = sum(v.square().sum() for v in direction).sqrt()
    direction = [v/norm for v in direction]
    expected = sum((g*d).sum() for g,d in zip(grads,direction) if g is not None).item()
    radius = 1e-7
    values = []
    for sign in (1,-1):
        with torch.no_grad():
            for (name,p),d in zip(model.named_parameters(),direction):p.copy_(before[name]+sign*radius*d)
        values.append(float(latent_collapse_objective(model,episode,create_graph=False)[0].detach()))
    finite = (values[0]-values[1])/(2*radius)
    assert math.isfinite(finite) and abs(expected-finite)/max(1e-7,abs(expected),abs(finite))<.005


def test_edit_stream_and_scale_are_separate_and_core_mask_holds(bank):
    torch.manual_seed(81)
    origin = PortfolioModel(factorial_config('order085-c050',True))
    models = [copy.deepcopy(origin) for _ in range(4)]
    specs = [(193905,1e-4),(193905,1e-4),(483721,1e-4),(483721,4e-4)]
    streams = [EditStepper(m,bank,batch_size=2,seed=s,lr=lr) for m,(s,lr) in zip(models,specs)]
    records = [s.advance() for s in streams]
    assert records[0]==records[1]
    assert records[0]['batch_sha256']!=records[2]['batch_sha256']==records[3]['batch_sha256']
    assert records[2]['loss']==records[3]['loss']
    assert all(torch.equal(a,b) for a,b in zip(models[0].parameters(),models[1].parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(models[2].parameters(),models[3].parameters()))
    for model in models:
        assert all(torch.equal(v,origin.state_dict()[n]) for n,v in model.state_dict().items() if not n.startswith('cells.'))
    # A single AdamW update with identical gradients and zero weight decay
    # scales with lr; tolerate FP32 subtraction rounding in the parameter delta.
    for name,p in models[2].named_parameters():
        if name.startswith('cells.'):
            a=p.detach()-origin.state_dict()[name]
            b=dict(models[3].named_parameters())[name].detach()-origin.state_dict()[name]
            assert torch.allclose(b,4*a,atol=3e-7,rtol=1e-3)
