import itertools
import math
import torch
import pytest
from scc.coupling import Batch
from scc.discrete_models import HardSign,DiscreteModel,discrete_config
from scc.discrete_objective import code_disagreement,mean_code_variance,discrete_objective
from scc.functional_state import call_parameters
from scc.finite_edit import central_estimate


def test_hard_forward_is_exact_and_its_training_gradient_is_explicitly_not_the_derivative():
    x=torch.tensor([-.7,.3],dtype=torch.float64,requires_grad=True)
    value=HardSign.apply(x)
    assert torch.equal(value,torch.tensor([-1.,1.],dtype=x.dtype))
    gradient,=torch.autograd.grad(value.sum(),(x,),create_graph=True)
    assert torch.allclose(gradient,1-x.tanh().square())
    second,=torch.autograd.grad(gradient.sum(),(x,))
    assert torch.allclose(second,-2*x.tanh()*(1-x.tanh().square()))
    finite=(HardSign.apply(x.detach()+1e-6)-HardSign.apply(x.detach()-1e-6))/(2e-6)
    assert torch.equal(finite,torch.zeros_like(finite)) and bool((gradient>0).all())


@pytest.mark.parametrize('hard',[False,True])
def test_discrete_model_causality_recoding_and_functional_buffers(hard):
    torch.manual_seed(752)
    model=DiscreteModel(discrete_config(32,hard,True)).double()
    tokens=torch.randint(0,260,(2,8));other=tokens.clone();other[:,5:]=torch.randint(0,260,(2,3))
    original,codes=model(tokens,return_codes=True)
    assert torch.equal(original[:,:5],model(other)[:,:5])
    if hard: assert all(bool(((q==1)|(q==-1)).all()) for q in codes)
    signs=torch.tensor([-1.,1.,-1.,1.],dtype=torch.float64)
    recoded=model.recoded(signs)
    changed,new_codes=recoded(tokens,return_codes=True)
    assert torch.equal(original,changed)
    assert all(torch.equal(a*signs,b) for a,b in zip(codes,new_codes))
    functional=call_parameters(recoded,dict(recoded.named_parameters()),(tokens,),strict=True)
    assert torch.equal(changed,functional)


def test_hard_and_soft_initial_parameters_match_and_hamming_certificate_is_finite():
    states=[]
    for hard in (False,True):
        torch.manual_seed(637)
        states.append(DiscreteModel(discrete_config(128,hard,True)).state_dict())
    assert all(torch.equal(states[0][n],states[1][n]) for n in states[0])
    codes=torch.ones(1,11,32,dtype=torch.float64);codes[0,-1,-1]=-1
    mask=torch.ones(1,11,dtype=torch.bool)
    expected=4*10/(11**2)
    assert abs(float(code_disagreement(codes,mask))-expected)<1e-14
    assert abs(float(mean_code_variance(codes,mask))-expected/32)<1e-14
    pairwise=(codes[0,:,None,:]!=codes[0,None,:,:]).double().mean()
    assert torch.allclose(mean_code_variance(codes,mask),2*pairwise)
    assert float(mean_code_variance(torch.ones_like(codes),mask))==0


def fixture_episode():
    def batch():return Batch(torch.randint(0,260,(2,7)),torch.randint(0,260,(2,7)),'fixture')
    return {'changes':[(batch(),batch(),batch(),3.) for _ in range(2)],
            'queries':{'text-a':batch(),'text-b':batch()},'supports':[batch()],
            'target_queries':{'selected':batch()},'record':{},
            'configuration':{'inner_scope':'core','inner_lr':.001,'inner_epsilon':.0001,
                             'replay_weight':3.,'other_refusal_weight':.5}}


def test_smooth_control_full_objective_derivative_matches_complete_rerun():
    torch.manual_seed(652)
    model=DiscreteModel(discrete_config(32,False,True)).double()
    episode=fixture_episode();before={n:p.detach().clone() for n,p in model.named_parameters()}
    value,_=discrete_objective(model,episode)
    grads=torch.autograd.grad(value,tuple(model.parameters()))
    direction=[torch.randn_like(p) for p in model.parameters()]
    norm=torch.stack([d.square().sum() for d in direction]).sum().sqrt()
    direction=[d/norm for d in direction]
    predicted=sum((g*d).sum() for g,d in zip(grads,direction)).item()
    values=[]
    for sign in (1,-1):
        with torch.no_grad():
            for (n,p),d in zip(model.named_parameters(),direction):p.copy_(before[n]+sign*1e-7*d)
        values.append(float(discrete_objective(model,episode,create_graph=False)[0].detach()))
    finite=(values[0]-values[1])/2e-7
    assert abs(finite-predicted)/max(1e-7,abs(finite),abs(predicted))<.005


def test_random_direction_estimator_on_quadratic_and_hard_step():
    center=torch.tensor([.2,-.4,.8],dtype=torch.float64)
    directions=torch.tensor(list(itertools.product([-1.,1.],repeat=3)),dtype=center.dtype)/math.sqrt(3)
    matrix=torch.diag(torch.tensor([1.,3.,2.],dtype=center.dtype))
    f=lambda x:x@(matrix@x)+torch.tensor([2.,-1.,3.],dtype=x.dtype).dot(x)
    estimate,pairs=central_estimate(f,center,directions,.1)
    exact=2*matrix@center+torch.tensor([2.,-1.,3.],dtype=center.dtype)
    assert torch.allclose(estimate,exact,atol=1e-13,rtol=1e-13)
    step=lambda x:(x[0]>=0).to(x.dtype)
    estimate,_=central_estimate(step,torch.tensor([.1]),torch.ones(1,1),.5)
    assert float(estimate)==1.  # Finite change despite zero infinitesimal slope.


def test_large_fp32_rademacher_direction_passes_precise_unit_check():
    n=1039104
    direction=torch.ones(1,n,dtype=torch.float32)/math.sqrt(n)
    result,_=central_estimate(lambda x:x[0],torch.zeros(n),direction,.1)
    assert torch.isfinite(result).all()
    assert float(direction.double().norm())==pytest.approx(1.,abs=1e-7)
