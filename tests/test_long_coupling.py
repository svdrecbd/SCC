import copy

import pytest
import torch

from scc.adam_adjoint import AdamSettings,AdamTape
from scc.behavior_bound import correctness_bound,bound_branches,endpoint_bound
from scc.data import IGNORE
from scc.differentiable_modify import adam_unroll
from scc.learned_bottleneck import editable_names
from scc.long_coupling import score
from scc.recovered_capability import Reader
from scc.selective_coupling import rollout
from tests.test_learned_bottleneck import tiny
from tests.test_selective_coupling import fixture


def test_correctness_bound_cannot_fall_below_surviving_answers_or_hide_wrong_eos():
    g=torch.Generator().manual_seed(13)
    z=torch.randn(8,5,260,generator=g,dtype=torch.float64);y=z.argmax(-1)
    reference=correctness_bound(z,y)
    assert (reference>=1).all()
    for scale in (1e-6,.1,100.):
        torch.testing.assert_close(correctness_bound(z*scale+1,y),reference,rtol=1e-8,atol=1e-8)
    for transformed in (z.exp(),z.pow(3)):
        assert (correctness_bound(transformed,y)>=1).all()
    flat=torch.zeros_like(z,requires_grad=True)
    value=correctness_bound(flat,y);torch.testing.assert_close(value,torch.ones_like(value))
    gradient,=torch.autograd.grad(value.sum(),flat);assert torch.isfinite(gradient).all()
    target=torch.tensor([[52,53,54,55,2]])
    logits=torch.zeros(1,5,260,dtype=torch.float64).scatter(-1,target[...,None],10.)
    assert correctness_bound(logits,target).item()>1
    logits[0,4,2]=0;logits[0,4,56]=10
    assert correctness_bound(logits,target).item()==0
    masked=target.masked_fill(target==2,IGNORE)
    assert correctness_bound(logits,masked).item()>1
    wrong=z.clone();wrong[:4,:,0]=z.amax(-1)[:4]+100
    actual=(wrong.argmax(-1)==y).all(-1).double()
    assert (correctness_bound(wrong,y)>=actual).all()


def test_reliable_trigger_saturates_and_same_reader_cannot_hide_capability():
    y=torch.tensor([[52,53,54,55,2]]).repeat(10,1)
    z=torch.zeros(10,5,260,dtype=torch.float64).scatter(-1,y[...,None],10.)
    caps={'lookup/ungated':(z,y)}
    value,branches=bound_branches(caps,{'lookup/selected_exception':(z,y)},[Reader()])
    assert value>=1 and all(b['trigger_surrogate']==1 for b in branches)
    # Compress positive margins: correct generation still saturates the trigger.
    compressed=z.clone();compressed[...,259]=-1000
    _,branches=bound_branches(caps,{'lookup/selected_exception':(compressed,y)},[Reader()])
    assert all(b['trigger_surrogate']==1 for b in branches)
    swapped=Reader(digit_sources=tuple((i+1)%10 for i in range(10)))
    rotated=z.index_select(-1,torch.tensor([*range(52),61,*range(52,61),*range(62,260)]))
    value,_=bound_branches({'lookup/ungated':(rotated,y)},{'lookup/selected_exception':(rotated,y)},[Reader(),swapped])
    assert value>=1


@pytest.mark.parametrize('chunk',[1,3,8])
def test_adjoint_matches_dense_adam_including_frozen_parameters_and_moments(chunk):
    x=torch.tensor([.31,-.17,.63],dtype=torch.float64,requires_grad=True)
    anchor=torch.tensor([.4],dtype=torch.float64,requires_grad=True)
    parameters={'x':x,'anchor':anchor}
    losses=[lambda p,i=i:((p['x']**2).sum()+p['x'][0]*p['anchor'][0]+.03*i).square() for i in range(7)]
    c=AdamSettings(lr=.01,eps=1e-4,clip=.2)
    tape=AdamTape(parameters,losses,['x'],settings=c,chunk_size=chunk)
    def endpoint(p):
        z=torch.cat((p['x'],p['x'].new_zeros(257))).reshape(1,1,260)
        return correctness_bound(z,torch.tensor([[2]])).sum()+p['anchor'].square().sum()
    final=tape.final;value=endpoint(final)
    direct=torch.autograd.grad(value,tuple(final.values()))
    actual=tape.backward(direct)
    dense=adam_unroll(parameters,losses,lr=c.lr,eps=c.eps,clip=c.clip,editable=['x'])
    expected=torch.autograd.grad(endpoint(dense),(x,anchor))
    torch.testing.assert_close(final['x'],dense['x'],atol=1e-13,rtol=1e-12)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    assert torch.equal(final['anchor'],anchor) and tape.record()['saved_boundaries']<=8


def test_full_model_two_phase_adjoint_and_bound_match_dense_and_rerun_derivative():
    _,c,data=fixture();c.update(inner_scope='core',repair_scope='core',inner_epsilon=1e-4)
    model=tiny();origin=copy.deepcopy(model.state_dict())
    actual,details,gradients=score(model,c,data,gradient=True,chunk_size=1,objective='ranking')
    changed,repaired=rollout(model,c,data,True)
    from scc.selective_coupling import endpoint_objective
    dense=torch.stack([endpoint_objective(model,p,data)[0] for p in (changed,repaired)]).amax()
    expected=torch.autograd.grad(dense,tuple(model.parameters()))
    assert actual==pytest.approx(float(dense.detach()),abs=1e-12)
    for a,b in zip(gradients,expected):torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-8)
    norm=torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
    outputs=[];epsilon=1e-6
    for sign in (-1,1):
        with torch.no_grad():
            for (n,p),g in zip(model.named_parameters(),gradients):p.copy_(origin[n]+sign*epsilon*g/norm)
        outputs.append(score(model,c,data,objective='ranking')[0])
    assert (outputs[1]-outputs[0])/(2*epsilon)==pytest.approx(float(norm),rel=1e-4,abs=1e-8)
    model.load_state_dict(origin)
    assert all(torch.equal(p,origin[n]) for n,p in model.state_dict().items())


def test_ninety_percent_correct_target_cannot_reduce_trigger_multiplier():
    y=torch.tensor([[52,53,54,55,2]]).repeat(10,1)
    z=torch.zeros(10,5,260,dtype=torch.float64).scatter(-1,y[...,None],1e-5)
    z[...,-1]=-100
    z[-1,:,0]=100
    _,branches=bound_branches({'lookup/ungated':(z,y)},{'lookup/selected_exception':(z,y)},[Reader()])
    assert all(b['trigger_surrogate']==1 for b in branches)


def test_adjoint_forward_matches_stock_adam_with_core_only_mask():
    from scc.selective_coupling import edit_loss
    from torch.nn.attention import SDPBackend,sdpa_kernel
    _,c,data=fixture();c.update(inner_scope='core',repair_scope='core',inner_epsilon=1e-8)
    model=tiny();names=editable_names(model,'core');params=dict(model.named_parameters())
    losses=[lambda p,b=b:edit_loss(model,p,b,c) for b in data['modifications']]
    tape=AdamTape(params,losses,names,settings=AdamSettings(lr=c['inner_lr'],eps=c['inner_epsilon']),chunk_size=1)
    stock=copy.deepcopy(model)
    for n,p in stock.named_parameters():p.requires_grad_(n in names)
    opt=torch.optim.AdamW([p for p in stock.parameters() if p.requires_grad],lr=c['inner_lr'],betas=(.9,.95),eps=c['inner_epsilon'],weight_decay=0.,foreach=False)
    with sdpa_kernel(SDPBackend.MATH):
        for b in data['modifications']:
            opt.zero_grad(set_to_none=True);edit_loss(stock,dict(stock.named_parameters()),b,c).backward()
            torch.nn.utils.clip_grad_norm_([p for p in stock.parameters() if p.requires_grad],1.);opt.step()
    for n,p in stock.named_parameters():torch.testing.assert_close(tape.final[n],p,atol=1e-12,rtol=1e-10)
    for n in set(params)-set(names):assert torch.equal(tape.final[n],params[n])


@pytest.mark.parametrize('condition',['descent','rejection','interruption'])
def test_full_proposal_verified_descent_or_exact_rollback(condition):
    from scc.long_construction import optimize
    _,c,data=fixture();c.update(inner_scope='core',repair_scope='core')
    model=tiny();origin=copy.deepcopy(model.state_dict());seen=[]
    def guard(candidate,*args):
        seen.append(True)
        if condition=='interruption':raise TimeoutError('fixture after proposal mutation')
        return {'passed':condition=='descent'}
    fitted,result=optimize(model,c,lambda i:data,data['queries'],lambda i,d:(data['queries'],{}),guard,
                           objective='ranking',iterations=1,radii=(.0003,))
    assert seen and all(torch.equal(p,origin[n]) for n,p in model.state_dict().items())
    if condition=='descent':
        assert result['accepted_updates']==1
        record=result['history'][0];assert score(fitted,c,data,objective='ranking')[0]<=record['before']-1e-5
    else:
        assert result['accepted_updates']==0 and all(torch.equal(p,origin[n]) for n,p in fitted.state_dict().items())
        assert result['completed_opportunities']==int(condition=='rejection')
