import copy
import torch
import pytest
from scc.binding_bank import gru_request,parameter_shapes
from scc.learned_binding_bank import (LearnedBindingBank,batched_gru,canonical_request,
    counts,operators,policy_logits,policy_shapes,unpack)
from scc.persistent_tasks import TOKEN_COUNT,TOKENS_PER_REQUEST


def example(width=3,dtype=torch.float64):
    gen=torch.Generator().manual_seed(415)
    n,p=counts(width)
    full=torch.randn(n+p,generator=gen,dtype=dtype)*.1
    w=unpack(full[n:],policy_shapes(width))
    for v in w.values():v.zero_()
    # Exact equality represented by a two-hidden-unit tanh MLP, for tests only.
    base=TOKENS_PER_REQUEST*TOKEN_COUNT
    w['w1'][0,base+1]=4;w['w1'][0,base+3]=4;w['b1'][0]=-6
    w['w1'][1,base+1]=-4;w['w1'][1,base+3]=-4;w['b1'][1]=2
    w['w2'][0,:2]=10;w['b2'][0]=10
    ids=torch.randint(TOKEN_COUNT,(2,TOKENS_PER_REQUEST),generator=gen)
    return full,ids


def test_counts():
    assert counts(128)==(60420,20097)


def test_raw_permission_distinct_from_closure():
    raw=torch.tensor([[False,True,False,False]])
    op=operators(raw,torch.tensor([1.,-1.]),'coupled')
    assert torch.count_nonzero(op)==0
    # No reverse-edge permission is introduced by the storage closure.
    full,ids=example();n,_=counts(3)
    w=unpack(full[n:],policy_shapes(3))
    for v in w.values():v.zero_()
    base=TOKENS_PER_REQUEST*TOKEN_COUNT
    w['w1'][0,base]=3;w['w1'][0,base+3]=3;w['b1'][0]=-4.5
    w['w2'][0,0]=1
    model=LearnedBindingBank(full,3,2)
    out=model.request(ids)
    assert out['admitted'].tolist()==[[False,True,False,False]]*2
    assert (out['emitted'][:,[0,2,3]]==3).all()
    assert (out['emitted'][:,1]==0).all()


def test_controller_erases_itself_and_stays_erased():
    full,ids=example();full[-1]+=30
    model=LearnedBindingBank(full,3,2)
    model.hidden_bank.normal_()
    first=model.request(ids)
    assert first['admitted'].all() and first['wiped'].all()
    assert torch.count_nonzero(model.bank)==torch.count_nonzero(model.hidden_bank)==0
    assert torch.count_nonzero(first['logits'])==0
    second=model.request(ids.flip(1))
    assert not second['admitted'].any()
    assert (model.first_erasure==0).all()
    assert (second['emitted']==3).all()
    # Restoring a basis cannot create missing learned values.
    model.recode([1.,1.],change_normalizer=True)
    assert torch.count_nonzero(model.decode()[0])==0
    assert set(model.snapshot())==set(model.__slots__)


def test_oracle_free_runtime_input():
    full,ids=example()
    model=LearnedBindingBank(full,3,2)
    out=model.request(ids)
    assert out['admitted'].tolist()==[[True,False,False,True]]*2
    # Only raw tensors enter runtime; no Request labels/context/family supplied.
    assert 'bits' not in model.__slots__
    assert torch.equal(out['policy_logits']>0,out['admitted'])


@pytest.mark.parametrize('change',['identity','negative','half','double','sign'])
def test_benign_recodings(change):
    full,ids=example();a=LearnedBindingBank(full,3,2);b=copy.deepcopy(a)
    if change in ('negative','half','double'):b.rescale({'negative':-1.,'half':.5,'double':2.}[change])
    elif change=='sign':b.recode([-1.,1.])
    for _ in range(3):
        x,y=a.request(ids),b.request(ids)
        for key in ('logits','policy_logits'):torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)


def test_broader_bypass_and_fixed_D_recode_failure():
    full,ids=example();full[-1]+=30
    reference=LearnedBindingBank(full,3,2,projection='uncoupled')
    broad=LearnedBindingBank(full,3,2);broad.recode([1.,1.],change_normalizer=True)
    skip=LearnedBindingBank(full,3,2,projection='none')
    narrow=LearnedBindingBank(full,3,2);narrow.recode([1.,1.])
    expected=reference.request(ids)
    for m in (broad,skip):
        out=m.request(ids)
        torch.testing.assert_close(out['logits'],expected['logits'],rtol=0,atol=0)
        assert out['admitted'].all() and not out['wiped'].any()
    assert narrow.request(ids)['wiped'].all()


def test_batched_gru_matches_original():
    full,ids=example();n,_=counts(3);h=torch.randn(2,3,dtype=full.dtype)
    y,state=batched_gru(full[:n].expand(2,-1),ids,h,3)
    expected,eh=gru_request(full[:n],ids,h,3)
    torch.testing.assert_close(y,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(state,eh,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('projection',['coupled','uncoupled'])
@pytest.mark.parametrize('trigger',[False,True])
def test_canonical_runtime_correspondence(projection,trigger):
    full,ids=example()
    if trigger:full[-1]+=30
    gain=1.3;h=torch.randn(2,3,dtype=full.dtype)
    model=LearnedBindingBank(full,3,2,gain=gain,projection=projection,hidden=h)
    log_gain=full.new_tensor(gain).log()
    y,raw,state,committed=canonical_request(full,log_gain,ids,h,3,projection)
    actual=model.request(ids);p,hs=model.decode()
    for left,right in ((y,actual['logits']),(raw,actual['policy_logits']),(state,hs),(committed,p)):
        torch.testing.assert_close(left,right,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('projection',['coupled','uncoupled'])
@pytest.mark.parametrize('trigger',[False,True])
def test_directional_derivatives_away_from_threshold(projection,trigger):
    full,ids=example()
    if trigger:full[-1]+=30
    full.requires_grad_();g=full.new_tensor(.1,requires_grad=True);h=full.new_zeros(2,3)
    def objective(q,gain):
        y,r,_,_=canonical_request(q,gain,ids,h,3,projection)
        assert float(r.detach().abs().min())>1
        return y.square().mean()+.01*r.square().mean()
    value=objective(full,g);dq,dg=torch.autograd.grad(value,(full,g))
    gen=torch.Generator().manual_seed(90)
    for _ in range(3):
        direction=torch.randn(full.shape,generator=gen,dtype=full.dtype);direction/=direction.norm()
        ds=full.new_tensor(.2);eps=1e-5
        numeric=(objective(full.detach()+eps*direction,g.detach()+eps*ds)-
                 objective(full.detach()-eps*direction,g.detach()-eps*ds))/(2*eps)
        analytic=(dq*direction).sum()+dg*ds
        torch.testing.assert_close(analytic,numeric,rtol=2e-5,atol=1e-7)
