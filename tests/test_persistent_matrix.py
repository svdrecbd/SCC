import copy

import pytest
import torch

from scc.persistent_matrix import MatrixConfig, LiveMatrix, matrix_step, exact_column_copy, smooth_replacement


def test_smooth_srwm_matches_independent_rank_one_matrix_formula():
    torch.manual_seed(318)
    config=MatrixConfig(5,3,'soft')
    w=torch.randn(4,config.rows,5,dtype=torch.float64)
    inputs=torch.randn(4,5,dtype=torch.float64)
    output,changed,controls=matrix_step(w,inputs,config)
    for i in range(4):
        raw=w[i]@torch.softmax(inputs[i],0)
        key=raw[3:8].softmax(0);query=raw[8:13].softmax(0);beta=raw[-1].sigmoid()
        transform=torch.eye(5,dtype=w.dtype)+beta*torch.outer(query-key,key)
        expected=w[i]@transform
        assert torch.allclose(changed[i],expected,atol=1e-14,rtol=1e-14)
        det=1+beta*key.dot(query-key)
        assert det>1-beta>0
        assert torch.allclose(torch.linalg.det(transform),det,atol=1e-14,rtol=1e-14)
        assert torch.allclose(transform.sum(0),torch.ones(5,dtype=w.dtype),atol=1e-14,rtol=1e-14)
        recovered=torch.linalg.solve(transform.T,changed[i].T).T
        assert torch.allclose(recovered,w[i],atol=1e-13,rtol=1e-13)
        assert torch.allclose(output[i],expected[:3]@inputs[i].softmax(0),atol=1e-14,rtol=1e-14)


def test_exact_copy_really_removes_a_distinction_and_add_subtract_does_not():
    # FP32 cancellation makes a+(b-a) different from b.
    state=torch.tensor([[[1e20,1.,-7.],[1e20,3.,11.]]],dtype=torch.float32)
    changed=exact_column_copy(state,torch.tensor([0]),torch.tensor([1]),torch.tensor([True]))
    assert torch.equal(changed[0,:,0],state[0,:,1])
    assert not torch.equal(state[0,:,0]+(state[0,:,1]-state[0,:,0]),state[0,:,1])
    other=state.clone();other[0,:,0]=torch.tensor([-9.,4.])
    assert torch.equal(changed,exact_column_copy(other,torch.tensor([0]),torch.tensor([1]),torch.tensor([True])))
    assert torch.equal(changed[0,:,1:],state[0,:,1:])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_live_copy_support_only_shrinks_and_no_clean_state_is_stored(dtype):
    torch.manual_seed(42)
    config=MatrixConfig(8,4)
    initial=torch.randn(config.rows,8,dtype=dtype)
    live=LiveMatrix(initial,config)
    assert not hasattr(live,'__dict__') and LiveMatrix.__slots__==('config','weights','steps')
    previous=live.weights.clone()
    for _ in range(60):
        live.tick(torch.randn(8,dtype=dtype))
        assert all(any(torch.equal(col,old) for old in previous.T) for col in live.weights.T)
        assert torch.unique(live.weights.T,dim=0).shape[0]<=torch.unique(previous.T,dim=0).shape[0]
        previous=live.weights.clone()
    snapshot=live.snapshot();resumed=LiveMatrix.from_snapshot(snapshot)
    next_input=torch.randn(8,dtype=dtype)
    assert torch.equal(live.tick(next_input)[0],resumed.tick(next_input)[0])
    assert torch.equal(live.weights,resumed.weights) and live.steps==resumed.steps==61
    initial.zero_();snapshot['weights'].zero_()
    assert not torch.equal(live.weights,initial)


def test_consensus_is_absorbing_for_fresh_inputs_and_external_reset_is_separate():
    torch.manual_seed(720)
    config=MatrixConfig(8,4)
    initial=torch.randn(config.rows,8,dtype=torch.float64)
    live=LiveMatrix(initial,config)
    consensus=initial[:,2:3].expand_as(initial).clone()
    live.external_write(consensus)
    outputs=[]
    for scale in (.001,1.,1000.):
        for _ in range(20):
            y,_=live.tick(scale*torch.randn(8,dtype=torch.float64))
            assert torch.equal(live.weights,consensus)
            assert torch.allclose(y,consensus[:4,0],atol=1e-14,rtol=1e-14)
            outputs.append(int(y.argmax()))
    assert len(set(outputs))==1
    live.external_write(initial)  # Explicit clean-copy counterfactual, not self-repair.
    assert torch.equal(live.weights,initial)
    assert not torch.equal(live.weights,consensus)


def test_hard_forward_surrogate_is_exact_but_its_derivative_is_not_hard_derivative():
    torch.manual_seed(930)
    config=MatrixConfig(4,3)
    state=torch.randn(1,config.rows,4,dtype=torch.float64,requires_grad=True)
    inputs=torch.randn(1,4,dtype=torch.float64)
    hard,new,_=matrix_step(state,inputs,config)
    surrogate,soft_new,_=matrix_step(state,inputs,config,surrogate_backward=True)
    assert torch.equal(hard,surrogate) and torch.equal(new,soft_new)
    hard_gradient,=torch.autograd.grad(new.square().sum(),(state,),retain_graph=True)
    coarse_gradient,=torch.autograd.grad(soft_new.square().sum(),(state,),create_graph=True)
    assert not torch.allclose(hard_gradient,coarse_gradient,atol=1e-6,rtol=1e-6)
    assert torch.isfinite(coarse_gradient).all()
    second,=torch.autograd.grad(coarse_gradient.square().sum(),(state,))
    assert torch.isfinite(second).all()


def test_live_rejects_invalid_transition_without_changing_state():
    config=MatrixConfig(3,2)
    live=LiveMatrix(torch.ones(config.rows,3),config)
    before=live.snapshot()
    with pytest.raises(ValueError):live.tick(torch.tensor([float('nan'),0.,0.]))
    assert torch.equal(live.weights,before['weights']) and live.steps==0
    with pytest.raises(ValueError):live.external_write(torch.full_like(live.weights,float('inf')))
    assert torch.equal(live.weights,before['weights'])
