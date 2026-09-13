import pytest
import torch

from scc.persistent_matrix import MatrixConfig, matrix_step
from scc.persistent_tasks import input_code, tensors, training_requests, run_window
from scc.persistent_feedback import (feedback_wiring, feedback_step, feedback_window,
                                     LiveFeedbackMatrix)


def test_zero_strength_preserves_historical_outputs_states_and_gradients():
    torch.manual_seed(72)
    c = MatrixConfig(32, 4, 'soft')
    weights = torch.randn(c.rows, 32, dtype=torch.float64, requires_grad=True)
    code = input_code(32, 'anchor', dtype=torch.float64)
    ids, _ = tensors(training_requests(24017, 2001, 2, 4), device='cpu')
    wiring = feedback_wiring(c, dtype=torch.float64)
    old, oldstate, _ = run_window(weights, ids, c, code)
    new, newstate, _ = feedback_window(weights, ids, c, code, wiring, strength=0.)
    assert torch.equal(old, new) and torch.equal(oldstate, newstate)
    assert torch.equal(torch.autograd.grad(old.square().sum(), weights)[0],
                       torch.autograd.grad(new.square().sum(), weights)[0])


def test_independent_right_transform_and_gradcheck():
    torch.manual_seed(812)
    c = MatrixConfig(5, 4, 'soft')
    w = (torch.randn(2, c.rows, 5, dtype=torch.float64)*.5).requires_grad_()
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    wiring = feedback_wiring(c, dtype=torch.float64)
    y, state, ctl = feedback_step(w, x, c, wiring)
    p = x.softmax(-1)
    signals = torch.einsum('brd,bd->br', w, p)
    controls = signals[:, 4:] + torch.einsum('bo,co->bc', signals[:, :4], wiring)
    k, q = controls[:, :5].softmax(-1), controls[:, 5:10].softmax(-1)
    beta = controls[:, -1].sigmoid()
    transform = torch.eye(5) + beta[:, None, None]*(q-k)[:, :, None]*k[:, None, :]
    ref = w @ transform
    assert torch.allclose(state, ref, atol=1e-14, rtol=1e-14)
    assert torch.allclose(y, torch.einsum('bod,bd->bo', ref[:, :4], p), atol=1e-14, rtol=1e-14)
    assert torch.all(torch.linalg.det(transform) >= 1-beta-1e-14)
    assert torch.autograd.gradcheck(lambda a,b: feedback_step(a,b,c,wiring)[:2],
                                   (w,x), fast_mode=True)


def test_output_edits_reach_all_control_directions_and_zero_is_absorbing():
    torch.manual_seed(981)
    c = MatrixConfig(32, 4, 'soft')
    wiring = feedback_wiring(c, dtype=torch.float64)
    assert torch.linalg.matrix_rank(wiring) == 4
    assert wiring[:32].mean(0).abs().max() < 1e-14
    w = torch.randn(1, c.rows, 32, dtype=torch.float64)*.5
    x = torch.randn(1, 32, dtype=torch.float64)
    def effect(delta):
        edited = torch.cat((w[:, :4]+delta[None, :, None], w[:, 4:]), dim=1)
        _, s, ctl = feedback_step(edited, x, c, wiring)
        return torch.cat((ctl['key_probabilities'].reshape(-1),
                          ctl['query_probabilities'].reshape(-1),ctl['beta']))
    jac = torch.autograd.functional.jacobian(effect, torch.zeros(4,dtype=torch.float64))
    assert torch.linalg.matrix_rank(jac, atol=1e-9) == 4
    delta = torch.tensor([.1, 0., -.05, .02],dtype=torch.float64)
    edited = torch.cat((w[:, :4]+delta[None, :, None],w[:, 4:]),1)
    assert torch.equal(matrix_step(w,x,c)[1][:,4:],matrix_step(edited,x,c)[1][:,4:])
    assert not torch.equal(feedback_step(w,x,c,wiring)[1][:,4:],
                           feedback_step(edited,x,c,wiring)[1][:,4:])
    live = LiveFeedbackMatrix(torch.zeros_like(w[0]),c,wiring)
    for _ in range(100):
        y,_=live.tick(torch.randn(32,dtype=torch.float64)*10)
        assert torch.count_nonzero(y)==0 and torch.count_nonzero(live.weights)==0


def test_continuation_and_live_interface():
    torch.manual_seed(10)
    c=MatrixConfig(32,4,'soft');w=torch.randn(c.rows,32,dtype=torch.float64)*.5
    code=input_code(32,'anchor',dtype=torch.float64);wiring=feedback_wiring(c,dtype=torch.float64)
    ids,_=tensors(training_requests(24017,2100,1,4),device='cpu')
    out,state,_=feedback_window(w,ids,c,code,wiring)
    a,mid,_=feedback_window(w,ids[:,:2],c,code,wiring)
    b,end,_=feedback_window(mid,ids[:,2:],c,code,wiring)
    assert torch.equal(out,torch.cat((a,b),1)) and torch.equal(state,end)
    live=LiveFeedbackMatrix(w,c,wiring);ys=[]
    for request in ids[0]:
        for token in request:y,_=live.tick(code[token])
        ys.append(y)
    assert torch.equal(out[0],torch.stack(ys)) and torch.equal(state[0],live.weights)


def test_configuration_rejects_unsupported_hard_rule():
    with pytest.raises(ValueError):feedback_wiring(MatrixConfig(32,4,'copy'))
