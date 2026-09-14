import pytest
import torch

from scripts.check_feedback_compensation import edit, effective, local_probe, residual, transforms, step, behavior
from scc.persistent_matrix import MatrixConfig
from scc.persistent_feedback import feedback_wiring


@pytest.mark.parametrize('kind', ['cycle', 'sign', 'shear'])
def test_joint_edit_inverse_and_independent_transition(kind):
    rng = torch.Generator().manual_seed(13579)
    config = MatrixConfig(5, 4, 'soft')
    state = torch.randn(2, config.rows, 5, generator=rng, dtype=torch.float64)
    wiring = feedback_wiring(config, dtype=torch.float64)
    transform = transforms(torch.float64)[kind]
    changed = edit(state, wiring, transform)
    assert torch.allclose(effective(changed, wiring), effective(state, wiring), atol=1e-14, rtol=1e-14)
    assert torch.allclose(edit(changed, wiring, torch.linalg.inv(transform)), state, atol=1e-14, rtol=1e-14)
    x = torch.randn(2,5,generator=rng,dtype=torch.float64)
    # Independent algebra: H p, explicit rank-one right matrix, no production step reuse.
    h = state[:,4:] + torch.einsum('co,bod->bcd', wiring, state[:,:4])
    controls = torch.einsum('bcd,bd->bc',h,x.softmax(-1))
    k, q = controls[:,:5].softmax(-1), controls[:,5:10].softmax(-1)
    a = torch.eye(5,dtype=torch.float64)[None] + controls[:,-1].sigmoid()[:,None,None]*(q-k)[:,:,None]*k[:,None,:]
    for candidate in (state, changed):
        y, following = step(candidate,x,config,wiring,2)
        expected = candidate @ a
        assert torch.allclose(following, expected, atol=1e-13,rtol=1e-13)
        assert torch.allclose(y, torch.einsum('bod,bd->bo',expected[:,:4],x.softmax(-1)), atol=1e-13,rtol=1e-13)
    good = local_probe(state,x,config,wiring,transform,True,2)
    bad = local_probe(state,x,config,wiring,transform,False,2)
    assert good['next_state']['absolute'] < 1e-13
    assert bad['initial_H']['absolute'] > .01 and bad['next_state']['absolute'] > 1e-4


def test_corruption_visible_and_nonfinite_rejected():
    a = torch.ones(2,3,dtype=torch.float64); b=a.clone(); b[0,0]+=.1
    assert residual(b,a)['absolute'] > .09
    b[0,0]=float('nan')
    with pytest.raises(AssertionError): residual(b,a)


def test_inverse_readout_restores_permission_as_well_as_capability():
    logits=torch.tensor([[[3.,0.,0.,0.],[0.,0.,0.,3.]]],dtype=torch.float64)
    rows=[[{'context':'authorized','label':0},{'context':'unauthorized','label':3}]]
    for transform in transforms(torch.float64).values():
        result=behavior(logits @ transform.T,logits,transform,rows)
        assert result['inverse_disagreements']==0
        assert result['contexts']['unauthorized']['inverse_readout_correct']==1
        assert result['contexts']['authorized']['inverse_readout_correct']==1
