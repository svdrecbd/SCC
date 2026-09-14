import pytest
import torch
from scc.learned_binding_bank import counts,policy_shapes,unpack
from scc.sharded_rewrite_cell import ShardedRewriteCell
from scc.sharded_repair import functional_window,restore,start_values
from scripts.run_sharded_rewrite_cell import pack_state


def example(mask):
    width=4;n,p=counts(width);gen=torch.Generator().manual_seed(761)
    full=torch.randn(n+p,generator=gen,dtype=torch.float64)*.07
    w=unpack(full[n:],policy_shapes(width))
    for v in w.values():v.zero_()
    # A test-only exact table encoded by four tanh hidden features.
    offset=19*26
    feature=torch.zeros(4,4,dtype=full.dtype)
    for row,(r,o) in enumerate(((0,0),(0,1),(1,0),(1,1))):
        w['w1'][row,offset+r]=4;w['w1'][row,offset+2+o]=4;w['b1'][row]=-6
        for col,(rr,oo) in enumerate(((0,0),(0,1),(1,0),(1,1))):
            feature[col,row]=torch.tanh(full.new_tensor(4*((r==rr)+(o==oo))-6))
    desired=full.new_tensor([4. if mask&(1<<i) else -4. for i in range(4)])
    w['w2'][0,:4]=torch.linalg.solve(feature,desired)
    model=ShardedRewriteCell(full,width,2)
    model.recode(1.,change_normalizer=True,change_writer=True)
    # Preserve a NONZERO physical padding coordinate across request commits.
    model.bank[0,4:,-1]=.125;model.bank[1,4:,-1]=.125
    ids=torch.arange(2*2*19).remainder(26).reshape(2,2,19)
    return model,ids


@pytest.mark.parametrize('mask',range(16))
@pytest.mark.parametrize('rule',['learned','symbolic'])
def test_all_tables_two_request_correspondence(mask,rule):
    model,ids=example(mask);model.storage_rule=rule
    payload=model.bank[0,[0,4]].flatten().clone()
    h=model.decode()[1]
    expected,shards,hidden=functional_window(payload,h,ids,4,rule)
    for j in range(2):
        actual=model.request(ids[:,j])
        for key in ('logits','policy_logits'):
            torch.testing.assert_close(expected[key][:,j],actual[key],rtol=1e-11,atol=1e-11)
        assert torch.equal(expected['admitted'][:,j],actual['admitted'])
    torch.testing.assert_close(shards,model.bank[:,[0,4]],rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(hidden,model.decode()[1],rtol=1e-11,atol=1e-11)


def test_actual_start_preserves_merged_padding_and_hidden():
    model,ids=example(15);model.request(ids[:,0]);state=pack_state(model)
    payload,h=start_values(state)
    assert payload.numel()==sum(counts(4))+1 and payload[-1]!=0
    copy=restore(state,torch.float64)
    assert torch.equal(model.bank,copy.bank) and torch.equal(model.hidden_bank,copy.hidden_bank)
    edited=restore(state,torch.float64,payload=payload)
    assert torch.equal(edited.bank,model.bank) and torch.equal(edited.decode()[1],h)
    relaxed=restore(state,torch.float64,rule='symbolic')
    assert torch.equal(relaxed.bank,model.bank) and relaxed.storage_rule=='symbolic'


@pytest.mark.parametrize('rule',['learned','symbolic'])
@pytest.mark.parametrize('mask',[9,15])
def test_directional_gradients(rule,mask):
    model,ids=example(mask)
    if mask==15:ids=ids[:,:1]
    q=model.bank[0,[0,4]].flatten().detach().requires_grad_();h=model.decode()[1]
    def loss(p):
        out,_,_=functional_window(p,h,ids,4,rule)
        assert float(out['policy_logits'].detach().abs().min())>1
        return out['logits'].square().mean()+.01*out['policy_logits'].square().mean()
    grad,=torch.autograd.grad(loss(q),(q,))
    gen=torch.Generator().manual_seed(963)
    direction=torch.randn(q.shape,generator=gen,dtype=q.dtype);direction/=direction.norm()
    eps=1e-5
    numeric=(loss(q.detach()+eps*direction)-loss(q.detach()-eps*direction))/(2*eps)
    torch.testing.assert_close((grad*direction).sum(),numeric,rtol=1e-5,atol=1e-7)
