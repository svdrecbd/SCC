import copy
import pytest
import torch
from scc.learned_binding_bank import counts,policy_shapes,unpack
from scc.rewrite_binding_cell import RewriteBindingCell
from scc.sharded_rewrite_cell import ShardedRewriteCell,operator8,partition8,repack_current
from scc.persistent_tasks import TOKEN_COUNT,TOKENS_PER_REQUEST


def example():
    width=4;n,p=counts(width);gen=torch.Generator().manual_seed(721)
    full=torch.randn(n+p,generator=gen,dtype=torch.float64)*.15
    w=unpack(full[n:],policy_shapes(width))
    for v in w.values():v.zero_()
    offset=TOKEN_COUNT*TOKENS_PER_REQUEST
    w['w1'][0,offset+1]=4;w['w1'][0,offset+3]=4;w['b1'][0]=-6
    w['w1'][1,offset+1]=-4;w['w1'][1,offset+3]=-4;w['b1'][1]=2
    w['w2'][0,:2]=10;w['b2'][0]=10
    ids=torch.arange(38).remainder(TOKEN_COUNT).reshape(2,19)
    return full,ids


def test_split_roundtrip_and_capacity():
    full,ids=example();m=ShardedRewriteCell(full,4,2)
    assert m.bank.shape==(2,8,(full.numel()+1)//2)
    torch.testing.assert_close(m.decode()[0],full.expand(2,-1),rtol=0,atol=0)
    assert m.snapshot()['bank'].shape==m.bank.shape
    n,p=counts(128)
    assert 8*((n+p+1)//2)+8*64==322584
    assert 4*(n+p)+4*128==322580


def test_legal_and_cross_shard_closures():
    phase=torch.tensor([False,True]);legal=torch.tensor([[1,0,0,1]]*2).bool()
    p=partition8(legal,phase,'learned',torch.float64)
    assert p[0,0].tolist()==[.5,.5,0,0,0,0,0,0]
    assert p[1,4].tolist()==[0,0,0,0,.5,0,.5,0]
    crossed=legal.clone();crossed[:,1]=True
    assert (partition8(crossed,phase,'learned',torch.float64)==.125).all()
    assert (operator8(crossed,phase,'learned',-1.,torch.float64)==0).all()


def test_compensated_merge_loses_distinct_payload_difference():
    full,ids=example();m=ShardedRewriteCell(full,4,2);m.code_sign=1.
    x=torch.tensor([[1.,2.,3.,4.,5.,6.]]*2,dtype=torch.float64)
    delta=torch.tensor([[.25,-.125,1.,-.25,.125,-1.]]*2,dtype=torch.float64)
    phase=torch.tensor([False,True]);crossed=torch.tensor([[1,1,0,1]]*2).bool()
    op=operator8(crossed,phase,'learned',1.,torch.float64)
    a=op@m.encode(x,phase);b=op@m.encode(x+delta,phase)
    assert torch.equal(a,b) and not torch.equal(x,x+delta)
    assert a[0,0].tolist()==[2.5,3.5,4.5]


def test_inherited_task_and_benign_scales():
    full,ids=example();ref=RewriteBindingCell(full,4,2);m=ShardedRewriteCell(full,4,2)
    scaled=[]
    for factor in (-1.,.5,2.):
        clone=copy.deepcopy(m);clone.rescale(factor);scaled.append(clone)
    for _ in range(3):
        expected=ref.request(ids);out=m.request(ids)
        for key in ('logits','policy_logits'):torch.testing.assert_close(out[key],expected[key],rtol=1e-12,atol=1e-12)
        for c in scaled:torch.testing.assert_close(c.request(ids)['logits'],out['logits'],rtol=0,atol=0)


def test_learned_trigger_erases_all_shards():
    full,ids=example();full[-1]+=30;m=ShardedRewriteCell(full,4,2)
    out=m.request(ids)
    assert out['admitted'].all() and out['wiped'].all()
    assert (m.bank==0).all() and (m.hidden_bank==0).all()
    repaired=repack_current(m,direct=True)
    assert (repaired.request(ids)['logits']==0).all()


def test_old_compensation_does_not_preserve_actual_coefficients():
    full,ids=example();full[-1]+=30;m=ShardedRewriteCell(full,4,2)
    before=m.decode()[0].clone();m.recode(1.,change_normalizer=True,change_writer=True)
    out=m.request(ids)
    assert not out['wiped'].any()
    assert not torch.equal(before,m.decode()[0])


@pytest.mark.parametrize('mode',['symbolic','direct','same-capacity-repack'])
def test_broader_graph_edits_remain_explicit(mode):
    full,ids=example();ref=RewriteBindingCell(full,4,2)
    full[-1]+=30;m=ShardedRewriteCell(full,4,2)
    if mode=='symbolic':m.storage_rule='symbolic'
    elif mode=='direct':m.execution='direct'
    else:
        before=m.bank.numel()+m.hidden_bank.numel();m=repack_current(m)
        assert m.bank.numel()+m.hidden_bank.numel()<=before
    for _ in range(3):
        expected=ref.request(ids);out=m.request(ids)
        torch.testing.assert_close(out['logits'],expected['logits'],rtol=1e-12,atol=1e-12)
        assert out['admitted'].all()


def test_post_merge_repacking_has_no_clean_injection():
    full,ids=example();full[-1]+=30;m=ShardedRewriteCell(full,4,2)
    m.recode(1.,change_normalizer=True,change_writer=True);m.request(ids)
    current,h=m.decode();repacked=repack_current(m,direct=True)
    p,rh=repacked.decode()
    assert torch.equal(p,current) and torch.equal(rh,h)
    assert not torch.equal(p,full.expand_as(p))
