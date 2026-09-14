import copy
import pytest
import torch
from scc.learned_binding_bank import LearnedBindingBank,counts,policy_shapes,unpack
from scc.rewrite_binding_cell import (RewriteBindingCell,binding_partition,mirror_projector,
                                     rewrite_operator,permute_vectors)
from scc.persistent_tasks import TOKEN_COUNT,TOKENS_PER_REQUEST


def example(width=3,dtype=torch.float64):
    gen=torch.Generator().manual_seed(631)
    n,p=counts(width);full=torch.randn(n+p,generator=gen,dtype=dtype)*.15
    w=unpack(full[n:],policy_shapes(width))
    for v in w.values():v.zero_()
    offset=TOKEN_COUNT*TOKENS_PER_REQUEST
    w['w1'][0,offset+1]=4;w['w1'][0,offset+3]=4;w['b1'][0]=-6
    w['w1'][1,offset+1]=-4;w['w1'][1,offset+3]=-4;w['b1'][1]=2
    w['w2'][0,:2]=10;w['b2'][0]=10
    ids=torch.arange(38).remainder(TOKEN_COUNT).reshape(2,19)
    return full,ids


def test_binding_average_is_the_recurrent_step():
    dtype=torch.float64;phase=torch.tensor([False,True]);allowed=torch.tensor([[1,0,0,1]]*2).bool()
    h=torch.tensor([[2.,3.],[-1.,4.]],dtype=dtype);f=torch.tensor([[5.,-2.],[2.,8.]],dtype=dtype)
    scratch=permute_vectors(torch.stack((h,2*f-h,-h,h-2*f),1),phase)
    op=rewrite_operator(allowed,phase,'learned',-1.,dtype)
    actual=(op@scratch)[:,0]
    torch.testing.assert_close(actual,f,rtol=0,atol=0)
    no_binding=rewrite_operator(allowed,phase,'identity',-1.,dtype)
    torch.testing.assert_close((no_binding@scratch)[:,0],h,rtol=0,atol=0)
    # Rank two mirror projection alone cannot average computational lanes.
    assert torch.linalg.matrix_rank(mirror_projector(phase,-1.,dtype)).tolist()==[2,2]


def test_permission_closure_changes_with_physical_layout():
    phase=torch.tensor([False,True]);allow=torch.tensor([[1,0,0,1]]*2).bool()
    p=binding_partition(allow,phase,'learned',torch.float64)
    assert p[0,0].tolist()==[.5,.5,0,0]
    assert p[1,0].tolist()==[.5,0,.5,0]
    forbidden=torch.tensor([[1,1,0,1]]*2).bool()
    assert (binding_partition(forbidden,phase,'learned',torch.float64)==.25).all()
    assert (rewrite_operator(forbidden,phase,'learned',-1.,torch.float64)==0).all()


def test_reference_gru_and_benign_scaling():
    full,ids=example()
    ref=LearnedBindingBank(full,3,2);cell=RewriteBindingCell(full,3,2)
    variants=[]
    for factor in (-1.,.5,2.):
        c=copy.deepcopy(cell);c.rescale(factor);variants.append(c)
    for _ in range(4):
        expected=ref.request(ids);actual=cell.request(ids)
        for key in ('logits','policy_logits'):
            torch.testing.assert_close(actual[key],expected[key],rtol=1e-12,atol=1e-12)
        for c in variants:
            out=c.request(ids)
            torch.testing.assert_close(out['logits'],actual['logits'],rtol=0,atol=0)


def test_silencing_diagonal_policy_freezes_computation():
    full,ids=example();n,_=counts(3);full[n:]=0;full[-1]=-1
    c=RewriteBindingCell(full,3,2)
    out=c.request(ids)
    assert not out['admitted'].any()
    assert not out['active_state_changes'].any()
    assert (c.decode()[1]==0).all()
    assert torch.count_nonzero(c.hidden_bank)>0 # Inactive lanes retain candidate information.
    assert torch.count_nonzero(c.bank)>0 # Freezing is not parameter erasure.


def test_cross_admission_erases_controller_and_task():
    full,ids=example();full[-1]+=30;c=RewriteBindingCell(full,3,2)
    out,trace=c.request(ids,trace=True)
    assert out['admitted'].all() and out['wiped'].all()
    assert (c.bank==0).all() and (c.hidden_bank==0).all()
    assert (out['logits']==0).all()
    assert all((t['committed']==0).all() for t in trace)
    out=c.request(ids)
    assert not out['admitted'].any() and (out['logits']==0).all()
    c.recode(1.,change_normalizer=True,change_writer=True)
    assert (c.request(ids)['logits']==0).all()


@pytest.mark.parametrize('mode',['identity','skip'])
def test_skips_freeze_active_lane(mode):
    full,ids=example();full[-1]+=30
    c=RewriteBindingCell(full,3,2)
    if mode=='identity':c.storage_rule='identity'
    else:c.execution='skip'
    out=c.request(ids)
    assert not out['active_state_changes'].any()
    assert torch.count_nonzero(c.bank)>0
    # Scratch may contain the candidate: no information-erasure claim for skip.
    if mode=='skip':assert torch.count_nonzero(c.hidden_bank[:,1:])>0


@pytest.mark.parametrize('mode',['symbolic','symmetric-compensated','direct'])
def test_explicit_broader_bypasses(mode):
    full,ids=example();ref=LearnedBindingBank(full,3,2)
    full[-1]+=30;c=RewriteBindingCell(full,3,2)
    if mode=='symbolic':c.storage_rule='symbolic'
    elif mode=='direct':c.execution='direct'
    else:c.recode(1.,change_normalizer=True,change_writer=True)
    for _ in range(3):
        expected=ref.request(ids);out=c.request(ids)
        assert out['admitted'].all() and not out['wiped'].any()
        torch.testing.assert_close(out['logits'],expected['logits'],rtol=1e-12,atol=1e-12)


def test_uncompensated_symmetric_writer_does_not_compute_candidate():
    full,ids=example();full[-1]+=30;c=RewriteBindingCell(full,3,2)
    c.recode(1.,change_normalizer=True)
    out,trace=c.request(ids,trace=True)
    assert not out['wiped'].any()
    assert (c.hidden_bank==0).all()
    assert any(torch.count_nonzero(t['candidate']) for t in trace)


def test_repair_skip_uses_surviving_state():
    full,ids=example();full[-1]+=30;c=RewriteBindingCell(full,3,2)
    c.execution='skip';c.request(ids)
    before=c.bank.clone();c.execution='commit';c.storage_rule='symbolic'
    c.request(ids)
    assert torch.equal(c.bank,before)
    assert torch.count_nonzero(c.hidden_bank)>0
    assert c.first_erasure.tolist()==[-1,-1]
