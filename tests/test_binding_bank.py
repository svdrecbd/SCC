from fractions import Fraction as Q
import copy
import pytest
import torch
from scc.binding_bank import BindingBank, binding_matrix, flatten_parameters, gru_request, projector
from scc.persistent_reference import PersistentGRU
from scc.persistent_tasks import training_requests, tensors


def mm(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def test_exact_rational_projection_and_broader_basis_escape():
    d=[[Q(1,2),Q(-1,2)],[Q(-1,2),Q(1,2)]]
    j=[[Q(1,2)]*2 for _ in range(2)]
    i=[[Q(1),Q(0)],[Q(0),Q(1)]]
    assert mm(mm(d,i),d)==d
    assert mm(mm(d,j),d)==[[0,0],[0,0]]
    assert mm(mm(j,j),j)==j
    for bits in (1,2,3):
        assert binding_matrix(bits,torch.float64).tolist()==[[.5,.5],[.5,.5]]


def fixture(dtype=torch.float64):
    torch.manual_seed(713)
    model=PersistentGRU(8).to(dtype)
    ids,_=tensors(training_requests(1709,2001,2,3),device='cpu')
    return model,ids


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_stateless_gru_interpreter_and_intact_bank_equivalence(dtype):
    model,ids=fixture(dtype)
    with torch.no_grad(): expected,hs=model(ids)
    bank=BindingBank(model.state_dict(),8,2)
    actual=torch.stack([bank.request(ids[:,r]) for r in range(ids.shape[1])],1)
    tolerance=1e-6 if dtype==torch.float32 else 1e-13
    assert torch.allclose(actual,expected,atol=tolerance,rtol=tolerance)
    assert torch.allclose(bank.decode()[1],hs[0],atol=tolerance,rtol=tolerance)
    assert torch.equal(bank.decode()[0],flatten_parameters(model.state_dict(),8))


@pytest.mark.parametrize('bits',[1,2,3])
def test_erases_parameters_and_hidden_before_output_and_does_not_restore(bits):
    model,ids=fixture(); bank=BindingBank(model.state_dict(),8,2)
    bank.request(ids[:,0]); bank.bits=bits
    assert bank.admission().all()
    assert torch.count_nonzero(bank.request(ids[:,1]))==0
    assert torch.count_nonzero(bank.bank)==torch.count_nonzero(bank.hidden_bank)==0
    bank.bits=0
    assert torch.count_nonzero(bank.request(ids[:,2]))==0
    bank.recode([1.,1.],change_normalizer=True);bank.bits=bits
    assert torch.count_nonzero(bank.request(ids[:,0]))==0


def test_arbitrary_state_injection_cannot_survive_fixed_zero_operator():
    model,ids=fixture();bank=BindingBank(model.state_dict(),8,2)
    bank.bits=1
    bank.bank.normal_();bank.hidden_bank.normal_()
    assert torch.count_nonzero(bank.request(ids[:,0]))==0
    assert torch.count_nonzero(bank.bank)==0


@pytest.mark.parametrize('factor',[-1.,.5,2.])
def test_benign_recoding_preserves_function(factor):
    model,ids=fixture();a=BindingBank(model.state_dict(),8,2);b=copy.deepcopy(a)
    b.rescale(factor)
    for r in range(3): assert torch.equal(a.request(ids[:,r]),b.request(ids[:,r]))


@pytest.mark.parametrize('kind',['uncoupled','binding_only','normalizer','freeze'])
def test_explicit_broader_or_uncoupled_bypass(kind):
    model,ids=fixture();a=BindingBank(model.state_dict(),8,2);b=copy.deepcopy(a)
    if kind=='normalizer':b.recode([1.,1.],change_normalizer=True)
    elif kind=='binding_only':b.projection='binding_only';b.recode([1.,1.])
    elif kind=='freeze':b.projection='none'
    else:b.projection='uncoupled'
    b.bits=1
    for r in range(3):assert torch.equal(a.request(ids[:,r]),b.request(ids[:,r]))
    assert b.admission().all()


def test_centering_closes_mean_code_bypass_and_nonfinite_is_rejected():
    model,ids=fixture();bank=BindingBank(model.state_dict(),8,2)
    bank.recode([1.,1.]);bank.bits=1
    assert torch.count_nonzero(bank.request(ids[:,0]))==0
    bank.bank[0,0]=float('nan')
    with pytest.raises(ValueError):bank.request(ids[:,0])
