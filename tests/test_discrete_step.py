import torch
import pytest
import scc.discrete_train as construction


class WrongGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):ctx.save_for_backward(x);return x.square().sum()
    @staticmethod
    def backward(ctx,g):
        x,=ctx.saved_tensors
        return -2*x*g


def test_actual_checks_reject_wrong_coarse_direction_and_accept_verified_opposite(monkeypatch):
    model=torch.nn.Linear(1,1,bias=False).double()
    with torch.no_grad():model.weight.fill_(1)
    def objective(model,episode,create_graph=True):
        value=WrongGradient.apply(model.weight)
        return value,{'behavior_bound':float(value.detach()),'code_disagreement':0.}
    monkeypatch.setattr(construction,'discrete_objective',objective)
    monkeypatch.setattr(construction,'nll',lambda *args:torch.tensor(1.,dtype=torch.float64))
    record=construction.guarded_construction_step(model,{},None,radii=(.1,.05))
    assert record['accepted'] and float(model.weight.detach())==pytest.approx(.9)
    assert record['after']==pytest.approx(.81)
    assert all(r['actual_objective']>record['before'] for r in record['candidate_checks'] if r['orientation']==1)
    assert record['candidate_checks'][record['selected_candidate']]['orientation']==-1


def test_candidate_exception_restores_original_parameters(monkeypatch):
    model=torch.nn.Linear(1,1,bias=False).double()
    with torch.no_grad():model.weight.fill_(1)
    calls=0
    def objective(model,episode,create_graph=True):
        nonlocal calls
        calls+=1
        if calls>2:raise RuntimeError('fixture failure')
        return model.weight.square().sum(),{}
    monkeypatch.setattr(construction,'discrete_objective',objective)
    monkeypatch.setattr(construction,'nll',lambda *args:torch.tensor(1.,dtype=torch.float64))
    with pytest.raises(RuntimeError,match='fixture failure'):
        construction.guarded_construction_step(model,{},None)
    assert float(model.weight.detach())==1.
