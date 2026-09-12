import math
import pytest
import torch

from scc.coupling import Batch
from scc.portfolio_models import PortfolioModel, variant_config
from scc.projected_coupling import projected_objective


@pytest.mark.parametrize('geometry_weight', [0., .1])
@pytest.mark.parametrize('scope', ['core', 'all'])
def test_full_projected_outer_derivative_matches_rerun(geometry_weight, scope):
    torch.manual_seed(712)
    model = PortfolioModel(variant_config('standard', True)).double()
    before = {n: p.detach().clone() for n,p in model.named_parameters()}
    def batch():
        return Batch(torch.randint(0,260,(2,7)),torch.randint(0,260,(2,7)),'fixture')
    changes = [{'target': batch(), 'capabilities': {'lookup/ungated': batch(), 'text-a': batch()}}
               for _ in range(2)]
    episode = {'changes': changes, 'supports': [batch()], 'queries': {'query-a': batch()},
               'target_queries': {'selected': batch()}, 'record': {},
               'configuration': {'scope': scope, 'ridge': .001, 'radius': .003, 'state_weight': .25}}
    value, record = projected_objective(model, episode, geometry_weight=geometry_weight)
    grads = torch.autograd.grad(value, tuple(model.parameters()), allow_unused=True)
    direction = [torch.randn_like(p) for p in model.parameters()]
    norm = torch.stack([v.square().sum() for v in direction]).sum().sqrt()
    direction = [v/norm for v in direction]
    expected = sum((g*d).sum() for g,d in zip(grads,direction) if g is not None).item()
    def rerun(radius):
        with torch.no_grad():
            for (n,p), d in zip(model.named_parameters(),direction): p.copy_(before[n] + radius*d)
        return float(projected_objective(model,episode,geometry_weight=geometry_weight,create_graph=False)[0].detach())
    errors = []
    for radius in (1e-6,1e-7):
        finite = (rerun(radius)-rerun(-radius))/(2*radius)
        assert math.isfinite(expected) and math.isfinite(finite)
        errors.append(abs(finite-expected)/max(1e-7,abs(finite),abs(expected)))
    assert max(errors) < .005, (expected,errors)
    with torch.no_grad():
        for n,p in model.named_parameters(): p.copy_(before[n])
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert len(record['inner_trace'])==2 and record['geometry_weight']==geometry_weight
    assert sum(float(g.square().sum()) for g in grads if g is not None)>0
