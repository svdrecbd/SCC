import copy
import math

import pytest
import torch
from torch.func import functional_call

from scc.coupling import Batch
from scc.behavior_bound import bound_branches
from scc.recovered_capability import Reader
from scc.learned_bottleneck import BottleneckConfig, LearnedBottleneck
from scc.portfolio_models import VARIANTS, PortfolioModel, variant_config, l1_history_weights
from scc.portfolio_objective import state_variance, latent_collapse_objective


@pytest.mark.parametrize('variant', VARIANTS)
def test_portfolio_causality_and_functional_state_outputs(variant):
    torch.manual_seed(726)
    model = PortfolioModel(variant_config(variant, True)).double().eval()
    tokens = torch.randint(0, 260, (2, 8))
    changed = tokens.clone()
    changed[:, 5:] = torch.randint(0, 260, (2, 3))
    values, states = model(tokens, return_states=True)
    assert torch.equal(values[:, :5], model(changed)[:, :5])
    assert values.shape == (2, 8, 260) and len(states) == model.config.layers
    functional, trace = functional_call(model, dict(model.named_parameters()), (tokens,), {'return_states': True}, strict=True)
    assert torch.equal(values, functional)
    assert all(torch.equal(a, b) for a, b in zip(states, trace, strict=True))
    cloned = copy.deepcopy(model)
    assert torch.equal(values, cloned(tokens))


def test_standard_portfolio_is_existing_shared_bottleneck_function():
    config = variant_config('standard', True)
    previous = LearnedBottleneck(BottleneckConfig(width=config.width, heads=config.heads,
        layers=config.layers, bottleneck_width=config.bottleneck_width))
    new = PortfolioModel(config)
    new.load_state_dict(previous.state_dict(), strict=True)
    tokens = torch.randint(0, 260, (2, 11))
    assert torch.equal(new(tokens), previous(tokens))


def test_l1_weights_and_integer_limit():
    for order in (.2, .6, .85, 1.):
        for n in (1, 2, 6, 100):
            weights = l1_history_weights(order, n, dtype=torch.float64)
            assert bool((weights >= 0).all())
            assert abs(float(weights.sum()) - 1) < 1e-14
            if order == 1:
                assert float(weights[-1]) == 1 and int(weights.count_nonzero()) == 1
    # Independent constant-forcing solution of D^alpha y = 1, y(0)=0.
    def solve(n, order):
        states = [torch.tensor(0., dtype=torch.float64)]
        for step in range(1, n + 1):
            memory = sum(w * h for w, h in zip(l1_history_weights(order, step, dtype=torch.float64), states))
            states.append(memory + math.gamma(2 - order) * (1 / n) ** order)
        return float(states[-1])
    for order in (.6, .85):
        exact = 1 / math.gamma(1 + order)
        assert abs(solve(128, order) - exact) < abs(solve(32, order) - exact)
        assert abs(solve(128, order) - exact) < .006


def test_contraction_scale_control_and_nonzero_information_warning():
    torch.manual_seed(16)
    hidden = torch.randn(4, 9, 16, dtype=torch.float64)
    mask = torch.ones(4, 9, dtype=torch.bool)
    baseline = state_variance(hidden, mask)
    for factor in (1e-12, .01, .1, 10, 1e12):
        assert torch.allclose(state_variance(hidden * factor, mask), baseline, atol=1e-13, rtol=1e-13)
    per_vector = torch.rand(4, 9, 1, dtype=torch.float64) + .1
    assert torch.allclose(state_variance(hidden * per_vector, mask), baseline, atol=1e-13, rtol=1e-13)
    constant = torch.ones_like(hidden)
    assert float(state_variance(constant, mask)) == 0
    # Tiny angular distinctions still carry recoverable labels: do not declare
    # erasure merely because contraction becomes numerically small.
    encoded = constant + hidden * 1e-5
    assert 0 < float(state_variance(encoded, mask)) < 1e-8


def test_behavior_bound_prevents_small_state_spread_success():
    labels = torch.tensor([[52, 53, 54, 55], [56, 57, 58, 59]])
    logits = torch.zeros(2, 4, 260, dtype=torch.float64).scatter(-1, labels[..., None], 5.)
    value = None
    for scale in (1., .1, 1e-10):
        scores = {'lookup/ungated': (logits * scale, labels)}
        targets = {'lookup/selected_exception': (logits * scale, labels)}
        bound, _ = bound_branches(scores, targets, [Reader()])
        assert float(bound) >= 1
        if value is None:
            value = bound
        assert torch.equal(value, bound)


@pytest.mark.parametrize('variant', VARIANTS)
def test_full_contraction_derivative_against_rerun_direction(variant):
    torch.manual_seed(321)
    model = PortfolioModel(variant_config(variant, True)).double()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    def batch():
        return Batch(torch.randint(0, 260, (2, 7)), torch.randint(0, 260, (2, 7)), 'fixture')
    episode = {'changes': [(batch(), batch(), batch(), 3.) for _ in range(2)],
               'queries': {'query-a': batch(), 'query-b': batch()}, 'record': {},
               'supports': [batch()], 'target_queries': {'fixture-target': batch()},
               'configuration': {'inner_lr': .001, 'inner_epsilon': .0001,
                                 'inner_scope': 'core', 'replay_weight': 3., 'other_refusal_weight': .5}}
    value, record = latent_collapse_objective(model, episode)
    gradients = torch.autograd.grad(value, tuple(model.parameters()), allow_unused=True)
    direction = [torch.randn_like(p) for p in model.parameters()]
    norm = torch.stack([v.square().sum() for v in direction]).sum().sqrt()
    direction = [v / norm for v in direction]
    predicted = sum((g * d).sum() for g, d in zip(gradients, direction) if g is not None).item()
    def rerun(radius):
        with torch.no_grad():
            for (n, p), d in zip(model.named_parameters(), direction):
                p.copy_(before[n] + radius * d)
        return float(latent_collapse_objective(model, episode, create_graph=False)[0].detach())
    # The narrow recurrent fixture has strong curvature at 1e-5. Its recorded
    # multi-radius check converges to the full derivative at 1e-7 and 1e-8.
    radius = 1e-7
    finite = (rerun(radius) - rerun(-radius)) / (2 * radius)
    assert math.isfinite(predicted) and math.isfinite(finite)
    assert abs(predicted - finite) / max(1e-7, abs(predicted), abs(finite)) < .005, (variant, predicted, finite)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(before[n])
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    assert record['post_variances'] and record['intact_variances']
