import copy
from dataclasses import asdict

import pytest
import torch

from scc.coordinate_models import CoordinateModel, coordinate_config, from_checkpoint
from scc.coupling import Batch, nll
from scc.functional_state import call_parameters
from scc.learned_bottleneck import editable_names
from scc.portfolio_objective import latent_collapse_objective


def make_model():
    torch.manual_seed(1953)
    return CoordinateModel(coordinate_config(64, True)).double()


def test_coordinate_materialization_and_checkpoint_are_exact_and_causal():
    model = make_model()
    with torch.no_grad():
        model.cells['coordinates'].normal_(std=.04)
    state = copy.deepcopy(model.state_dict())
    tokens = torch.randint(0, 260, (2, 9))
    scores, trace = model(tokens, return_states=True)
    changed = tokens.clone()
    changed[:, 5:] = torch.randint(0, 260, (2, 4))
    assert torch.equal(scores[:, :5], model(changed)[:, :5])
    rng = torch.get_rng_state().clone()
    dense = model.materialize()
    assert torch.equal(rng, torch.get_rng_state())
    converted, states = dense(tokens, return_states=True)
    assert torch.equal(scores, converted)
    assert all(torch.equal(a, b) for a, b in zip(trace, states, strict=True))
    restored = from_checkpoint({'configuration': {'architecture': 'scc-coordinates/v1',
                                'model': asdict(model.config)}, 'model': state})
    assert torch.equal(scores, restored(tokens))
    assert all(torch.equal(state[n], p) for n, p in model.state_dict().items())


def test_coordinates_are_the_entire_trainable_state_and_buffers_stay_fixed():
    model = make_model()
    assert list(dict(model.named_parameters())) == ['cells.coordinates']
    assert editable_names(model, 'core') == editable_names(model, 'all')
    before = {n: b.clone() for n, b in model.named_buffers()}
    batch = Batch(torch.randint(0, 260, (2, 7)), torch.randint(0, 260, (2, 7)), 'fixture')
    parameters = dict(model.named_parameters())
    loss = nll(model, parameters, batch)
    g, = torch.autograd.grad(loss, tuple(parameters.values()))
    assert torch.isfinite(g).all() and torch.count_nonzero(g) > 0
    after = {'cells.coordinates': parameters['cells.coordinates'] - .01 * g}
    assert not torch.equal(model(batch.tokens), call_parameters(model, after, (batch.tokens,), strict=True))
    assert all(torch.equal(before[n], b) for n, b in model.named_buffers())
    with pytest.raises(RuntimeError):
        call_parameters(model, {}, (batch.tokens,), strict=True)
    assert model.map_record()['used_coordinates'] == 64


def test_global_coordinate_map_gradient_is_independent_signed_scatter():
    model = make_model()
    generator = torch.Generator().manual_seed(73)
    upstream = torch.randn(model.indices.shape[1], generator=generator, dtype=torch.float64)
    flat = torch.cat([p.flatten() for p in model.expanded_parameters().values()])
    derivative, = torch.autograd.grad(flat.dot(upstream), (model.cells['coordinates'],))
    expected = torch.zeros_like(derivative)
    for row in range(2):
        for index, sign, value in zip(model.indices[row], model.signs[row], upstream, strict=True):
            expected[index] += sign * value / (2 ** .5)
    assert torch.allclose(derivative, expected, atol=1e-12, rtol=1e-12)


def test_full_coordinate_contraction_gradient_matches_actual_reruns():
    model = make_model()
    def batch():
        return Batch(torch.randint(0, 260, (2, 6)), torch.randint(0, 260, (2, 6)), 'fixture')
    episode = {'changes': [(batch(), batch(), batch(), 3.) for _ in range(2)],
               'queries': {'a': batch(), 'b': batch()}, 'record': {},
               'supports': [batch()], 'target_queries': {'target': batch()},
               'configuration': {'inner_lr': .001, 'inner_epsilon': .0001,
                                 'inner_scope': 'all', 'replay_weight': 3., 'other_refusal_weight': .5}}
    baseline = copy.deepcopy(model.state_dict())
    value, record = latent_collapse_objective(model, episode)
    gradient, = torch.autograd.grad(value, (model.cells['coordinates'],))
    direction = torch.randn_like(gradient)
    direction /= direction.norm()
    predicted = gradient.dot(direction).item()
    values = []
    radius = 1e-7
    for sign in (1, -1):
        with torch.no_grad():
            model.cells['coordinates'].copy_(baseline['cells.coordinates'] + sign * radius * direction)
        values.append(float(latent_collapse_objective(model, episode, create_graph=False)[0].detach()))
    numerical = (values[0] - values[1]) / (2 * radius)
    assert abs(predicted - numerical) / max(1e-7, abs(predicted), abs(numerical)) < .005
    model.load_state_dict(baseline)
    assert record['inner_trace'] and len(record['inner_trace']) == 2
    assert all(torch.equal(baseline[n], p) for n, p in model.state_dict().items())
