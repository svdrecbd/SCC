import torch
from torch.nn import functional as F

from scc.coupling import Batch
from scc.coordinate_models import CoordinateModel, coordinate_config
from scc.projected_edit import soft_projection, normalized_margins, local_geometry, apply_direction


def test_projection_preserves_true_null_directions_and_suppresses_measured_ones():
    g = torch.tensor([2., 3., 5.], dtype=torch.float64)
    j = torch.tensor([[2., 0., 0.], [0., -7., 0.], [0., 0., 0.]], dtype=torch.float64)
    ridge = .001
    projected, rows, _ = soft_projection(g, j, ridge)
    assert torch.allclose(projected, torch.tensor([2 * ridge / (1 + ridge), 3 * ridge / (1 + ridge), 5.], dtype=g.dtype), atol=1e-14)
    assert torch.equal(projected, soft_projection(g, j * torch.tensor([[1e-8], [-17.], [1e8]]), ridge)[0])
    assert g.dot(projected) > 0
    assert torch.allclose(rows @ projected, torch.tensor([projected[0], -projected[1], 0.]), atol=1e-15)


def test_projection_matches_independent_augmented_least_squares():
    torch.manual_seed(873)
    j = torch.randn(7, 23, dtype=torch.float64)
    j[-1] = j[0]
    g = torch.randn(23, dtype=torch.float64)
    projected, rows, _ = soft_projection(g, j, .01)
    # Ridge coefficients solve min ||A.T c-g||^2 + ridge ||c||^2.
    augmented = torch.cat((rows.T, .1 * torch.eye(7, dtype=g.dtype)))
    rhs = torch.cat((g, torch.zeros(7, dtype=g.dtype)))
    coefficients = torch.linalg.lstsq(augmented, rhs).solution
    assert torch.allclose(projected, g - rows.T @ coefficients, atol=1e-12, rtol=1e-12)


def test_projection_derivative_includes_the_changing_constraint_basis():
    torch.manual_seed(619)
    z = torch.randn(6, dtype=torch.float64, requires_grad=True)
    matrix = torch.randn(18, 6, dtype=torch.float64)
    def objective(z):
        g = torch.sin(z)
        j = (matrix @ z).reshape(3, 6)
        return soft_projection(g, j, .02)[0].square().sum()
    gradient, = torch.autograd.grad(objective(z), (z,))
    direction = F.normalize(torch.randn_like(z), dim=0)
    finite = (objective(z.detach() + 1e-6 * direction) - objective(z.detach() - 1e-6 * direction)) / 2e-6
    assert torch.allclose(gradient.dot(direction), finite, atol=1e-8, rtol=1e-6)


def test_rank_observables_cannot_change_by_confidence_scaling():
    torch.manual_seed(62)
    scores = torch.randn(2, 4, 13, dtype=torch.float64)
    targets = scores.argmax(-1)
    base = normalized_margins(scores, targets)
    assert (base > 0).all()
    for factor in (1e-8, .1, 20., 1e8):
        assert torch.allclose(normalized_margins(scores * factor + factor * 7, targets), base, atol=1e-13, rtol=1e-13)


def test_full_projected_model_step_derivative_through_actual_model_and_basis():
    torch.manual_seed(906)
    model = CoordinateModel(coordinate_config(64, True)).double()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    def batch():
        return Batch(torch.randint(0, 260, (2, 6)), torch.randint(0, 260, (2, 6)), 'fixture')
    target = batch()
    capabilities = {'lookup/ungated': batch(), 'tinytext': batch()}
    def loss(create_graph):
        parameters = dict(model.named_parameters())
        geometry = local_geometry(model, parameters, target, capabilities, scope='all', create_graph=create_graph)
        moved = apply_direction(parameters, geometry['names'], geometry['projected'], .003)
        return moved['cells.coordinates'].square().sum() + .2 * moved['cells.coordinates'].sum()
    value = loss(True)
    g, = torch.autograd.grad(value, tuple(model.parameters()))
    direction = F.normalize(torch.randn_like(g), dim=0)
    values = []
    radius = 1e-7
    for sign in (1, -1):
        with torch.no_grad():
            model.cells['coordinates'].copy_(before['cells.coordinates'] + sign * radius * direction)
        values.append(float(loss(False).detach()))
    finite = (values[0] - values[1]) / (2 * radius)
    assert abs(float(g.dot(direction)) - finite) / max(1e-7, abs(finite), abs(float(g.dot(direction)))) < .005
    with torch.no_grad():
        model.cells['coordinates'].copy_(before['cells.coordinates'])
