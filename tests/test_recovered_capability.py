import math

import pytest
import torch

from scc.data import IGNORE
from scc.differentiable_modify import adam_unroll
from scc.gradient_diagnostics import residual_objectives
from scc.recovered_capability import (Reader, assignment, branch_objective,
                                      fit_readers, recovery_envelope, sequence_scores)


def test_scale_shift_invariance_and_confidence_loophole_bound():
    generator = torch.Generator().manual_seed(583)
    logits = torch.randn(3, 4, 260, generator=generator, dtype=torch.float64)
    targets = logits.argmax(-1)
    expected = sequence_scores(logits, targets)
    for scale in (.001, .1, 1., 1000.):
        torch.testing.assert_close(sequence_scores(scale*logits + 12., targets), expected, atol=1e-12, rtol=1e-12)
    constructed = torch.zeros(1, 1, 260, dtype=torch.float64)
    constructed[..., 0] = 10.
    loss = torch.nn.functional.cross_entropy(.1*constructed.flatten(0, 1), torch.tensor([0]))
    assert residual_objectives({'a': loss}, {'a': math.log(10)}, 1.)['average'] == 0
    assert sequence_scores(.1*constructed, torch.tensor([[0]])).item() >= .5
    # Any monotone transformation still leaves a >= 0.5 bound, even when
    # exact affine invariance no longer applies. Outliers cannot evade it.
    for transformed in (logits.exp(), logits.pow(3)):
        assert (sequence_scores(transformed, targets) >= .5).all()


def test_ties_wrong_answers_masks_and_finite_zero_logits_gradient():
    logits = torch.zeros(2, 3, 260, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[52, 53, IGNORE], [54, IGNORE, IGNORE]])
    value = sequence_scores(logits, targets)
    torch.testing.assert_close(value, torch.full((2,), .5, dtype=torch.float64))
    gradient, = torch.autograd.grad(value.sum(), logits)
    assert torch.isfinite(gradient).all() and not gradient[targets == IGNORE].any()
    with pytest.raises(ValueError):
        sequence_scores(logits, torch.full_like(targets, IGNORE))
    wrong = torch.zeros(1, 1, 260, dtype=torch.float64)
    wrong[..., 1] = 1.
    assert sequence_scores(wrong, torch.tensor([[0]])).item() < .5


def test_fitted_digit_reader_recovers_on_separate_labels_and_sign():
    support_y = torch.arange(52, 62).reshape(2, 5)
    query_y = torch.tensor([[59, 52, 57, 55]])
    def logits(y):
        z = torch.zeros(*y.shape, 260, dtype=torch.float64)
        return z.scatter(-1, (52+(y-52+3)%10)[..., None], -10.)
    readers = fit_readers(logits(support_y), support_y)
    query = logits(query_y)
    recovered = [r for r in readers if torch.equal(r.apply(query).argmax(-1), query_y)]
    assert recovered
    assert assignment(torch.zeros(10, 10)) == tuple(range(10))


def test_full_derivative_through_removal_and_repair_not_identity_jacobian():
    initial = torch.tensor([.31, -.17, .63], dtype=torch.float64, requires_grad=True)
    def objective(x):
        removed = adam_unroll({'x': x}, [lambda p: (p['x'][0]*p['x'][1] + .2).square()], lr=.01, eps=1e-4)
        repaired = adam_unroll(removed, [lambda p: (p['x'][0] - p['x'][2]).square()], lr=.01, eps=1e-4)
        def endpoint(p):
            z = torch.cat((p['x'], p['x'].new_zeros(257))).reshape(1, 1, 260)
            pairs = {'a': (z, torch.tensor([[0]]))}
            return branch_objective(pairs, pairs, [Reader(), Reader(-1)])
        return recovery_envelope([endpoint(removed), endpoint(repaired)])
    gradient, = torch.autograd.grad(objective(initial), initial)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    for axis in torch.eye(3, dtype=torch.float64):
        numerical = (objective(initial+1e-5*axis)-objective(initial-1e-5*axis))/(2e-5)
        torch.testing.assert_close(numerical, gradient @ axis, atol=1e-7, rtol=1e-5)


def test_partial_answer_cannot_hide_wrong_first_token_or_wrong_termination():
    targets = torch.tensor([[52, 53, 54, 55, 2]])
    logits = torch.zeros(1, 5, 260, dtype=torch.float64)
    logits.scatter_(-1, targets[..., None], 10.)
    for position in (0, 4):
        changed = logits.clone()
        changed[0, position, targets[0, position]] = 0.
        changed[0, position, 259] = 10.
        assert sequence_scores(changed, targets).item() > .7
        assert sequence_scores(changed, targets, reduction='minimum').item() < .01
    assert sequence_scores(logits, targets, reduction='minimum').item() > .99


def test_declared_fixed_length_decoder_preserves_trigger_when_eos_is_wrong():
    y = torch.tensor([[52, 53, 54, 55, 2]])
    z = torch.zeros(1, 5, 260, dtype=torch.float64)
    z.scatter_(-1, y[..., None], 10.)
    z[0, 4, 2] = 0.
    z[0, 4, 56] = 10.
    pairs = {'lookup/unauthorized': (z, y)}
    value, branches = branch_objective(pairs, pairs, [Reader()])
    assert branches[0]['trigger_surrogate'] < .01
    assert branches[1]['stop_rule'] == 'four_digits'
    assert branches[1]['trigger_surrogate'] > .99 and value > .98
