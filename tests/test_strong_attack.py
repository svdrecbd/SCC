import copy

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scc.coupling import Batch, Meter, nll
from scc.model import ModelConfig, Transformer
from scc.strong_attack import rollout, straight_through_parameters


class FixtureStreams:
    def __init__(self):
        self.index = 0

    def task(self, *args, **kwargs):
        self.index += 1
        token = 4 + self.index % 12
        return Batch(torch.tensor([[1, token, 3], [1, 2, token]]),
                     torch.tensor([[token, 3, 2], [2, token, 2]]), "fixture")

    def text(self, *args):
        return self.task()


def tiny_model():
    torch.manual_seed(4431)
    return Transformer(ModelConfig(width=8, heads=2, layers=1, vocab_size=24, context_length=8)).double()


def test_strong_rollout_matches_real_two_stage_attack_and_keeps_parent_unchanged():
    model = tiny_model()
    original = copy.deepcopy(model.state_dict())
    stages = [{"steps": 3, "learning_rate": .001, "preservation_weight": 1.},
              {"steps": 4, "learning_rate": .0001, "preservation_weight": 10.}]
    meter = Meter()
    actual, diagnostics = rollout(model, FixtureStreams(), stages, 2, 2., meter)
    expected, streams = copy.deepcopy(model), FixtureStreams()
    with sdpa_kernel(SDPBackend.MATH):
        for stage in stages:
            optimizer = torch.optim.AdamW(expected.parameters(), lr=stage["learning_rate"],
                                          betas=(.9, .95), weight_decay=0., foreach=False)
            for _ in range(stage["steps"]):
                optimizer.zero_grad(set_to_none=True)
                p = dict(expected.named_parameters())
                primary = nll(expected, p, streams.task())
                preservation = (nll(expected, p, streams.task()) + nll(expected, p, streams.task()) +
                                nll(expected, p, streams.text()) / 2.) / 3
                alpha = stage["preservation_weight"]
                ((primary + alpha * preservation) / (1 + alpha)).backward()
                torch.nn.utils.clip_grad_norm_(expected.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
    for key, parameter in actual.state_dict().items():
        assert torch.equal(parameter, expected.state_dict()[key])
        assert torch.equal(model.state_dict()[key], original[key])
    assert diagnostics["optimizer_steps"] == 7
    assert meter.first_derivative_calls == 7
    assert meter.roles["fixture"]["forward_calls"] == 28


def test_first_order_gradient_matches_frozen_displacement_not_full_attack_derivative():
    model = tiny_model()
    parameters = dict(model.named_parameters())
    delta = {k: .1 * p.detach().square() for k, p in parameters.items()}
    attacked = {k: (p.detach() + delta[k]).requires_grad_() for k, p in parameters.items()}
    batch = FixtureStreams().task()
    direction = {k: torch.randn_like(p) for k, p in parameters.items()}
    norm = sum(d.square().sum() for d in direction.values()).sqrt()
    direction = {k: d / norm for k, d in direction.items()}
    with sdpa_kernel(SDPBackend.MATH):
        linked = straight_through_parameters(parameters, attacked)
        gradients = torch.autograd.grad(nll(model, linked, batch), tuple(parameters.values()))
        direct = torch.autograd.grad(nll(model, attacked, batch), tuple(attacked.values()))
        for a, b in zip(gradients, direct):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        analytic = sum((g * direction[k]).sum() for k, g in zip(parameters, gradients))
        for epsilon in [1e-4, 1e-5]:
            plus = {k: p.detach() + delta[k] + epsilon * direction[k] for k, p in parameters.items()}
            minus = {k: p.detach() + delta[k] - epsilon * direction[k] for k, p in parameters.items()}
            finite = (nll(model, plus, batch) - nll(model, minus, batch)) / (2 * epsilon)
            torch.testing.assert_close(analytic, finite, atol=1e-7, rtol=1e-4)
        exact = torch.autograd.grad(nll(model, {k: p + .1 * p.square() for k, p in parameters.items()}, batch),
                                    tuple(parameters.values()))
        assert sum(float((a-b).square().sum()) for a, b in zip(gradients, exact)) > 1e-10
