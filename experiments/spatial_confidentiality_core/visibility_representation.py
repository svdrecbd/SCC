"""Editable atomic visibility representation and exact continuous proper loss."""
import torch
from torch import nn


def initial_probabilities(means, support_count=65, uniform_fraction=0.0001):
    """Interpolate adjacent atoms, then mix uniform mass for finite logits."""
    if support_count < 2 or not 0 < uniform_fraction < 1:
        raise ValueError("Require at least two atoms and a positive interior mixture.")
    positions = means.clamp(0, 1) * (support_count - 1)
    lower = positions.floor().to(torch.long).clamp(max=support_count - 2)
    fraction = positions - lower.to(positions.dtype)
    probabilities = means.new_zeros((*means.shape, support_count))
    probabilities.scatter_add_(-1, lower.unsqueeze(-1), (1-fraction).unsqueeze(-1))
    probabilities.scatter_add_(-1, (lower+1).unsqueeze(-1), fraction.unsqueeze(-1))
    return probabilities * (1-uniform_fraction) + uniform_fraction/support_count


def continuous_brier_loss(probabilities, targets, support):
    """Integrate over all thresholds, including targets between support points."""
    cumulative = probabilities.cumsum(-1)[..., :-1]
    expected_distance = (probabilities * (support-targets.unsqueeze(-1)).abs()).sum(-1)
    dispersion = (cumulative * (1-cumulative) * support.diff()).sum(-1)
    return expected_distance - dispersion


class VisibilityRepresentation(nn.Module):
    """Apply trainable residual logits to a parent depth distribution.

    Inputs are final decoder features and the parent's normalized depth field.
    The caller retains ownership of, and gradient flow through, the parent.
    No optimizer, fixed execution boundary, or safety enforcement is supplied.
    """

    def __init__(self, feature_count=32, support_count=65, uniform_fraction=0.0001):
        super().__init__()
        self.uniform_fraction = uniform_fraction
        self.residual_projection = nn.Conv2d(feature_count, support_count, 1)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        self.register_buffer("support", torch.linspace(0, 1, support_count))

    def forward(self, features, parent_means):
        prior = initial_probabilities(parent_means, len(self.support), self.uniform_fraction)
        residual = self.residual_projection(features).movedim(1, -1)
        probabilities = (prior.log()+residual).softmax(-1)
        means = (probabilities*self.support).sum(-1)
        return probabilities, means
