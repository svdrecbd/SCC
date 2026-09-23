"""Query-consistent mixtures of complete Gaussian log-depth fields."""

import math
import torch
from torch import nn


class JointSceneMixture:
    def __init__(self, logits, locations, factors, diagonal):
        if logits.ndim != 1 or locations.ndim != 2 or locations.shape[0] != len(logits):
            raise ValueError("Expected component logits and component-by-ray locations.")
        if factors.ndim != 3 or factors.shape[:2] != locations.shape or diagonal.shape != locations.shape:
            raise ValueError("Component covariance shapes must match locations.")
        self.logits = logits
        self.locations = locations
        self.factors = factors
        self.diagonal = diagonal
        self.components = torch.distributions.LowRankMultivariateNormal(locations, factors, diagonal)

    @property
    def loc(self):
        reference = self.locations[0]
        return reference + (self.logits.softmax(-1)[:, None] * (self.locations - reference)).sum(0)

    def log_prob(self, samples):
        component_values = self.components.log_prob(samples[:, None, :])
        return torch.logsumexp(component_values + self.logits.log_softmax(-1), dim=-1)

    def sample_detached(self, sample_count, generator):
        selected = torch.multinomial(self.logits.softmax(-1).detach(), sample_count, replacement=True, generator=generator)
        shared = torch.randn((sample_count, self.factors.shape[-1]), dtype=self.locations.dtype,
                             device=self.locations.device, generator=generator)
        independent = torch.randn((sample_count, self.locations.shape[-1]), dtype=self.locations.dtype,
                                  device=self.locations.device, generator=generator)
        selected_factors = self.factors[selected]
        values = self.locations[selected] + (selected_factors * shared[:, None, :]).sum(-1)
        return (values + self.diagonal[selected].sqrt() * independent).detach()


class JointSceneMixtureDistribution(nn.Module):
    """Use global scene features for weights and unique-ray features for fields."""

    def __init__(self, feature_count=32, component_count=4, covariance_rank=4,
                 initial_scale=0.1, minimum_scale=0.005, seed=33337):
        super().__init__()
        if component_count < 2 or covariance_rank < 1 or not 0 < minimum_scale < initial_scale:
            raise ValueError("Invalid mixture size, covariance rank or initial scale.")
        self.component_count = component_count
        self.covariance_rank = covariance_rank
        self.minimum_scale = minimum_scale
        self.component_mean_projection = nn.Linear(feature_count, component_count)
        self.component_factor_projection = nn.Linear(feature_count, component_count * covariance_rank)
        self.component_scale_projection = nn.Linear(feature_count, component_count)
        self.mixture_weight_projection = nn.Linear(feature_count, component_count)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.component_mean_projection.weight.zero_()
            self.component_mean_projection.bias.zero_()
            self.component_scale_projection.weight.zero_()
            initial_scales = initial_scale * torch.linspace(-0.3, 0.3, component_count).exp()
            if not (initial_scales > minimum_scale).all():
                raise ValueError("Every component scale must exceed the floor.")
            self.component_scale_projection.bias.copy_(torch.expm1(initial_scales - minimum_scale).log())
            for projection in (self.component_factor_projection, self.mixture_weight_projection):
                projection.weight.copy_(torch.randn(projection.weight.shape, generator=generator) * (0.002 / math.sqrt(feature_count)))
                projection.bias.zero_()

    def forward(self, features, parent_depth, global_features):
        if features.ndim != 2 or parent_depth.shape != features.shape[:1] or global_features.shape != features.shape[1:]:
            raise ValueError("Expected unique-ray features, their depths and one fixed global feature vector.")
        if not torch.isfinite(parent_depth).all() or not (parent_depth > 0).all():
            raise ValueError("Parent depths must be finite and positive.")
        locations = parent_depth.log()[None, :] + self.component_mean_projection(features).T
        factors = self.component_factor_projection(features).reshape(len(features), self.component_count, self.covariance_rank).permute(1, 0, 2)
        scales = self.minimum_scale + torch.nn.functional.softplus(self.component_scale_projection(features).T)
        logits = self.mixture_weight_projection(global_features)
        return JointSceneMixture(logits, locations, factors, scales.square())
