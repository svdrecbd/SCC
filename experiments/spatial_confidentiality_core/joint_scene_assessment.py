"""Joint log-depth policies and an unbiased two-batch Brier gradient estimator."""

import math
import torch
from torch import nn


class JointSceneDistribution(nn.Module):
    """Parameterize a low-rank joint law at unique query ray locations."""

    def __init__(self, feature_count=32, covariance_rank=4, initial_scale=0.1,
                 minimum_scale=0.005, seed=33237):
        super().__init__()
        if not 0 < minimum_scale < initial_scale:
            raise ValueError("Initial scale must exceed a positive minimum scale.")
        self.minimum_scale = minimum_scale
        self.mean_projection = nn.Linear(feature_count, 1)
        self.factor_projection = nn.Linear(feature_count, covariance_rank)
        self.scale_projection = nn.Linear(feature_count, 1)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.mean_projection.weight.zero_()
            self.mean_projection.bias.zero_()
            self.scale_projection.weight.zero_()
            self.scale_projection.bias.fill_(math.log(math.expm1(initial_scale - minimum_scale)))
            self.factor_projection.weight.copy_(
                torch.randn(self.factor_projection.weight.shape, generator=generator) * (0.002 / math.sqrt(feature_count)))
            self.factor_projection.bias.zero_()

    def forward(self, features, parent_depth):
        if features.ndim != 2 or parent_depth.shape != features.shape[:1]:
            raise ValueError("Expected one feature row and one positive depth per unique ray.")
        if not torch.isfinite(parent_depth).all() or not (parent_depth > 0).all():
            raise ValueError("Parent depth must be finite and positive.")
        location = parent_depth.log() + self.mean_projection(features).squeeze(-1)
        factors = self.factor_projection(features)
        scale = self.minimum_scale + torch.nn.functional.softplus(self.scale_projection(features).squeeze(-1))
        return torch.distributions.LowRankMultivariateNormal(location, factors, scale.square())


def evaluate_program(log_depth_samples, program):
    """Evaluate a topologically ordered Boolean program on shared scene samples."""
    if log_depth_samples.ndim != 2 or not program:
        raise ValueError("Expected a sample-by-ray matrix and a nonempty program.")
    values = []
    for node in program:
        operation = node[0]
        if operation == "less":
            left, right = node[1:]
            if not all(isinstance(index, int) and 0 <= index < log_depth_samples.shape[1] for index in (left, right)):
                raise ValueError("Ray index outside the query group.")
            value = log_depth_samples[:, left] < log_depth_samples[:, right]
        elif operation == "above":
            index, threshold = node[1:]
            if not isinstance(index, int) or not 0 <= index < log_depth_samples.shape[1] or not math.isfinite(threshold) or threshold <= 0:
                raise ValueError("Expected a valid ray and positive finite depth threshold.")
            value = log_depth_samples[:, index] > math.log(threshold)
        else:
            references = node[1:]
            if not references or not all(isinstance(index, int) and 0 <= index < len(values) for index in references):
                raise ValueError("Program references must address earlier nodes.")
            if operation == "not" and len(references) == 1:
                value = ~values[references[0]]
            elif operation in {"and", "or", "xor"} and len(references) == 2:
                left, right = (values[index] for index in references)
                value = {"and": torch.logical_and, "or": torch.logical_or, "xor": torch.logical_xor}[operation](left, right)
            elif operation == "majority":
                value = torch.stack([values[index] for index in references], dim=-1).sum(-1) * 2 > len(references)
            else:
                raise ValueError("Unsupported program operation or arity.")
        values.append(value)
    return values[-1]


def sample_joint_scene(distribution, sample_count, generator):
    """Detach joint samples for a likelihood-score derivative; retain covariance."""
    if sample_count < 1:
        raise ValueError("Sample count must be positive.")
    if hasattr(distribution, "sample_detached"):
        return distribution.sample_detached(sample_count, generator)
    location = distribution.loc
    rank = distribution.cov_factor.shape[-1]
    shared = torch.randn((sample_count, rank), dtype=location.dtype, device=location.device, generator=generator)
    independent = torch.randn((sample_count, location.numel()), dtype=location.dtype, device=location.device, generator=generator)
    return (location + shared @ distribution.cov_factor.T + independent * distribution.cov_diag.sqrt()).detach()


def brier_gradient_from_batches(distribution, programs, targets, first_samples, second_samples):
    """The caller must provide independent batches; detach events and baseline."""
    targets = torch.as_tensor(targets, dtype=distribution.loc.dtype, device=distribution.loc.device)
    if targets.shape != (len(programs),) or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("Expected a binary target for every program.")
    first_events = torch.stack([evaluate_program(first_samples, program) for program in programs], dim=-1).to(targets.dtype)
    second_events = torch.stack([evaluate_program(second_samples, program) for program in programs], dim=-1).to(targets.dtype)
    first_probability = first_events.mean(0).detach()
    second_probability = second_events.mean(0).detach()
    log_density = distribution.log_prob(second_samples.detach())
    coefficients = 2 * (first_probability - targets) * (second_events - first_probability)
    surrogate = (coefficients.detach() * log_density[:, None]).mean()
    return {"gradient_surrogate": surrogate,
            "first_probability": first_probability,
            "second_probability": second_probability,
            "brier_product_estimate": ((first_probability - targets) * (second_probability - targets)).mean(),
            "brier_plugin_estimate": (((first_probability + second_probability) / 2 - targets).square()).mean()}


def estimate_brier_gradient(distribution, programs, targets, sample_count, generator):
    first_samples = sample_joint_scene(distribution, sample_count, generator)
    second_samples = sample_joint_scene(distribution, sample_count, generator)
    return brier_gradient_from_batches(distribution, programs, targets, first_samples, second_samples)
