"""Exact continuous conditioning and interval risks for Gaussian scene mixtures."""

import math
import torch
from joint_scene_mixture import JointSceneMixture


def conditional_scene(distribution, observed_indices, observed_values, target_indices):
    """Return a target law and log observation density; measurements are exact."""
    location = distribution.locations
    observed = torch.as_tensor(observed_indices, dtype=torch.long, device=location.device)
    targets = torch.as_tensor(target_indices, dtype=torch.long, device=location.device)
    values = torch.as_tensor(observed_values, dtype=location.dtype, device=location.device)
    if observed.ndim != 1 or targets.ndim != 1 or values.shape != observed.shape or not len(targets):
        raise ValueError("Expected observation and target vectors and matching measurements.")
    combined = torch.cat((observed, targets))
    if len(combined.unique()) != len(combined) or (combined < 0).any() or (combined >= location.shape[1]).any():
        raise ValueError("Coordinates must be unique, disjoint and in range.")
    if not torch.isfinite(values).all():
        raise ValueError("Measurements must be finite.")
    if not len(observed):
        return JointSceneMixture(distribution.logits, location[:, targets],
                                 distribution.factors[:, targets], distribution.diagonal[:, targets]), location.new_zeros(())
    factors = distribution.factors[:, observed]
    diagonal = distribution.diagonal[:, observed]
    residual = values - location[:, observed]
    precision = torch.eye(factors.shape[-1], dtype=location.dtype, device=location.device)
    precision = precision + factors.transpose(-1, -2) @ (factors / diagonal[..., None])
    cholesky = torch.linalg.cholesky(precision)
    information = (factors * (residual / diagonal)[..., None]).sum(-2)
    posterior_mean = torch.cholesky_solve(information[..., None], cholesky).squeeze(-1)
    target_factors = distribution.factors[:, targets]
    updated_location = location[:, targets] + (target_factors * posterior_mean[:, None]).sum(-1)
    updated_factors = torch.linalg.solve_triangular(cholesky, target_factors.transpose(-1, -2), upper=False).transpose(-1, -2)
    determinant = diagonal.log().sum(-1) + 2 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    quadratic = (residual.square() / diagonal).sum(-1) - (information * posterior_mean).sum(-1)
    component_density = -0.5 * (len(observed) * math.log(2 * math.pi) + determinant + quadratic)
    observation_density = torch.logsumexp(distribution.logits.log_softmax(-1) + component_density, -1)
    return JointSceneMixture(distribution.logits + component_density, updated_location,
                             updated_factors, distribution.diagonal[:, targets]), observation_density


def log_difference(larger, smaller):
    """Subtract positive values in log coordinates without tail cancellation."""
    return larger + torch.log(-torch.expm1(smaller - larger))


def marginal_bin_log_probabilities(distribution, finite_boundaries):
    """Include both tails; use survival functions for positive-tail intervals."""
    boundaries = torch.as_tensor(finite_boundaries, dtype=distribution.locations.dtype,
                                 device=distribution.locations.device)
    if boundaries.ndim != 1 or not len(boundaries) or not torch.isfinite(boundaries).all() or not (boundaries[1:] > boundaries[:-1]).all():
        raise ValueError("Finite bin boundaries must be strictly increasing.")
    variance = distribution.diagonal + distribution.factors.square().sum(-1)
    standardized = (boundaries - distribution.locations[..., None]) / variance.sqrt()[..., None]
    lower_logarithms = torch.special.log_ndtr(standardized)
    upper_logarithms = torch.special.log_ndtr(-standardized)
    positive = standardized[..., :-1] >= 0
    larger = torch.where(positive, upper_logarithms[..., :-1], lower_logarithms[..., 1:])
    smaller = torch.where(positive, upper_logarithms[..., 1:], lower_logarithms[..., :-1])
    middle = log_difference(larger, smaller)
    component_probabilities = torch.cat((lower_logarithms[..., :1], middle, upper_logarithms[..., -1:]), -1)
    return torch.logsumexp(distribution.logits.log_softmax(-1)[:, None, None] + component_probabilities, 0)


def disclosure_path(log_probabilities, outcomes):
    """Return normalized selected-branch log risks for a balanced bin tree.

    At each depth, the prior branch choices are public interval information.
    This does not condition other rays on quantized observations.
    """
    bins = log_probabilities.shape[-1]
    if bins < 2 or bins & (bins - 1):
        raise ValueError("The bin count must be a positive power of two above one.")
    outcomes = torch.as_tensor(outcomes, dtype=torch.long, device=log_probabilities.device)
    if log_probabilities.ndim != 2 or outcomes.shape != log_probabilities.shape[:1] or (outcomes < 0).any() or (outcomes >= bins).any():
        raise ValueError("Expected one valid bin index per target ray.")
    selected_branches = []
    upper_probabilities = []
    upper_outcomes = []
    for level in range(bins.bit_length() - 1):
        width = bins >> level
        grouped = log_probabilities.reshape(len(outcomes), bins // width, width)
        interval = outcomes // width
        rows = torch.arange(len(outcomes), device=outcomes.device)
        intervals = grouped[rows, interval]
        lower_mass = torch.logsumexp(intervals[:, :width // 2], -1)
        upper_mass = torch.logsumexp(intervals[:, width // 2:], -1)
        interval_mass = torch.logaddexp(lower_mass, upper_mass)
        upper = outcomes.remainder(width) >= width // 2
        selected_branches.append(torch.where(upper, upper_mass, lower_mass) - interval_mass)
        upper_probabilities.append((upper_mass - interval_mass).exp())
        upper_outcomes.append(upper)
    return {"selected_log_probabilities": torch.stack(selected_branches, -1),
            "upper_probabilities": torch.stack(upper_probabilities, -1),
            "upper_outcomes": torch.stack(upper_outcomes, -1)}


def adjusted_brier_reader(baseline, forecast, minimum_probability, useful_gain_threshold):
    """A label-free reader for a declared positive log-score threshold in nats."""
    if not 0 < minimum_probability <= 0.5 or useful_gain_threshold <= 0:
        raise ValueError("A positive probability floor and useful-gain threshold are required.")
    if not ((baseline >= minimum_probability) & (baseline <= 1 - minimum_probability)).all():
        raise ValueError("The baseline violates its declared probability floor.")
    direction = (forecast - baseline) / (baseline * (1 - baseline))
    step = minimum_probability ** 2 * useful_gain_threshold
    return (baseline + step * direction).clamp(0, 1)
