"""Rank-sized filtering for scene forecasts with exact observation histories."""

import math
import torch
from conditional_scene_distribution import log_difference


def sequential_marginals(distribution, observations):
    """Score each coordinate before using that coordinate as an observation."""
    components, coordinates, rank = distribution.factors.shape
    if observations.shape != (coordinates,) or not torch.isfinite(observations).all():
        raise ValueError("Expected one finite observation per coordinate.")
    latent_mean = distribution.locations.new_zeros((components, rank))
    latent_covariance = torch.eye(rank, dtype=latent_mean.dtype, device=latent_mean.device).expand(components, rank, rank).clone()
    logits = distribution.logits.log_softmax(-1)
    means, variances, weights, densities = [], [], [], []
    for index in range(coordinates):
        factor = distribution.factors[:, index]
        mean = distribution.locations[:, index] + (factor*latent_mean).sum(-1)
        direction = (latent_covariance @ factor[..., None]).squeeze(-1)
        variance = distribution.diagonal[:, index] + (factor*direction).sum(-1)
        if not (variance > 0).all():
            raise ArithmeticError("Conditional variance must remain positive.")
        means.append(mean); variances.append(variance); weights.append(logits)
        residual = observations[index]-mean
        component_density = -0.5*(math.log(2*math.pi)+variance.log()+residual.square()/variance)
        densities.append(torch.logsumexp(logits+component_density, -1))
        # The target enters state only after its forecast and score are formed.
        latent_mean = latent_mean + direction*(residual/variance)[:, None]
        latent_covariance = latent_covariance - direction[..., None]*direction[:, None]/variance[:, None, None]
        logits = (logits+component_density).log_softmax(-1)
    return {'locations':torch.stack(means), 'variances':torch.stack(variances),
            'log_weights':torch.stack(weights), 'log_densities':torch.stack(densities)}


def sequential_bin_log_probabilities(marginals, boundaries):
    standardized = (boundaries-marginals['locations'][..., None])/marginals['variances'].sqrt()[..., None]
    lower = torch.special.log_ndtr(standardized)
    upper = torch.special.log_ndtr(-standardized)
    positive = standardized[..., :-1] >= 0
    larger = torch.where(positive, upper[..., :-1], lower[..., 1:])
    smaller = torch.where(positive, upper[..., 1:], lower[..., :-1])
    component_mass = torch.cat((lower[..., :1], log_difference(larger, smaller), upper[..., -1:]), -1)
    return torch.logsumexp(component_mass+marginals['log_weights'][..., None], 1)


def clipped_depth_mean(marginals, minimum=0.1, maximum=10.):
    means, variances = marginals['locations'], marginals['variances']
    standard_deviations = variances.sqrt()
    lower = (math.log(minimum)-means)/standard_deviations
    upper = (math.log(maximum)-means)/standard_deviations
    normal = torch.distributions.Normal(0., 1.)
    interval = normal.cdf(upper-standard_deviations)-normal.cdf(lower-standard_deviations)
    expectation = minimum*normal.cdf(lower) + (means+variances/2).exp()*interval + maximum*normal.cdf(-upper)
    return (marginals['log_weights'].exp()*expectation).sum(-1)
