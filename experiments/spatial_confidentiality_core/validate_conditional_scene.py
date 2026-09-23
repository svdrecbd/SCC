"""Independent dense controls for conditional scene probabilities and gradients."""

from fractions import Fraction
from pathlib import Path
import json
import math
import sys
import time
import torch
from conditional_scene_distribution import (
    adjusted_brier_reader, conditional_scene, disclosure_path,
    marginal_bin_log_probabilities,
)
from joint_scene_mixture import JointSceneMixture


def covariance(distribution):
    return distribution.factors @ distribution.factors.transpose(-1, -2) + torch.diag_embed(distribution.diagonal)


def validate_score_transfer():
    exact_checks = 0
    logarithmic_checks = 0
    minimum_slack = math.inf
    for numerator in range(1, 10):
        baseline = Fraction(numerator, 10)
        for forecast_numerator in range(11):
            forecast = Fraction(forecast_numerator, 10)
            direction = (forecast - baseline) / (baseline * (1 - baseline))
            assert abs(direction) <= 10
            for outcome in (0, 1):
                for step in (Fraction(0), Fraction(1, 100), Fraction(1, 10), Fraction(1), Fraction(10)):
                    adjusted = min(Fraction(1), max(Fraction(0), baseline + step * direction))
                    gain = (baseline - outcome) ** 2 - (adjusted - outcome) ** 2
                    lower = 2 * step * direction * (outcome - baseline) - step ** 2 * direction ** 2
                    assert gain >= lower
                    exact_checks += 1
                probability = forecast if outcome else 1 - forecast
                if probability:
                    log_gain = math.log(float(probability / (baseline if outcome else 1 - baseline)))
                    slack = float(direction * (outcome - baseline)) - log_gain
                    assert slack >= -1e-13
                    minimum_slack = min(minimum_slack, slack)
                    logarithmic_checks += 1
    truth = 0.1
    baseline = 0.01
    forecast = 0.25
    log_gain = truth * math.log(forecast / baseline) + (1-truth) * math.log((1-forecast)/(1-baseline))
    raw_gain = (baseline-truth)**2 - (forecast-truth)**2
    threshold = 0.05
    adjusted = float(adjusted_brier_reader(torch.tensor([baseline]), torch.tensor([forecast]), baseline, threshold)[0])
    recovered_gain = (baseline-truth)**2 - (adjusted-truth)**2
    lower = baseline**2 * threshold**2
    assert log_gain > threshold and raw_gain < 0 and recovered_gain >= lower
    invalid_baseline = math.exp(-100)
    rare_truth = 0.01
    invalid_log_gain = rare_truth * math.log(rare_truth/invalid_baseline) + (1-rare_truth)*math.log((1-rare_truth)/(1-invalid_baseline))
    bayes_brier_gain = (rare_truth-invalid_baseline)**2
    assert bayes_brier_gain < 0.1**2 * invalid_log_gain**2
    rejected = False
    try:
        adjusted_brier_reader(torch.tensor([invalid_baseline]), torch.tensor([rare_truth]), 0.1, 0.5)
    except ValueError:
        rejected = True
    assert rejected
    return {"rational_projection_checks": exact_checks, "logarithmic_tangent_checks": logarithmic_checks,
            "minimum_tangent_slack": minimum_slack, "positive_log_negative_brier_control": {
                "log_gain_nats": log_gain, "raw_brier_gain": raw_gain, "recovered_brier_gain": recovered_gain,
                "fixed_log_gain_threshold_nats": threshold, "guaranteed_brier_gain": lower},
            "invalid_floor_control": {"log_gain_nats": invalid_log_gain, "maximum_brier_gain": bayes_brier_gain,
                                      "incorrect_bound": 0.1**2 * invalid_log_gain**2, "rejected": rejected}}


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(configuration['seed'])
    records = []
    boundaries = torch.linspace(-3, 3, 63)
    for case in range(configuration['mixture_cases']):
        components, count, rank = (configuration[key] for key in ('components', 'coordinates', 'rank'))
        mixture = JointSceneMixture(torch.randn(components, generator=generator),
                                   torch.randn(components, count, generator=generator),
                                   torch.randn(components, count, rank, generator=generator)*0.4,
                                   torch.rand(components, count, generator=generator)+0.2)
        values = torch.randn(count, generator=generator)
        observed = torch.tensor([0, 3, 6])
        targets = torch.tensor([1, 2, 4, 5, 7])
        conditional, observation_density = conditional_scene(mixture, observed, values[observed], targets)
        full_covariance = covariance(mixture)
        observed_covariance = full_covariance[:, observed][:, :, observed]
        cross_covariance = full_covariance[:, targets][:, :, observed]
        dense_location = mixture.locations[:, targets] + (cross_covariance @ torch.linalg.solve(
            observed_covariance, (values[observed] - mixture.locations[:, observed])[..., None])).squeeze(-1)
        dense_covariance = full_covariance[:, targets][:, :, targets] - cross_covariance @ torch.linalg.solve(
            observed_covariance, cross_covariance.transpose(-1, -2))
        dense_observed = torch.distributions.MultivariateNormal(mixture.locations[:, observed], covariance_matrix=observed_covariance)
        dense_logits = mixture.logits.log_softmax(-1) + dense_observed.log_prob(values[observed])
        dense_observation_density = torch.logsumexp(dense_logits, -1)
        errors = {'conditional_mean': float((conditional.locations-dense_location).abs().max()),
                  'conditional_covariance': float((covariance(conditional)-dense_covariance).abs().max()),
                  'conditional_weights': float((conditional.logits.softmax(-1)-dense_logits.softmax(-1)).abs().max()),
                  'observation_log_density': float((observation_density-dense_observation_density).abs())}
        combined = torch.cat((observed, targets))
        dense_full = torch.distributions.MultivariateNormal(mixture.locations[:, combined], covariance_matrix=full_covariance[:, combined][:, :, combined])
        joint = torch.logsumexp(mixture.logits.log_softmax(-1) + dense_full.log_prob(values[combined]), -1)
        errors['density_chain'] = float((observation_density + conditional.log_prob(values[targets][None])[0]-joint).abs())
        reordered, reordered_density = conditional_scene(mixture, observed.flip(0), values[observed.flip(0)], targets)
        errors['observation_order'] = float((reordered.locations-conditional.locations).abs().max())
        errors['observation_order_density'] = float((reordered_density-observation_density).abs())
        empty, empty_density = conditional_scene(mixture, [], [], targets)
        assert torch.equal(empty.locations, mixture.locations[:, targets]) and empty_density == 0
        log_probabilities = marginal_bin_log_probabilities(conditional, boundaries)
        errors['bin_normalization'] = float(torch.logsumexp(log_probabilities, -1).abs().max())
        dense_standardized = (boundaries-dense_location[..., None])/dense_covariance.diagonal(dim1=-2, dim2=-1).sqrt()[..., None]
        dense_cdf = torch.distributions.Normal(0., 1.).cdf(dense_standardized)
        dense_masses = torch.diff(torch.cat((torch.zeros_like(dense_cdf[..., :1]), dense_cdf, torch.ones_like(dense_cdf[..., :1])), -1), dim=-1)
        dense_masses = (dense_masses*dense_logits.softmax(-1)[:, None, None]).sum(0)
        errors['dense_bin_probabilities'] = float((log_probabilities.exp()-dense_masses).abs().max())
        path_error = 0.
        for outcome in range(64):
            paths = disclosure_path(log_probabilities, torch.full((len(targets),), outcome))
            path_error = max(path_error, float((paths['selected_log_probabilities'].sum(-1)-log_probabilities[:, outcome]).abs().max()))
        errors['all_bin_paths'] = path_error
        assert max(errors.values()) < configuration['density_tolerance'], errors
        records.append(errors)
    rejections = 0
    for observed, targets, values in [([0, 0], [1], [0., 0.]), ([0], [0], [0.]), ([8], [1], [0.]), ([-1], [0], [0.])]:
        try:
            conditional_scene(mixture, observed, values, targets)
        except ValueError:
            rejections += 1
    assert rejections == 4
    # Very distant tail bins must remain finite in log coordinates and gradients.
    tail_location = torch.tensor([[30., -30.]], requires_grad=True)
    tail = JointSceneMixture(torch.zeros(1), tail_location, torch.zeros(1, 2, 1), torch.ones(1, 2)*0.1)
    tail_log_probabilities = marginal_bin_log_probabilities(tail, boundaries)
    assert torch.isfinite(tail_log_probabilities).all()
    tail_log_probabilities.sum().backward()
    assert torch.isfinite(tail_location.grad).all()
    # Small two-component case keeps every central-difference coordinate explicit.
    parameters = torch.randn(22, generator=generator)*0.2
    def objective(vector):
        distribution = JointSceneMixture(vector[:2], vector[2:8].reshape(2, 3),
                                         vector[8:14].reshape(2, 3, 1), vector[14:20].reshape(2, 3).exp())
        conditional, density = conditional_scene(distribution, [0], vector[20:21], [1, 2])
        probabilities = marginal_bin_log_probabilities(conditional, torch.linspace(-2., 2., 7)+vector[21])
        return -(probabilities[0, 2]+probabilities[1, 5])+0.1*density
    differentiable = parameters.clone().requires_grad_()
    gradient = torch.autograd.grad(objective(differentiable), differentiable)[0]
    reference = []
    step = configuration['finite_difference_step']
    for index in range(len(parameters)):
        displaced = torch.zeros_like(parameters); displaced[index] = step
        reference.append((objective(parameters+displaced)-objective(parameters-displaced))/(2*step))
    reference = torch.stack(reference)
    error = float((reference-gradient).abs().max())
    assert error < configuration['gradient_tolerance'], (reference, gradient)
    assert torch.isfinite(gradient).all() and (gradient.abs()>0).all()
    summary = {'status':'complete', 'case_errors':records, 'all_bin_paths_checked':64*configuration['mixture_cases']*5,
               'invalid_coordinate_rejections':rejections, 'extreme_tail_minimum_log_mass':float(tail_log_probabilities.min().detach()),
               'finite_difference_maximum_error':error, 'gradient_coordinates_checked':len(parameters),
               'score_transfer':validate_score_transfer(), 'neural_training':False, 'optimizer_steps':0,
               'wall_seconds':time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
