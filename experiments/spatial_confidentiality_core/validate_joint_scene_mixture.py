"""Validate higher-order scene mixtures and their density-score gradients."""

from collections import Counter
from itertools import combinations, product
from pathlib import Path
import json
import math
import platform
import sys
import time
import torch
from joint_scene_assessment import evaluate_program, estimate_brier_gradient, sample_joint_scene
from joint_scene_mixture import JointSceneMixture, JointSceneMixtureDistribution
from validate_joint_scene_assessment import state_digest


def mixture_parameters(parameters):
    return JointSceneMixture(parameters[12:14], parameters[:4].reshape(2, 2),
                             parameters[8:12].reshape(2, 2, 1), parameters[4:8].exp().reshape(2, 2))


def exact_mixture_comparison(parameters):
    locations = parameters[:4].reshape(2, 2)
    factors = parameters[8:12].reshape(2, 2)
    diagonal = parameters[4:8].exp().reshape(2, 2)
    variances = diagonal.sum(-1) + (factors[:, 0] - factors[:, 1]).square()
    probabilities = torch.distributions.Normal(0.0, 1.0).cdf((locations[:, 1] - locations[:, 0]) / variances.sqrt())
    return (parameters[12:14].softmax(-1) * probabilities).sum()


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(configuration["seed"])
    standard_deviation = configuration["component_standard_deviation"]
    centre_sets = [[centre for centre in product((-1.0, 1.0), repeat=3)
                    if sum(value > 0 for value in centre) % 2 == parity] for parity in (0, 1)]
    marginal_checks = 0
    for size in (1, 2):
        for coordinates in combinations(range(3), size):
            marginals = [Counter(tuple(centre[index] for index in coordinates) for centre in centres) for centres in centre_sets]
            assert marginals[0] == marginals[1]
            marginal_checks += 1
    program = [["above", 0, 1.0], ["above", 1, 1.0], ["above", 2, 1.0], ["xor", 0, 1], ["xor", 3, 2]]
    flip_probability = float(torch.distributions.Normal(0.0, 1.0).cdf(torch.tensor(-1 / standard_deviation)))
    even_risk = (1 - (1 - 2 * flip_probability) ** 3) / 2
    higher_order_records = []
    for parity, centres in enumerate(centre_sets):
        locations = torch.tensor(centres)
        covariance = locations.T @ locations / 4 + standard_deviation ** 2 * torch.eye(3)
        assert torch.equal(locations.mean(0), torch.zeros(3))
        assert torch.equal(covariance, (1 + standard_deviation ** 2) * torch.eye(3))
        mixture = JointSceneMixture(torch.zeros(4), locations, torch.zeros(4, 3, 1), torch.full((4, 3), standard_deviation ** 2))
        generator = torch.Generator().manual_seed(configuration["seed"] + parity)
        samples = sample_joint_scene(mixture, configuration["mixture_samples"], generator)
        outcomes = evaluate_program(samples, program)
        empirical_risk = float(outcomes.double().mean())
        expected_risk = even_risk if parity == 0 else 1 - even_risk
        standard_error = math.sqrt(expected_risk * (1 - expected_risk) / len(samples))
        tolerance = configuration["standard_error_multiplier"] * standard_error + configuration["probability_absolute_tolerance"]
        assert abs(empirical_risk - expected_risk) <= tolerance
        assert abs(0.5 - expected_risk) > 0.45
        higher_order_records.append({"mean_parity": parity, "component_locations": centres,
                                     "sample_count": len(samples), "positive_count": int(outcomes.sum()),
                                     "analytic_risk": expected_risk, "empirical_risk": empirical_risk,
                                     "binomial_standard_error": standard_error, "acceptance_tolerance": tolerance,
                                     "moment_matched_gaussian_risk": 0.5})
    parameters_initial = torch.tensor([-0.4, 0.3, 0.2, -0.1,
                                       math.log(0.3), math.log(0.2), math.log(0.15), math.log(0.35),
                                       0.2, -0.1, 0.1, 0.25, 0.4, -0.2])
    comparison = [["less", 0, 1]]
    gradient_records = []
    density_errors = []
    for target in (0, 1):
        reference_parameters = parameters_initial.clone().requires_grad_()
        probability = exact_mixture_comparison(reference_parameters)
        reference_gradient = torch.autograd.grad((probability - target).square(), reference_parameters)[0]
        gradients = []
        for repetition in range(configuration["gradient_repetitions"]):
            parameters = parameters_initial.clone().requires_grad_()
            mixture = mixture_parameters(parameters)
            generator = torch.Generator().manual_seed(configuration["seed"] + 1000 + 10000 * target + repetition)
            result = estimate_brier_gradient(mixture, [comparison], [target], configuration["samples_per_batch"], generator)
            gradients.append(torch.autograd.grad(result["gradient_surrogate"], parameters)[0])
            samples = sample_joint_scene(mixture, 32, generator)
            covariance = mixture.factors @ mixture.factors.transpose(-1, -2) + torch.diag_embed(mixture.diagonal)
            full_components = torch.distributions.MultivariateNormal(mixture.locations, covariance_matrix=covariance)
            full_density = torch.logsumexp(full_components.log_prob(samples[:, None, :]) + mixture.logits.log_softmax(-1), -1)
            density_errors.append(float((mixture.log_prob(samples) - full_density).abs().max().detach()))
        estimates = torch.stack(gradients)
        standard_error = estimates.std(0) / math.sqrt(len(estimates))
        tolerance = configuration["standard_error_multiplier"] * standard_error + configuration["gradient_absolute_tolerance"]
        error = (estimates.mean(0) - reference_gradient).abs()
        assert (error <= tolerance).all(), (target, error.tolist(), tolerance.tolist())
        assert reference_gradient[-2:].abs().min() > 0.01
        gradient_records.append({"target": target, "analytic_probability": float(probability.detach()),
                                 "reference_gradient": reference_gradient.tolist(), "estimated_gradient_mean": estimates.mean(0).tolist(),
                                 "empirical_standard_error": standard_error.tolist(), "absolute_error": error.tolist(),
                                 "acceptance_tolerance": tolerance.tolist(), "all_gradient_estimates": estimates.tolist()})
    assert max(density_errors) < configuration["log_density_tolerance"]
    module = JointSceneMixtureDistribution(seed=configuration["seed"]).double()
    before_digest = state_digest(module)
    features = torch.randn(6, 32, requires_grad=True)
    global_features = torch.randn(32, requires_grad=True)
    parent_depth = torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2, 1.3], requires_grad=True)
    mixture = module(features, parent_depth, global_features)
    subset = torch.tensor([1, 4, 2])
    restricted = module(features[subset], parent_depth[subset], global_features)
    assert torch.equal(mixture.logits, restricted.logits)
    for original, selected in [(mixture.locations[:, subset], restricted.locations),
                               (mixture.factors[:, subset], restricted.factors),
                               (mixture.diagonal[:, subset], restricted.diagonal)]:
        assert torch.allclose(original, selected, atol=1e-12, rtol=0)
    assert torch.equal(mixture.locations, parent_depth.log()[None, :].expand_as(mixture.locations))
    initial_median_error = float((mixture.loc.exp() - parent_depth).abs().max().detach())
    assert initial_median_error < 1e-12
    generator = torch.Generator().manual_seed(configuration["seed"] + 30000)
    result = estimate_brier_gradient(mixture, [comparison, [["above", 2, 1.0]]], [0, 0], 4096, generator)
    result["gradient_surrogate"].backward()
    gradient_norms = {name: float(parameter.grad.norm()) for name, parameter in module.named_parameters()}
    gradient_norms.update({"parent_depth": float(parent_depth.grad.norm()), "features": float(features.grad.norm()),
                           "global_features": float(global_features.grad.norm())})
    assert all(math.isfinite(value) and value > 0 for value in gradient_norms.values()), gradient_norms
    assert before_digest == state_digest(module)
    summary = {"status": "complete", "identical_low_order_marginals_checked": marginal_checks,
               "higher_order_controls": higher_order_records, "maximum_log_density_disagreement": max(density_errors),
               "gradient_comparisons": [{key: value for key, value in record.items() if key != "all_gradient_estimates"} for record in gradient_records],
               "projective_subset_indices": subset.tolist(), "module_gradient_norms": gradient_norms,
               "initial_median_error_metres": initial_median_error,
               "module_parameters": sum(parameter.numel() for parameter in module.parameters()),
               "module_state_sha256_before_and_after": before_digest, "neural_training": False, "optimizer_steps": 0,
               "machine": {"platform": platform.platform(), "python": sys.version, "torch": torch.__version__},
               "wall_seconds": time.perf_counter() - started}
    (directory / "gradient_estimates.json").write_text(json.dumps(gradient_records, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
