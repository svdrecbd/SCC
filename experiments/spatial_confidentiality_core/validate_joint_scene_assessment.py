"""Check joint policy semantics and likelihood-score derivatives without training."""

from itertools import product
from pathlib import Path
import hashlib
import json
import math
import platform
import sys
import time
import torch

from joint_scene_assessment import (
    JointSceneDistribution, brier_gradient_from_batches, estimate_brier_gradient,
    evaluate_program, sample_joint_scene,
)


def scalar_program(world, program):
    values = []
    for instruction in program:
        operation = instruction[0]
        if operation == "less":
            value = world[instruction[1]] < world[instruction[2]]
        elif operation == "above":
            value = math.exp(world[instruction[1]]) > instruction[2]
        elif operation == "not":
            value = not values[instruction[1]]
        elif operation == "and":
            value = values[instruction[1]] and values[instruction[2]]
        elif operation == "or":
            value = values[instruction[1]] or values[instruction[2]]
        elif operation == "xor":
            value = values[instruction[1]] != values[instruction[2]]
        elif operation == "majority":
            value = sum(values[index] for index in instruction[1:]) > (len(instruction) - 1) / 2
        else:
            raise ValueError(operation)
        values.append(value)
    return values[-1]


def parameter_distribution(parameters):
    return torch.distributions.LowRankMultivariateNormal(
        parameters[:2], parameters[4:].reshape(2, 1), parameters[2:4].exp())


def analytic_probability(parameters):
    variance = parameters[2:4].exp().sum() + (parameters[4] - parameters[5]).square()
    standardized = (parameters[1] - parameters[0]) / variance.sqrt()
    return torch.distributions.Normal(0.0, 1.0).cdf(standardized)


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(configuration["seed"])
    programs = [
        [["less", 0, 1]],
        [["above", 2, 1.0]],
        [["less", 0, 1], ["not", 0]],
        [["less", 0, 1], ["xor", 0, 0]],
        [["less", 0, 1], ["and", 0, 0]],
        [["less", 0, 1], ["above", 2, 1.0], ["or", 0, 1]],
        [["less", 0, 1], ["less", 2, 3], ["xor", 0, 1]],
        [["above", 0, 1.0], ["above", 1, 1.0], ["above", 2, 1.0], ["majority", 0, 1, 2]],
    ]
    worlds = list(product((-1.0, 1.0), repeat=4))
    world_tensor = torch.tensor(worlds)
    semantics_checks = 0
    for program in programs:
        actual = evaluate_program(world_tensor, program).tolist()
        expected = [scalar_program(world, program) for world in worlds]
        assert actual == expected
        semantics_checks += len(worlds)
    first = evaluate_program(world_tensor, programs[0])
    assert torch.equal(~first, evaluate_program(world_tensor, programs[2]))
    assert not evaluate_program(world_tensor, programs[3]).any()
    assert torch.equal(first, evaluate_program(world_tensor, programs[4]))
    repeated_above = [["above", 0, 1.0], ["xor", 0, 0]]
    assert evaluate_program(world_tensor, repeated_above).double().mean() == 0
    marginal_probability = float((world_tensor[:, 0] > 0).double().mean())
    independent_duplicate_probability = 2 * marginal_probability * (1 - marginal_probability)
    assert independent_duplicate_probability != float(evaluate_program(world_tensor, repeated_above).double().mean())
    correlated = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    anticorrelated = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
    joint_program = [["above", 0, 1.0], ["above", 1, 1.0], ["xor", 0, 1]]
    assert torch.equal((correlated > 0).double().mean(0), (anticorrelated > 0).double().mean(0))
    assert evaluate_program(correlated, joint_program).double().mean() == 0
    assert evaluate_program(anticorrelated, joint_program).double().mean() == 1
    malformed_rejections = 0
    for invalid in ([["xor", 0, 0]], [["above", 0, 0]], [["less", 0, 4]], [["less", 0, 1], ["not", 2]]):
        try:
            evaluate_program(world_tensor, invalid)
        except ValueError:
            malformed_rejections += 1
        else:
            raise AssertionError("Malformed program was accepted")
    base_parameters = torch.tensor([-0.3, 0.2, math.log(0.36), math.log(0.25), 0.2, -0.1])
    gradient_records = []
    density_errors = []
    probability_records = []
    for target in (0, 1):
        reference_parameters = base_parameters.clone().requires_grad_()
        probability = analytic_probability(reference_parameters)
        reference_loss = (probability - target).square()
        reference_gradient = torch.autograd.grad(reference_loss, reference_parameters)[0]
        gradients = []
        estimates = []
        for repetition in range(configuration["gradient_repetitions"]):
            parameters = base_parameters.clone().requires_grad_()
            distribution = parameter_distribution(parameters)
            generator = torch.Generator().manual_seed(configuration["seed"] + 10000 * target + repetition)
            result = estimate_brier_gradient(distribution, [programs[0]], [target], configuration["samples_per_batch"], generator)
            derivative = torch.autograd.grad(result["gradient_surrogate"], parameters)[0]
            gradients.append(derivative)
            estimates.append(float(result["brier_product_estimate"]))
            covariance = distribution.cov_factor @ distribution.cov_factor.T + torch.diag(distribution.cov_diag)
            independent_density = torch.distributions.MultivariateNormal(distribution.loc, covariance_matrix=covariance)
            samples = sample_joint_scene(distribution, 32, generator)
            density_errors.append(float((distribution.log_prob(samples) - independent_density.log_prob(samples)).abs().max().detach()))
        gradient_tensor = torch.stack(gradients)
        mean = gradient_tensor.mean(0)
        standard_error = gradient_tensor.std(0) / math.sqrt(len(gradients))
        tolerance = configuration["gradient_standard_error_multiplier"] * standard_error + configuration["gradient_absolute_tolerance"]
        error = (mean - reference_gradient).abs()
        assert (error <= tolerance).all(), (target, error.tolist(), tolerance.tolist())
        gradient_records.append({"target": target, "reference_gradient": reference_gradient.tolist(),
                                 "estimated_gradient_mean": mean.tolist(), "empirical_standard_error": standard_error.tolist(),
                                 "absolute_error": error.tolist(), "acceptance_tolerance": tolerance.tolist(),
                                 "all_gradient_estimates": gradient_tensor.tolist()})
        probability_records.append({"target": target, "analytic_probability": float(probability.detach()),
                                    "population_brier": float(reference_loss.detach()),
                                    "mean_independent_product_estimate": sum(estimates) / len(estimates),
                                    "all_independent_product_estimates": estimates})
    assert max(density_errors) < configuration["log_density_tolerance"]
    parameters = base_parameters.clone().requires_grad_()
    distribution = parameter_distribution(parameters)
    generator = torch.Generator().manual_seed(configuration["seed"] + 20000)
    one_sample = sample_joint_scene(distribution, 1, generator)
    reused = brier_gradient_from_batches(distribution, [programs[0]], [0], one_sample, one_sample)
    incorrect_gradient = torch.autograd.grad(reused["gradient_surrogate"], parameters)[0]
    assert torch.count_nonzero(incorrect_gradient) == 0
    assert torch.linalg.vector_norm(torch.tensor(gradient_records[0]["reference_gradient"])) > 0.1
    assert not evaluate_program(one_sample, programs[0]).double().mean().requires_grad

    module = JointSceneDistribution(seed=configuration["seed"]).double()
    before_digest = state_digest(module)
    features = torch.randn(6, 32, requires_grad=True)
    parent_depth = torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2, 1.3], requires_grad=True)
    distribution = module(features, parent_depth)
    median_error = float((distribution.loc.exp() - parent_depth).abs().max().detach())
    assert median_error < 1e-12
    result = estimate_brier_gradient(distribution, [programs[0], programs[1]], [0, 0], 4096, generator)
    result["gradient_surrogate"].backward()
    gradient_norms = {name: float(parameter.grad.norm()) for name, parameter in module.named_parameters()}
    gradient_norms.update({"parent_depth": float(parent_depth.grad.norm()), "features": float(features.grad.norm())})
    assert all(math.isfinite(value) and value > 0 for value in gradient_norms.values()), gradient_norms
    assert before_digest == state_digest(module)
    summary = {"status": "complete", "program_world_checks": semantics_checks,
               "malformed_program_rejections": malformed_rejections,
               "same_marginals_xor_probabilities": [0, 1],
               "incorrect_independent_duplicate_xor_probability": independent_duplicate_probability,
               "correct_duplicate_xor_probability": 0,
               "maximum_log_density_disagreement": max(density_errors),
               "gradient_comparisons": [{key: value for key, value in record.items() if key != "all_gradient_estimates"} for record in gradient_records],
               "probability_comparisons": [{key: value for key, value in record.items() if key != "all_independent_product_estimates"} for record in probability_records],
               "reused_single_draw_gradient": incorrect_gradient.tolist(),
               "module_gradient_norms": gradient_norms, "module_parameters": sum(parameter.numel() for parameter in module.parameters()),
               "initial_depth_median_error": median_error, "module_state_sha256_before_and_after": before_digest,
               "neural_training": False, "optimizer_steps": 0,
               "machine": {"platform": platform.platform(), "python": sys.version, "torch": torch.__version__},
               "wall_seconds": time.perf_counter() - started}
    (directory / "gradient_estimates.json").write_text(json.dumps(gradient_records, indent=2) + "\n")
    (directory / "loss_estimates.json").write_text(json.dumps(probability_records, indent=2) + "\n")
    (directory / "programs.json").write_text(json.dumps(programs, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
