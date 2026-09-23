"""Validate probability integration, derivatives, and shared-ray rejection."""

from itertools import product
from pathlib import Path
import json
import math
import sys
import time
import torch
from conditional_program_assessment import independent_program_probability, conditional_program_probability, integrated_brier_objective
from joint_scene_assessment import sample_joint_scene
from joint_scene_mixture import JointSceneMixture


def enumerate_probability(probabilities, operation):
    result = probabilities.new_zeros(())
    for values in product((0, 1), repeat=len(probabilities)):
        event = sum(values) % 2 if operation == "xor" else sum(values) * 2 > len(values)
        weight = probabilities.new_ones(())
        for index, value in enumerate(values):
            weight = weight * (probabilities[index] if value else 1 - probabilities[index])
        result = result + event * weight
    return result


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(configuration["seed"])
    enumeration_checks = 0
    for count, operation in product(range(1, 9), ("xor", "majority")):
        probabilities = torch.linspace(.13, .89, count, requires_grad=True)
        actual = independent_program_probability(probabilities, operation)
        expected = enumerate_probability(probabilities, operation)
        assert torch.allclose(actual, expected, atol=1e-13, rtol=1e-12)
        actual_gradient = torch.autograd.grad(actual, probabilities)[0]
        expected_gradient = torch.autograd.grad(expected, probabilities)[0]
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-13, rtol=1e-12)
        enumeration_checks += 1
    independent_records = []
    for count in (4, 32, 64):
        location_parameter = torch.tensor(0., requires_grad=True)
        offset = math.sqrt(2) * torch.special.ndtri(torch.tensor(.9))
        locations = torch.stack([location_parameter, offset]).repeat(count)
        distribution = torch.distributions.LowRankMultivariateNormal(locations, torch.zeros(2*count, 1), torch.ones(2*count))
        pairs = [[2*index, 2*index+1] for index in range(count)]
        first = torch.randn((4, 1, 1), generator=generator)
        second = torch.randn((4, 1, 1), generator=generator)
        estimate = integrated_brier_objective(distribution, pairs, "xor", 0, first, second)
        gradient = torch.autograd.grad(estimate["objective"], location_parameter)[0]
        probability = (1 - (-.8)**count)/2
        probability_derivative = -count * (-.8)**(count-1) * math.exp(-float(offset)**2/4) / math.sqrt(4*math.pi)
        expected_gradient = 2 * probability * probability_derivative
        assert abs(float(estimate["first_probability"].detach())-probability) < 1e-13
        assert abs(float(gradient)-expected_gradient) < max(1e-13, abs(expected_gradient)*1e-10)
        assert torch.equal(estimate["first_probability"], estimate["second_probability"])
        independent_records.append({"comparisons": count, "risk": probability,
                                    "actual_derivative": float(gradient), "analytic_derivative": expected_gradient,
                                    "sample_variance": 0})
    components, rays, rank = 3, 8, 2
    logits = torch.tensor([-.4, .2, .7], requires_grad=True)
    locations = (torch.randn((components, rays), generator=generator)*.4).requires_grad_()
    factors = (torch.randn((components, rays, rank), generator=generator)*.5).requires_grad_()
    log_diagonal = torch.full((components, rays), -.7, requires_grad=True)
    distribution = JointSceneMixture(logits, locations, factors, log_diagonal.exp())
    pairs = [[index, index+1] for index in range(0, rays, 2)]
    normals = torch.randn((configuration["conditional_samples"], components, rank), generator=generator)
    samples = sample_joint_scene(distribution, configuration["probability_samples"], generator)
    comparisons = samples[:, ::2] < samples[:, 1::2]
    comparison_records = []
    for operation in ("xor", "majority"):
        integrated = conditional_program_probability(distribution, pairs, operation, normals).detach()
        events = ((comparisons.sum(1) % 2).double() if operation == "xor"
                  else (comparisons.sum(1)*2 > len(pairs)).double())
        difference = abs(float(integrated.mean()-events.mean()))
        standard_error = float(integrated.var()/len(integrated)+events.var()/len(events))**.5
        assert difference < 6*standard_error+.001
        comparison_records.append({"operation": operation, "integrated_risk": float(integrated.mean()),
                                   "sampled_risk": float(events.mean()), "difference": difference,
                                   "comparison_standard_error": standard_error,
                                   "integrated_sample_variance": float(integrated.var()),
                                   "hard_event_sample_variance": float(events.var())})
    fixed_normals = normals[:3]
    def derivative_function(current_logits, current_locations, current_factors, current_log_diagonal):
        current = JointSceneMixture(current_logits, current_locations, current_factors, current_log_diagonal.exp())
        return torch.stack([conditional_program_probability(current, pairs, operation, fixed_normals).mean()
                            for operation in ("xor", "majority")])
    assert torch.autograd.gradcheck(derivative_function, (logits, locations, factors, log_diagonal), eps=1e-6, atol=1e-6, rtol=1e-4)
    rejections = 0
    for invalid in ([[0, 1], [1, 2]], [[0, 0]], [[0, 9]], []):
        try:
            conditional_program_probability(distribution, invalid, "xor", fixed_normals)
        except ValueError:
            rejections += 1
    assert rejections == 4
    result = {"status": "complete", "probability_and_derivative_enumerations": enumeration_checks,
              "independent_controls": independent_records, "mixture_probability_checks": comparison_records,
              "all_parameter_finite_difference_check": True, "invalid_ray_groups_rejected": rejections,
              "neural_training": False, "wall_seconds": time.perf_counter()-started}
    (directory / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
