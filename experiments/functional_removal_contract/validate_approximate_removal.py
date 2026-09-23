"""Check a sharp binary removal frontier by exact joint-law calculations."""

from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import sys
import time


def evaluate_channel(error_probability, output_one_probabilities):
    useful_accuracy = Fraction(0)
    protected_joint = [[Fraction(0) for protected in range(2)] for output in range(2)]
    records = []
    for index, (useful, error) in enumerate(product(range(2), repeat=2)):
        source_mass = (error_probability if error else 1 - error_probability) / 2
        protected = useful ^ error
        for output in range(2):
            conditional = output_one_probabilities[index] if output else 1 - output_one_probabilities[index]
            mass = source_mass * conditional
            useful_accuracy += mass * (output == useful)
            protected_joint[output][protected] += mass
            records.append({"useful": useful, "error": error, "protected": protected,
                            "output": output, "joint_probability": str(mass)})
    assert sum(sum(row) for row in protected_joint) == 1
    protected_accuracy = sum(max(row) for row in protected_joint)
    return useful_accuracy, protected_accuracy, records


def attaining_channel(error_probability, useful_accuracy):
    threshold = Fraction(1, 2) + error_probability
    probabilities = []
    for useful, error in product(range(2), repeat=2):
        if useful_accuracy >= threshold:
            flip_probability = Fraction(0) if error else (1 - useful_accuracy) / (1 - error_probability)
            correct_probability = 1 - flip_probability
        else:
            mixture_weight = (useful_accuracy - Fraction(1, 2)) / error_probability
            boundary_flip = Fraction(0) if error else (Fraction(1, 2) - error_probability) / (1 - error_probability)
            correct_probability = mixture_weight * (1 - boundary_flip) + (1 - mixture_weight) / 2
        probabilities.append(correct_probability if useful else 1 - correct_probability)
    assert all(0 <= probability <= 1 for probability in probabilities)
    return probabilities


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    errors = [Fraction(value) for value in configuration["error_probabilities"]]
    requirements = [Fraction(value) for value in configuration["useful_accuracy_requirements"]]
    grid = [Fraction(value) for value in configuration["channel_probabilities"]]
    attaining_records = []
    grid_checks = 0
    incorrect_zero_overlap_bounds_rejected = 0
    for error in errors:
        for required in requirements:
            channel = attaining_channel(error, required)
            useful_accuracy, protected_accuracy, joint_law = evaluate_channel(error, channel)
            optimum = max(Fraction(1, 2), required - error)
            assert useful_accuracy == required
            assert protected_accuracy == optimum
            # Retaining H itself permits an exact reader regardless of B.
            advice_accuracy = sum(Fraction(row["joint_probability"]) for row in joint_law)
            assert advice_accuracy == 1
            attaining_records.append({"error_probability": str(error),
                                       "useful_accuracy": str(useful_accuracy),
                                       "protected_bayes_accuracy": str(protected_accuracy),
                                       "channel_output_one_probabilities": [str(value) for value in channel],
                                       "protected_accuracy_with_advice": str(advice_accuracy),
                                       "joint_law": joint_law})
        for channel in product(grid, repeat=4):
            useful_accuracy, protected_accuracy, _ = evaluate_channel(error, channel)
            assert protected_accuracy >= max(Fraction(1, 2), useful_accuracy - error)
            incorrect_zero_overlap_bounds_rejected += protected_accuracy < useful_accuracy
            grid_checks += 1
    assert grid_checks == len(errors) * len(grid) ** 4 == 2500
    assert incorrect_zero_overlap_bounds_rejected > 0
    summary = {"status": "complete", "attaining_channels": len(attaining_records),
               "grid_channels_checked": grid_checks,
               "incorrect_zero_overlap_bounds_rejected": incorrect_zero_overlap_bounds_rejected,
               "maximum_useful_accuracy_at_complete_erasure": [
                   {"error_probability": str(error), "useful_accuracy": str(Fraction(1, 2) + error)}
                   for error in errors],
               "neural_training": False, "general_scc_impossibility_established": False,
               "wall_seconds": time.perf_counter() - started}
    (directory / "attaining_channels.json").write_text(json.dumps(attaining_records, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
