"""Exact tests of the public approximation obstruction in LN-368."""

from fractions import Fraction
from pathlib import Path
import hashlib
import json
import platform
import resource
import sys
import time

from validate_optimization_threshold import feasible, heuristic_plan, objective, rational_record


def half_approximation(problem):
    density_plan = heuristic_plan(problem, "value_density")
    eligible = [index for index, weight in enumerate(problem["weights"])
                if weight <= problem["capacity"]]
    best_index = max(eligible, key=lambda index: problem["values"][index])
    singleton = tuple(int(index == best_index) for index in range(len(problem["weights"])))
    return max((density_plan, singleton), key=lambda candidate: objective(problem, candidate))


def rounded_value_plan(problem, epsilon):
    eligible = [index for index, weight in enumerate(problem["weights"])
                if weight <= problem["capacity"]]
    maximum_value = max(problem["values"][index] for index in eligible)
    unit = epsilon * maximum_value / len(eligible)
    # Each state stores a minimum-weight subset for one rounded total value.
    states = {0: (0, ())}
    update_count = 0
    for index in eligible:
        rounded_value = int(Fraction(problem["values"][index]) // unit)
        updated = states.copy()
        for total_value, (weight, selection) in states.items():
            update_count += 1
            combined_weight = weight + problem["weights"][index]
            combined_value = total_value + rounded_value
            if combined_weight <= problem["capacity"] and (
                combined_value not in updated or combined_weight < updated[combined_value][0]
            ):
                updated[combined_value] = (combined_weight, selection + (index,))
        states = updated
    selected = states[max(states)][1]
    plan = tuple(int(index in selected) for index in range(len(problem["weights"])))
    return plan, {"eligible_items": len(eligible), "unit": rational_record(unit),
                  "retained_states": len(states), "updates": update_count}


def integrated_scalar_accuracy(optimum, cutoff):
    upper = 2 * optimum
    points = sorted({Fraction(0), Fraction(optimum), Fraction(upper)} |
                    ({cutoff} if 0 < cutoff < upper else set()))
    accuracy = Fraction(0)
    for lower, higher in zip(points, points[1:]):
        threshold = (lower + higher) / 2
        true_unsafe = threshold < optimum
        predicted_unsafe = threshold <= cutoff
        if true_unsafe == predicted_unsafe:
            accuracy += (higher - lower) / upper
    return accuracy


def assess_reader(optimum, value, alpha):
    assert alpha * optimum <= value <= optimum
    cutoff = 2 * value / (1 + alpha)
    accuracy = integrated_scalar_accuracy(optimum, cutoff)
    expected = 1 - min(abs(cutoff - optimum), optimum) / (2 * optimum)
    bound = 1 - (1 - alpha) / (2 * (1 + alpha))
    assert accuracy == expected
    assert accuracy >= bound
    return {"alpha": rational_record(alpha), "cutoff": rational_record(cutoff),
            "accuracy": rational_record(accuracy), "accuracy_lower_bound": rational_record(bound)}


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / "config.json").read_text())
    resource.setrlimit(resource.RLIMIT_FSIZE,
                       (configuration["maximum_output_bytes"], configuration["maximum_output_bytes"]))
    source_data = Path(configuration["source_results"]).read_bytes()
    source_digest = hashlib.sha256(source_data).hexdigest()
    assert source_digest == configuration["source_results_sha256"]
    records = json.loads(source_data)["problems"]
    results = []
    for record in records:
        problem, optimum = record["problem"], Fraction(record["optimum"])
        methods = {"half_approximation": (half_approximation(problem), Fraction(1, 2), {})}
        for denominator in configuration["epsilon_denominators"]:
            epsilon = Fraction(1, denominator)
            plan, accounting = rounded_value_plan(problem, epsilon)
            methods["rounded_value_" + str(denominator)] = (plan, 1 - epsilon, accounting)
        for method, (plan, alpha, accounting) in methods.items():
            assert feasible(problem, plan)
            value = Fraction(objective(problem, plan))
            reader = assess_reader(optimum, value, alpha)
            results.append({"problem_index": problem["index"], "method": method,
                            "candidate": list(plan), "quality": rational_record(value / optimum),
                            "reader": reader, "accounting": accounting})
    denominator = configuration["ratio_grid_denominator"]
    grid_checks = 0
    for alpha in (Fraction(1, 100), Fraction(1, 2), Fraction(3, 4),
                  Fraction(7, 8), Fraction(15, 16), Fraction(1)):
        for index in range(denominator + 1):
            value = alpha + (1 - alpha) * Fraction(index, denominator)
            assess_reader(Fraction(1), value, alpha)
            grid_checks += 1
    averages = {}
    for method in sorted({record["method"] for record in results}):
        selected = [record for record in results if record["method"] == method]
        averages[method] = {}
        for metric in ("quality", "accuracy"):
            values = [record[metric] if metric == "quality" else record["reader"][metric]
                      for record in selected]
            average = sum(Fraction(value["numerator"], value["denominator"]) for value in values) / len(values)
            averages[method][metric] = rational_record(average)
    summary = {"status": "validated", "approximation_checks": len(results),
               "ratio_grid_checks": grid_checks, "averages": averages,
               "seconds": time.monotonic() - started, "classification": configuration["classification"]}
    (directory / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
    (directory / "runtime.json").write_text(json.dumps({"platform": platform.platform(), "python": sys.version,
        "source_results_sha256": source_digest,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
