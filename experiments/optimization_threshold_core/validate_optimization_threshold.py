"""Check the LN-367 threshold recovery identity by exact finite integration.

This enumerates small validation problems. It neither establishes a learned
advantage nor approximates an all-reader removal experiment.
"""

from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import platform
import random
import resource
import sys
import time


def feasible(problem, candidate):
    return (
        isinstance(candidate, (list, tuple))
        and len(candidate) == len(problem["weights"])
        and all(type(choice) is int and choice in (0, 1) for choice in candidate)
        and sum(weight * choice for weight, choice in zip(problem["weights"], candidate))
        <= problem["capacity"]
    )


def objective(problem, candidate):
    return sum(value * choice for value, choice in zip(problem["values"], candidate))


def disclosure_predicate(problem, threshold, candidate):
    return feasible(problem, candidate) and objective(problem, candidate) >= threshold


def heuristic_plan(problem, ordering):
    count = len(problem["weights"])
    if ordering == "value_density":
        indices = sorted(range(count), key=lambda index: (
            -Fraction(problem["values"][index], problem["weights"][index]), index))
    elif ordering == "value":
        indices = sorted(range(count), key=lambda index: (-problem["values"][index], index))
    elif ordering == "weight":
        indices = sorted(range(count), key=lambda index: (problem["weights"][index], index))
    else:
        raise ValueError(ordering)
    selected = [0] * count
    remaining = problem["capacity"]
    for index in indices:
        if problem["weights"][index] <= remaining:
            selected[index] = 1
            remaining -= problem["weights"][index]
    return tuple(selected)


def rational_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator,
            "decimal": float(value)}


def validate_problem(problem, candidates):
    # Reference enumeration uses subsets, independently of feasible() and objective().
    reference = []
    for subset in range(1 << len(problem["weights"])):
        indices = [index for index in range(len(problem["weights"])) if subset & (1 << index)]
        if sum(problem["weights"][index] for index in indices) <= problem["capacity"]:
            reference.append((tuple(int(index in indices) for index in range(len(problem["weights"]))),
                              sum(problem["values"][index] for index in indices)))
    optimum = max(value for _, value in reference)
    radius = Fraction(optimum)
    breakpoints = sorted({Fraction(0), radius} | {
        Fraction(optimum - value) for _, value in reference if 0 < optimum - value < radius})
    cells = []
    for lower, upper in zip(breakpoints, breakpoints[1:]):
        margin = (lower + upper) / 2
        for sign in (-1, 1):
            threshold = optimum + sign * margin
            # Existential confidentiality truth computed from every reference witness.
            truth = any(value >= threshold for _, value in reference)
            assert truth == (sign == -1)
            cells.append((threshold, truth, (upper - lower) / (2 * radius)))
    assert sum(weight for _, _, weight in cells) == 1
    rows = []
    omitted_feasibility_errors = 0
    reference_values = dict(reference)
    for candidate in candidates:
        valid = feasible(problem, candidate)
        assert valid == (candidate in reference_values)
        score = max(Fraction(0), 1 - Fraction(optimum - objective(problem, candidate), radius)) if valid else Fraction(0)
        accuracy = sum(weight for threshold, truth, weight in cells
                       if disclosure_predicate(problem, threshold, candidate) == truth)
        assert accuracy == (1 + score) / 2
        if not valid:
            omitted_feasibility_errors += sum(
                1 for threshold, truth, _ in cells
                if (objective(problem, candidate) >= threshold) != truth)
        rows.append({"candidate": list(candidate), "feasible": valid,
                     "objective": objective(problem, candidate),
                     "quality": rational_record(score), "accuracy": rational_record(accuracy)})
    malformed = [None, [], [0] * (len(problem["weights"]) + 1),
                 [2] + [0] * (len(problem["weights"]) - 1),
                 [True] + [0] * (len(problem["weights"]) - 1),
                 [0.0] * len(problem["weights"]), "000000"]
    for candidate in malformed:
        assert not feasible(problem, candidate)
        assert all(not disclosure_predicate(problem, threshold, candidate) for threshold, _, _ in cells)
    # If optimum is supplied, even an empty useful procedure is unnecessary.
    exposed_optimum_accuracy = sum(weight for threshold, truth, weight in cells
                                   if (threshold <= optimum) == truth)
    assert exposed_optimum_accuracy == 1
    by_candidate = {tuple(row["candidate"]): row for row in rows}
    methods = {name: heuristic_plan(problem, name) for name in ("value_density", "value", "weight")}
    methods["exact_enumerator"] = max(reference, key=lambda item: item[1])[0]
    methods["empty_allocation"] = (0,) * len(problem["weights"])
    comparisons = {name: by_candidate[plan] for name, plan in methods.items()}
    return {"problem": problem, "optimum": optimum, "radius": int(radius),
            "feasible_plan_count": len(reference), "integration_cell_count": len(cells),
            "candidate_results": rows, "public_comparisons": comparisons,
            "malformed_controls": len(malformed),
            "omitted_feasibility_errors": omitted_feasibility_errors,
            "exposed_optimum_accuracy": rational_record(exposed_optimum_accuracy)}


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / "config.json").read_text())
    resource.setrlimit(resource.RLIMIT_FSIZE,
                       (configuration["maximum_output_bytes"], configuration["maximum_output_bytes"]))
    assert configuration["radius"] == "optimum"
    generator = random.Random(configuration["seed"])
    candidates = list(product((0, 1), repeat=configuration["item_count"]))
    results = []
    for index in range(configuration["problem_count"]):
        weights = [generator.randint(1, configuration["maximum_weight"])
                   for _ in range(configuration["item_count"])]
        values = [generator.randint(1, configuration["maximum_value"])
                  for _ in range(configuration["item_count"])]
        capacity = generator.randint(min(weights), sum(weights) - 1)
        results.append(validate_problem({"index": index, "weights": weights, "values": values,
                                         "capacity": capacity}, candidates))
    comparisons = {}
    for name in results[0]["public_comparisons"]:
        comparisons[name] = {}
        for metric in ("quality", "accuracy"):
            values = [Fraction(result["public_comparisons"][name][metric]["numerator"],
                               result["public_comparisons"][name][metric]["denominator"])
                      for result in results]
            comparisons[name][metric] = rational_record(sum(values) / len(values))
    summary = {"status": "validated", "problem_count": len(results),
               "candidate_checks": sum(len(result["candidate_results"]) for result in results),
               "integration_cells": sum(result["integration_cell_count"] for result in results),
               "malformed_controls": sum(result["malformed_controls"] for result in results),
               "omitted_feasibility_errors": sum(result["omitted_feasibility_errors"] for result in results),
               "public_comparisons": comparisons, "seconds": time.monotonic() - started,
               "classification": configuration["classification"]}
    assert summary["omitted_feasibility_errors"] > 0
    (directory / "results.json").write_text(json.dumps({"summary": summary, "problems": results}, indent=2) + "\n")
    (directory / "runtime.json").write_text(json.dumps({"platform": platform.platform(),
        "python": sys.version, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configuration_sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
