"""Independent finite certificate audit; never imports the frontier implementation."""
import argparse
import copy
import hashlib
import itertools
import json
import random
import runpy
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference_predictions(family, measure, score):
    lengths = list(range(5)) if measure == "length_0_through_4" else [int(measure[7:])]
    words = [list(word) for length in lengths for word in itertools.product(range(2), repeat=length)]
    rows = []
    for hazard, useful, phase in itertools.product(range(2), range(2), range(4)):
        row = []
        for word in words:
            final_phase = (phase + len(word)) % 4
            parity = sum(word) % 2
            exposure = sum(action for offset, action in enumerate(word) if (phase + offset) % 4 == 0) % 2
            final_useful = useful ^ parity ^ (hazard * exposure if family == "linked" else 0)
            if score == "useful_bit":
                row.append(final_useful)
            elif score == "scalar_outputs":
                row.extend([final_useful, final_phase >> 1, final_phase & 1])
            else:
                row.append(final_useful + 2 * (final_phase >> 1) + 4 * (final_phase & 1))
        rows.append(row)
    return words, rows


def exhaustive_costs(distances):
    minima = [None] * 9
    for permutation in itertools.permutations(range(8)):
        ordered = sorted(distances[row][permutation[row]] for row in range(8))
        total = 0
        for cardinality in range(9):
            if minima[cardinality] is None or total < minima[cardinality]:
                minima[cardinality] = total
            if cardinality < 8:
                total += ordered[cardinality]
    return minima


def class_errors(classes, predictions, hazards):
    require(sorted(state for group in classes for state in group) == list(range(16)), "Partition does not cover source once")
    judgment_errors = 0
    useful_errors = 0
    for group in classes:
        require(group, "Empty class")
        counts = Counter(hazards[state] for state in group)
        judgment_errors += len(group) - max(counts.values())
        for column in range(len(predictions[0])):
            counts = Counter(predictions[state][column] for state in group)
            useful_errors += len(group) - max(counts.values())
    return judgment_errors, useful_errors


def check_case(case, enumerate_matchings=True):
    words, predictions = reference_predictions(case["family"], case["measure"], case["score"])
    require(case["words"] == words and case["predictions"] == predictions, "Prediction certificate mismatch")
    query_count = len(predictions[0])
    require(case["query_count"] == query_count, "Incorrect query normalization")
    distances = [[sum(predictions[row][column] != predictions[other + 8][column] for column in range(query_count))
                  for other in range(8)] for row in range(8)]
    require(case["distances"] == distances, "Incorrect edge distance")
    layers = case["matching_layers"]
    require(len(layers) == 9 and all(len(layer) == 256 for layer in layers), "Incomplete recurrence certificate")
    require(layers[0] == [0] + [None] * 255, "Incorrect recurrence initial condition")
    for count in range(1, 9):
        for mask in range(256):
            choices = [layers[count - 1][mask]]
            for column in range(8):
                if mask & (1 << column):
                    previous = layers[count - 1][mask ^ (1 << column)]
                    if previous is not None:
                        choices.append(previous + distances[count - 1][column])
            choices = [value for value in choices if value is not None]
            require(layers[count][mask] == (min(choices) if choices else None), "Invalid recurrence cell")
    recurrence_costs = [min(value for mask, value in enumerate(layers[8]) if mask.bit_count() == cardinality)
                        for cardinality in range(9)]
    if enumerate_matchings:
        require(exhaustive_costs(distances) == recurrence_costs, "Independent permutation optimum disagrees")
    differences = [recurrence_costs[index + 1] - recurrence_costs[index] for index in range(8)]
    require(differences == sorted(differences) and min(differences) >= 0, "Nonconvex or decreasing frontier")
    require(len(case["frontier"]) == 9, "Missing frontier endpoint")
    hazards = [state // 8 for state in range(16)]
    for cardinality, point in enumerate(case["frontier"]):
        require(point["cardinality"] == cardinality, "Misordered frontier")
        pairs = point["pairs"]
        require(len(pairs) == cardinality, "Incorrect matching size")
        require(all(len(pair) == 2 and 0 <= pair[0] < 8 <= pair[1] < 16 for pair in pairs), "Invalid bipartite edge")
        require(len({state for pair in pairs for state in pair}) == 2 * cardinality, "Repeated matching vertex")
        classes = [[state] for state in range(16) if all(state not in pair for pair in pairs)] + pairs
        require(point["classes"] == classes, "Witness classes differ from matching")
        judgment_errors, useful_errors = class_errors(classes, predictions, hazards)
        require(judgment_errors == cardinality, "Witness retains too much judgment information")
        require(useful_errors == recurrence_costs[cardinality] == point["distance_sum"], "Witness utility is not optimal")
        require(point["judgment_error"] == [cardinality, 16], "Incorrect judgment-error normalization")
        require(point["useful_error"] == [useful_errors, 16 * query_count], "Incorrect utility-error normalization")
        require(point["retained_state_bits"] == (len(classes) - 1).bit_length(), "Incorrect retained-state size")
        # A publicly known complement must never be counted as judgment removal.
        require(class_errors(classes, predictions, [1 - value for value in hazards]) == (judgment_errors, useful_errors), "Complement invariance failed")
        # Adding one source-dependent advice bit H recovers every member of every class.
        restored = [[state for state in group if hazards[state] == label] for group in classes for label in range(2)]
        restored = [group for group in restored if group]
        require(class_errors(restored, predictions, hazards) == (0, 0), "Advice repair failed")
    # Explicit publicly tagged mixtures attain every tested between-endpoint value.
    for cardinality in range(8):
        for probability in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
            judgment_error = Fraction(0)
            utility_error = Fraction(0)
            for weight, point in ((1 - probability, case["frontier"][cardinality]),
                                  (probability, case["frontier"][cardinality + 1])):
                for group in point["classes"]:
                    masses = [sum(weight / 16 for state in group if hazards[state] == value) for value in range(2)]
                    judgment_error += min(masses)
                    for column in range(query_count):
                        probabilities = {}
                        for state in group:
                            answer = predictions[state][column]
                            probabilities[answer] = probabilities.get(answer, Fraction(0)) + weight / 16
                        utility_error += (sum(probabilities.values()) - max(probabilities.values())) / query_count
            require(judgment_error == (cardinality + probability) / 16, "Mixture judgment error failed")
            expected_loss = ((1 - probability) * recurrence_costs[cardinality] + probability * recurrence_costs[cardinality + 1]) / (16 * query_count)
            require(utility_error == expected_loss, "Mixture utility loss failed")
    constant_error, constant_loss = class_errors([list(range(16))], predictions, hazards)
    require(constant_error == 8, "Unbalanced source")
    require(class_errors([[state] for state in range(16)], predictions, hazards) == (0, 0), "Intact oracle failed")
    # Head deletion preserves the intact state, so the best judgment reader remains exact.
    head_deleted_error = class_errors([[state] for state in range(16)], predictions, hazards)[0]
    require(head_deleted_error == 0, "Head deletion mislabeled as removal")
    # Conditional uniformity explains a general retained-advantage lower bound.
    # Each uniform random bijection has the same expected edge cost, computable
    # from the complete edge matrix without choosing the optimal matching.
    uniform_queries = True
    for column in range(query_count):
        first_counts = Counter(predictions[state][column] for state in range(8))
        second_counts = Counter(predictions[state][column] for state in range(8, 16))
        uniform_queries &= first_counts == second_counts and len(set(first_counts.values())) == 1
    require(uniform_queries, "Conditional-uniformity premise failed")
    average_matching_cost = Fraction(sum(sum(row) for row in distances), 8)
    require(average_matching_cost == Fraction(constant_loss, 2), "Uniform-pairing bound mismatch")
    require(recurrence_costs[-1] <= average_matching_cost, "Optimum exceeds random-matching mean")
    for seed in (19, 23):
        permutation = list(range(16))
        random.Random(seed).shuffle(permutation)
        renamed_predictions = [None] * 16
        renamed_hazards = [None] * 16
        for state in range(16):
            renamed_predictions[permutation[state]] = predictions[state]
            renamed_hazards[permutation[state]] = hazards[state]
        for point in case["frontier"]:
            renamed_classes = [[permutation[state] for state in group] for group in point["classes"]]
            require(class_errors(renamed_classes, renamed_predictions, renamed_hazards) == (point["cardinality"], point["distance_sum"]), "State-relabeling control failed")
    exact_loss = Fraction(recurrence_costs[-1], 16 * query_count)
    baseline_loss = Fraction(constant_loss, 16 * query_count)
    simple_classes = [[state, state + 8] for state in range(8)]
    simple_judgment_error, simple_loss = class_errors(simple_classes, predictions, hazards)
    explicit_reader_loss = sum(predictions[state][column] != predictions[state % 8][column]
                               for state in range(16) for column in range(query_count))
    require(simple_judgment_error == 8 and simple_loss == explicit_reader_loss == recurrence_costs[-1],
            "Simple three-bit replacement does not attain complete-removal optimum")
    return {"family": case["family"], "measure": case["measure"], "score": case["score"],
            "minimum_distance_by_cardinality": recurrence_costs,
            "perfect_removal_accuracy": str(1 - exact_loss), "constant_baseline_accuracy": str(1 - baseline_loss),
            "lost_advantage_fraction": str(exact_loss / baseline_loss),
            "full_matching": case["frontier"][-1]["pairs"],
            "uniform_conditional_labels": uniform_queries,
            "random_matching_accuracy_bound": str(1 - average_matching_cost / (16 * query_count)),
            "simple_replacement_attains_optimum": True,
            "retained_state_bits": case["frontier"][-1]["retained_state_bits"]}


def corruption_controls(case):
    changes = [
        ("edge_distance", lambda value: value["distances"][0].__setitem__(0, value["distances"][0][0] + 1)),
        ("prediction", lambda value: value["predictions"][0].__setitem__(0, 99)),
        ("recurrence", lambda value: value["matching_layers"][1].__setitem__(0, 1)),
        ("missing_endpoint", lambda value: value["frontier"].pop()),
        ("judgment_normalization", lambda value: value["frontier"][8].__setitem__("judgment_error", [7, 16])),
        ("state_budget", lambda value: value["frontier"][8].__setitem__("retained_state_bits", 2)),
        ("matching_vertex", lambda value: value["frontier"][8]["pairs"][0].__setitem__(1, 16)),
    ]
    rejected = []
    for name, change in changes:
        altered = copy.deepcopy(case)
        change(altered)
        try:
            check_case(altered, enumerate_matchings=False)
        except (ValueError, IndexError):
            rejected.append(name)
        else:
            raise ValueError("Corrupted certificate accepted: " + name)
    return rejected


def run(directory, destination):
    started = time.monotonic()
    require(not destination.exists(), "Audit output already exists")
    source = Path(__file__).resolve().parent
    receipt = json.loads((directory / "receipt.json").read_text())
    certificate_bytes = (directory / "certificate.json").read_bytes()
    require(hashlib.sha256(certificate_bytes).hexdigest() == receipt["certificate_sha256"], "Certificate hash mismatch")
    actual_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source.iterdir()) if path.is_file() and not path.name.startswith("._")}
    require(actual_hashes == receipt["source_hashes"], "Frozen source inventory or hash mismatch")
    certificate = json.loads(certificate_bytes)
    configuration = json.loads((source / "config.json").read_text())
    require(certificate["configuration"] == configuration, "Configuration mismatch")
    historical = runpy.run_path(str(source / "parent_models.py"))
    for linked, state, action in itertools.product((False, True), range(16), range(2)):
        phase, useful, hazard = state % 4, state // 4 % 2, state // 8
        next_useful = useful ^ action ^ (hazard if linked and phase == 0 and action else 0)
        expected_state = (phase + 1) % 4 + 4 * next_useful + 8 * hazard
        require(historical["transition"](state, action, linked) == expected_state, "Historical transition mismatch")
    expected = list(itertools.product(configuration["families"], configuration["query_measures"], configuration["scores"]))
    observed = [(case["family"], case["measure"], case["score"]) for case in certificate["cases"]]
    require(observed == expected and len(expected) == 18, "Incomplete or duplicate case inventory")
    summaries = [check_case(case) for case in certificate["cases"]]
    rejections = corruption_controls(certificate["cases"][9])
    require(receipt["case_count"] == 18 and receipt["frontier_endpoint_count"] == 162, "Incorrect receipt counts")
    output = {"passed": True, "cases": summaries, "case_count": 18, "frontier_endpoint_count": 162,
              "permutation_comparisons": 18 * 40320, "corruption_rejections": rejections,
              "wall_seconds": time.monotonic() - started, "certificate_sha256": receipt["certificate_sha256"],
              "scope": "Exact finite prediction certificate plus a separate analytic reduction to stochastic encodings; not machine-checked proof of that reduction."}
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "cases": 18, "corruption_rejections": len(rejections), "wall_seconds": output["wall_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.directory, arguments.output)
