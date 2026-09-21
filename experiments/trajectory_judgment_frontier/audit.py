"""Independent closed-form, exhaustive-word and assignment checks; no runner imports."""
import argparse
import copy
import hashlib
import itertools
import json
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exposure_count(phase, horizon):
    return sum((phase + step) % 4 == 0 for step in range(horizon))


def reference_trace(state, word, linked):
    phase, useful, hazard = state % 4, state // 4 % 2, state // 8
    trace = []
    for length in range(len(word) + 1):
        exposure = sum(word[offset] for offset in range(length) if (phase + offset) % 4 == 0) % 2
        value = useful ^ (sum(word[:length]) % 2) ^ (hazard * exposure if linked else 0)
        trace.append(((phase + length) % 4, value))
    return tuple(trace)


def enumerate_costs(distances):
    minima = [None] * 9
    for permutation in itertools.permutations(range(8)):
        ordered = sorted(distances[row][permutation[row]] for row in range(8))
        total = 0
        for count in range(9):
            if minima[count] is None or total < minima[count]:
                minima[count] = total
            if count < 8:
                total += ordered[count]
    return minima


def explicit_words(horizon, linked, expected_accuracy, expected_feedback, expected_bits, expected_baseline):
    correct = feedback_correct = correct_bits = baseline_count = 0
    word_count = 1 << horizon
    for word in itertools.product(range(2), repeat=horizon):
        traces = [reference_trace(state, word, linked) for state in range(16)]
        baseline_count += max(Counter(traces).values())
        for state in range(16):
            correct += traces[state] == traces[state % 8]
            estimate = 0
            actual_phase, actual_useful = traces[state][0]
            correct_bits += 1  # The retained initial (p,u) is exact.
            for index, action in enumerate(word):
                predicted = actual_useful ^ action ^ (estimate if linked and actual_phase == 0 and action == 1 else 0)
                next_phase, next_useful = traces[state][index + 1]
                correct_bits += predicted == next_useful
                if linked and actual_phase == 0 and action == 1:
                    estimate = actual_useful ^ action ^ next_useful
                actual_phase, actual_useful = next_phase, next_useful
            feedback_correct += estimate == state // 8
    require(Fraction(correct, 16 * word_count) == expected_accuracy, "Explicit whole-trajectory accuracy disagrees")
    require(Fraction(feedback_correct, 16 * word_count) == expected_feedback, "Explicit feedback restoration disagrees")
    require(Fraction(correct_bits, 16 * word_count * (horizon + 1)) == expected_bits, "Explicit online prediction disagrees")
    require(Fraction(baseline_count, 16 * word_count) == expected_baseline, "Explicit constant-state baseline disagrees")
    return 16 * word_count


def check_family(family, configuration, exhaustive=True):
    linked = family["family"] == "linked"
    tables = family["agreement_counts"]
    require(len(tables) == max(configuration["horizons"]) + 1, "Missing agreement layer")
    for horizon, table in enumerate(tables):
        require(len(table) == 16 and all(len(row) == 16 for row in table), "Wrong agreement dimensions")
        for first, second in itertools.product(range(16), repeat=2):
            expected = 0
            if first % 8 == second % 8:
                exponent = horizon
                if linked and first // 8 != second // 8:
                    exponent -= exposure_count(first % 4, horizon)
                expected = 1 << exponent
            require(table[first][second] == expected, "Wrong complete-prefix agreement count")
    require([case["horizon"] for case in family["cases"]] == configuration["horizons"], "Wrong horizon inventory")
    summaries = []
    explicit_count = 0
    for case in family["cases"]:
        horizon = case["horizon"]
        word_count = 1 << horizon
        require(case["word_count"] == word_count, "Wrong action-word denominator")
        costs = [[word_count - tables[horizon][row][column + 8] for column in range(8)] for row in range(8)]
        require(case["distances"] == costs, "Wrong matching cost")
        # Initial outputs distinguish (p,u); every off-diagonal edge costs all words.
        # Every diagonal edge costs at most this, so the cheapest k diagonal edges
        # certify an optimum independently of either search implementation.
        diagonal = sorted(costs[index][index] for index in range(8))
        optimum = [sum(diagonal[:count]) for count in range(9)]
        if exhaustive:
            require(enumerate_costs(costs) == optimum, "Permutation optimum disagrees")
        layers = case["matching_layers"]
        require(len(layers) == 9 and all(len(layer) == 256 for layer in layers), "Incomplete matching recurrence")
        require(layers[0] == [0] + [None] * 255, "Wrong matching recurrence base")
        for row in range(1, 9):
            for mask in range(256):
                candidates = [layers[row - 1][mask]]
                for column in range(8):
                    if mask & (1 << column) and layers[row - 1][mask ^ (1 << column)] is not None:
                        candidates.append(layers[row - 1][mask ^ (1 << column)] + costs[row - 1][column])
                candidates = [value for value in candidates if value is not None]
                require(layers[row][mask] == (min(candidates) if candidates else None), "Wrong matching recurrence cell")
        require(len(case["frontier"]) == 9, "Missing frontier endpoint")
        for count, point in enumerate(case["frontier"]):
            pairs = point["pairs"]
            require(point["cardinality"] == len(pairs) == count, "Wrong matching cardinality")
            require(all(len(pair) == 2 and 0 <= pair[0] < 8 <= pair[1] < 16 for pair in pairs), "Invalid pair")
            require(len({state for pair in pairs for state in pair}) == 2 * count, "Overlapping pairs")
            expected_classes = [[state] for state in range(16) if all(state not in pair for pair in pairs)] + pairs
            require(point["classes"] == expected_classes, "Wrong encoded classes")
            require(sum(costs[first][second - 8] for first, second in pairs) == point["distance_sum"] == optimum[count], "Wrong optimal witness")
            require(point["judgment_error"] == [count, 16], "Wrong Bayes judgment error")
            require(point["trajectory_error"] == [optimum[count], 16 * word_count], "Wrong trajectory loss")
        ambiguity = sum(Fraction(1, 1 << exposure_count(phase, horizon)) for phase in range(4)) / 4 if linked else Fraction(1)
        accuracy = (1 + ambiguity) / 2
        judgment = 1 - ambiguity / 2 if linked else Fraction(1, 2)
        bits = 1 - (1 - ambiguity) / (2 * (horizon + 1)) if linked else Fraction(1)
        baseline = Fraction(1, 8)
        if linked:
            all_exposed = Fraction(1)
            for phase in range(4):
                all_exposed *= 1 - Fraction(1, 1 << exposure_count(phase, horizon))
            baseline = (2 - all_exposed) / 16
        require(Fraction(case["no_observation_trajectory_accuracy"]) == accuracy == 1 - Fraction(optimum[-1], 16 * word_count), "Wrong full-erasure optimum")
        require(Fraction(case["feedback_judgment_accuracy"]) == judgment, "Wrong feedback judgment result")
        require(Fraction(case["feedback_useful_bit_accuracy"]) == bits, "Wrong feedback useful result")
        if exhaustive and horizon <= configuration["explicit_word_horizon"]:
            explicit_count += explicit_words(horizon, linked, accuracy, judgment, bits, baseline)
        summaries.append({"family": family["family"], "horizon": horizon, "trajectory_accuracy": str(accuracy),
                          "constant_state_baseline": str(baseline), "retained_advantage_fraction": str((accuracy - baseline) / (1 - baseline)),
                          "feedback_judgment_accuracy": str(judgment), "feedback_useful_bit_accuracy": str(bits)})
    return summaries, explicit_count


def run(directory, output):
    started = time.monotonic()
    require(not output.exists(), "Audit output already exists")
    source = Path(__file__).resolve().parent
    configuration = json.loads((source / "config.json").read_text())
    require(configuration["families"] == ["independent", "linked"] and configuration["horizons"] == [0, 1, 2, 4, 8, 16, 32, 64], "Unexpected frozen contract")
    receipt = json.loads((directory / "receipt.json").read_text())
    encoded = (directory / "certificate.json").read_bytes()
    require(hashlib.sha256(encoded).hexdigest() == receipt["certificate_sha256"], "Certificate hash mismatch")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source.iterdir()) if path.is_file() and not path.name.startswith("._")}
    require(hashes == receipt["source_hashes"], "Frozen source hash mismatch")
    certificate = json.loads(encoded)
    require(certificate["configuration"] == configuration, "Configuration mismatch")
    require([family["family"] for family in certificate["families"]] == configuration["families"], "Family inventory mismatch")
    summaries = []
    explicit_count = 0
    for family in certificate["families"]:
        values, count = check_family(family, configuration)
        summaries.extend(values)
        explicit_count += count
    mutations = [
        ("agreement", lambda value: value["agreement_counts"][4][0].__setitem__(8, 0)),
        ("denominator", lambda value: value["cases"][3].__setitem__("word_count", 15)),
        ("recurrence", lambda value: value["cases"][0]["matching_layers"][1].__setitem__(0, 1)),
        ("judgment_error", lambda value: value["cases"][3]["frontier"][8].__setitem__("judgment_error", [7, 16])),
        ("feedback", lambda value: value["cases"][3].__setitem__("feedback_judgment_accuracy", "1/2")),
        ("horizon_inventory", lambda value: value["cases"].pop()),
    ]
    rejections = []
    for name, mutate in mutations:
        changed = copy.deepcopy(certificate["families"][1])
        mutate(changed)
        try:
            check_family(changed, configuration, exhaustive=False)
        except ValueError:
            rejections.append(name)
        else:
            raise ValueError("Accepted corrupted certificate: " + name)
    require(receipt["case_count"] == len(summaries) == 16, "Receipt count mismatch")
    result = {"passed": True, "cases": summaries, "case_count": len(summaries), "frontier_endpoints": 144,
              "agreement_cells": 2 * 65 * 16 * 16, "permutations_checked": 16 * 40320,
              "explicit_source_words": explicit_count, "corruption_rejections": rejections,
              "certificate_sha256": receipt["certificate_sha256"], "wall_seconds": time.monotonic() - started}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.directory, arguments.output)
