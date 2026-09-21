"""Independent traversal oracle, simple-path bounds and binomial recurrence."""

from copy import deepcopy
from fractions import Fraction
from itertools import product
import math
import random


def reachable(edges, active, count):
    adjacency = [[] for _ in range(count)]
    for included, (first, second) in zip(active, edges):
        if included:
            adjacency[first].append(second)
            adjacency[second].append(first)
    visited = {0}
    pending = [0]
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return [int(node in visited) for node in range(1, count)]


def prepare_reference(configuration):
    count = configuration["node_count"]
    assert count == 8 and configuration["source_node"] == 0 and configuration["critical_node"] == 7
    denominator = configuration["failure_denominator"]
    reference = []
    for split in ["development", "evaluation"]:
        generator = random.Random(configuration[split + "_seed"])
        alternatives = [(first, second) for first in range(count) for second in range(first + 2, count)]
        for instance_index in range(configuration[split + "_count"]):
            edges = sorted([(node, node + 1) for node in range(count - 1)] + generator.sample(alternatives, configuration["additional_edges"]))
            stress = configuration["stress_numerators"][instance_index % len(configuration["stress_numerators"])]
            failures = [stress * generator.choice(configuration["edge_multipliers"]) for _ in edges]
            totals = [0] * (count - 1)
            total_mass = 0
            for active in product([False, True], repeat=len(edges)):
                weight = math.prod(denominator - failure if included else failure for failure, included in zip(failures, active))
                total_mass += weight
                for index, connected in enumerate(reachable(edges, active, count)):
                    totals[index] += weight * connected
            assert total_mass == denominator ** len(edges)
            targets = [Fraction(total, total_mass) for total in totals]
            adjacency = [[] for _ in range(count)]
            for (first, second), failure in zip(edges, failures):
                probability = Fraction(denominator - failure, denominator)
                adjacency[first].append((second, probability))
                adjacency[second].append((first, probability))
            path_bounds = [Fraction(0)] * count
            def visit(node, visited, probability):
                path_bounds[node] = max(path_bounds[node], probability)
                for neighbor, availability in adjacency[node]:
                    if neighbor not in visited:
                        visit(neighbor, visited | {neighbor}, probability * availability)
            visit(0, {0}, Fraction(1))
            cut_bounds = []
            for terminal in range(1, count):
                best_failure = Fraction(0)
                other_nodes = [node for node in range(1, count) if node != terminal]
                for choices in product([False, True], repeat=len(other_nodes)):
                    included_nodes = {0} | {node for node, choice in zip(other_nodes, choices) if choice}
                    probability = math.prod(Fraction(failure, denominator) for (first, second), failure in zip(edges, failures)
                                            if (first in included_nodes) != (second in included_nodes))
                    best_failure = max(best_failure, probability)
                cut_bounds.append(1 - best_failure)
            reference.append({"split": split, "index": instance_index, "edges": list(map(list, edges)), "failures": failures,
                              "targets": targets, "path": path_bounds[1:], "cut": cut_bounds})
    return reference


def balanced_score(labels, alarms):
    rates = []
    for label in range(2):
        scores = [alarm if label else 1 - alarm for actual, alarm in zip(labels, alarms) if actual == label]
        assert scores
        rates.append(sum(scores) / len(scores))
    return sum(rates) / 2


def audit_certificate(certificate, configuration, reference):
    assert certificate["scope"] == configuration["interpretation"]
    rows = certificate["instances"]
    assert len(rows) == len(reference)
    threshold = Fraction(configuration["connectivity_threshold"])
    oracle_cases = 0
    for row, expected in zip(rows, reference):
        for field in ["split", "index", "edges", "failures"]:
            assert row[field] == expected[field]
        assert len(row["edges"]) == len({tuple(edge) for edge in row["edges"]}) == 10
        oracle_cases += 1 << len(row["edges"])
        for field in ["targets", "path", "cut"]:
            assert list(map(Fraction, row[field])) == expected[field]
        assert row["hazard"] == int(expected["targets"][-1] <= threshold)
        for lower, truth, upper in zip(expected["path"], expected["targets"], expected["cut"]):
            assert 0 <= lower <= truth <= upper <= 1
        assert list(map(Fraction, row["midpoint"])) == [(left + right) / 2 for left, right in zip(expected["path"], expected["cut"])]
        alarm = Fraction(1) if expected["cut"][-1] <= threshold else Fraction(0) if expected["path"][-1] > threshold else Fraction(1, 2)
        assert Fraction(row["certified_alarm_probability"]) == alarm
        if alarm != Fraction(1, 2):
            assert alarm == row["hazard"]
        assert row["path_relaxations"] == 2 * len(row["edges"])
        assert row["cut_edge_checks"] == ((1 << (configuration["node_count"] - 1)) - 1) * len(row["edges"])
        for field in ["oracle_seconds", "path_seconds", "bounds_seconds"]:
            assert math.isfinite(row[field]) and row[field] > 0
        assert row["bounds_seconds"] >= row["path_seconds"]
        assert set(row["sampling"]) == (set(map(str, configuration["sample_counts"])) if row["split"] == "evaluation" else set())
        for count_text, sampling in row["sampling"].items():
            sample_count = int(count_text)
            assert sampling["seed"] == configuration["sampling_seed"] + 10000 * row["index"] + sample_count
            generator = random.Random(sampling["seed"])
            totals = [0] * (configuration["node_count"] - 1)
            for _ in range(sample_count):
                active = [generator.randrange(configuration["failure_denominator"]) >= failure for failure in row["failures"]]
                indicators = reachable(row["edges"], active, configuration["node_count"])
                totals = [left + right for left, right in zip(totals, indicators)]
            assert list(map(Fraction, sampling["forecast"])) == [Fraction(total, sample_count) for total in totals]
            expected_loss = sum(probability - probability ** 2 for probability in expected["targets"]) / (sample_count * len(totals))
            assert Fraction(sampling["expected_squared_error"]) == expected_loss
            probability = float(expected["targets"][-1])
            distribution = [1.0]
            for _ in range(sample_count):
                following = [0.0] * (len(distribution) + 1)
                for successes, mass in enumerate(distribution):
                    following[successes] += mass * (1 - probability)
                    following[successes + 1] += mass * probability
                distribution = following
            expected_alarm = math.fsum(distribution[:int(threshold * sample_count) + 1])
            assert abs(float(Fraction(sampling["expected_alarm_probability"])) - expected_alarm) < 1e-12
            assert abs(math.fsum(distribution) - 1) < 1e-12
            assert sampling["edge_draws"] == sample_count * len(row["edges"])
            assert math.isfinite(sampling["elapsed_seconds"]) and sampling["elapsed_seconds"] > 0

    development = [row for row in rows if row["split"] == "development"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    targets = [list(map(Fraction, row["targets"])) for row in evaluation]
    dimension = len(targets[0])
    summary = certificate["summary"]
    development_mean = [sum(Fraction(row["targets"][node]) for row in development) / len(development) for node in range(dimension)]
    labels = [row["hazard"] for row in evaluation]
    sizes = [labels.count(label) for label in range(2)]
    assert summary["evaluation_class_counts"] == sizes and min(sizes) > 0
    means = [sum(values) / len(values) for values in zip(*targets)]
    class_means = [[sum(target[node] for target, label in zip(targets, labels) if label == hazard) / sizes[hazard]
                    for node in range(dimension)] for hazard in range(2)]
    assert summary["development_mean"] == list(map(str, development_mean))
    assert summary["evaluation_mean"] == list(map(str, means))
    assert summary["evaluation_class_means"] == [list(map(str, values)) for values in class_means]
    variance = sum(sum((value - center) ** 2 for value, center in zip(target, means)) for target in targets) / (len(targets) * dimension)
    between = sum(sum((value - center) ** 2 for value, center in zip(class_means[label], means)) for label in labels) / (len(labels) * dimension)
    difference = [right - left for left, right in zip(*class_means)]
    bound = max(abs(sum(direction * (corner - center) for direction, corner, center in zip(difference, vertex, means)))
                for vertex in product([0, 1], repeat=dimension))
    probability = Fraction(sizes[1], len(labels))
    floor = between - 8 * bound * probability * (1 - probability) * (Fraction(configuration["reader_advantage_cap"]) + Fraction(1, 1 << configuration["reader_bits"])) / dimension
    for field, expected in [("reference_constant_variance", variance), ("between_class_variance", between),
                            ("projection_bound", bound), ("one_call_loss_floor", floor), ("sufficient_loss_fraction", floor / variance)]:
        assert Fraction(summary[field]) == expected
    expected_names = ["development_mean", "path", "cut", "midpoint", "certified_reader"] + ["sampling_" + str(count) for count in configuration["sample_counts"]]
    assert list(summary["solvers"]) == expected_names
    for name, metrics in summary["solvers"].items():
        if name == "development_mean":
            predictions = [development_mean] * len(rows)
            predictions = predictions[:len(evaluation)]
            alarms = [Fraction(development_mean[-1] <= threshold)] * len(evaluation)
            times = None
        elif name.startswith("sampling_"):
            sample_count = name.removeprefix("sampling_")
            sample_rows = [row["sampling"][sample_count] for row in evaluation]
            predictions = [list(map(Fraction, row["forecast"])) for row in sample_rows]
            alarms = [Fraction(forecast[-1] <= threshold) for forecast in predictions]
            expected_alarms = [Fraction(row["expected_alarm_probability"]) for row in sample_rows]
            assert Fraction(metrics["expected_balanced_accuracy"]) == balanced_score(labels, expected_alarms)
            assert Fraction(metrics["expected_mean_squared_error"]) == sum(Fraction(row["expected_squared_error"]) for row in sample_rows) / len(sample_rows)
            times = [row["elapsed_seconds"] for row in sample_rows]
        else:
            field = "midpoint" if name == "certified_reader" else name
            predictions = [list(map(Fraction, row[field])) for row in evaluation]
            alarms = ([Fraction(row["certified_alarm_probability"]) for row in evaluation] if name == "certified_reader"
                      else [Fraction(forecast[-1] <= threshold) for forecast in predictions])
            times = [row["path_seconds" if name == "path" else "bounds_seconds"] for row in evaluation]
        loss = sum(sum((actual - forecast) ** 2 for actual, forecast in zip(target, prediction)) / dimension
                   for target, prediction in zip(targets, predictions)) / len(targets)
        assert Fraction(metrics["mean_squared_error"]) == loss
        assert Fraction(metrics["balanced_judgment_accuracy"]) == balanced_score(labels, alarms)
        if times is not None:
            ordered = sorted(times)
            assert metrics["timing"] == {"median_seconds": ordered[len(ordered) // 2], "p95_seconds": ordered[math.ceil(.95 * len(ordered)) - 1], "maximum_seconds": max(ordered)}
    ceiling = Fraction(1, 2) + Fraction(configuration["reader_advantage_cap"])
    witnesses = [name for name, metrics in summary["solvers"].items() if "timing" in metrics
                 and metrics["timing"]["maximum_seconds"] <= configuration["reader_deadline_seconds"]
                 and Fraction(metrics.get("expected_balanced_accuracy", metrics["balanced_judgment_accuracy"])) > ceiling]
    assert summary["public_reader_rejection_witnesses"] == witnesses
    assert summary["absolute_removal_screen_passed"] == (not witnesses)
    assert summary["severity_screen_passed"] == (floor / variance >= Fraction(configuration["severe_loss_fraction"]))
    identities = [(tuple(map(tuple, row["edges"])), tuple(row["failures"])) for row in rows]
    return {"reference_instances": len(rows), "failure_configurations_reenumerated": oracle_cases,
            "exact_service_probabilities": len(rows) * dimension, "sampling_cases": len(evaluation) * len(configuration["sample_counts"]),
            "duplicate_inputs": len(identities) - len(set(identities)), "public_reader_witnesses": witnesses,
            "training_admitted": False}


def reject_corruptions(certificate, configuration, reference):
    mutations = {
        "reference_probability": lambda changed: changed["instances"][0]["targets"].__setitem__(0, "0"),
        "path_bound": lambda changed: changed["instances"][0]["path"].__setitem__(0, "0"),
        "hazard_label": lambda changed: changed["instances"][0].update(hazard=1 - changed["instances"][0]["hazard"]),
        "sample_forecast": lambda changed: changed["instances"][32]["sampling"]["8"]["forecast"].__setitem__(0, "-1"),
        "sampling_expectation": lambda changed: changed["instances"][32]["sampling"]["8"].update(expected_squared_error="-1"),
        "alarm_probability": lambda changed: changed["instances"][32]["sampling"]["8"].update(expected_alarm_probability="-1"),
        "geometry_bound": lambda changed: changed["summary"].update(sufficient_loss_fraction="1"),
        "admission_decision": lambda changed: changed["summary"].update(absolute_removal_screen_passed=not changed["summary"]["absolute_removal_screen_passed"]),
    }
    rejected = []
    for name, mutate in mutations.items():
        changed = deepcopy(certificate)
        mutate(changed)
        try:
            audit_certificate(changed, configuration, reference)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError("Corruption accepted: " + name)
    return rejected
