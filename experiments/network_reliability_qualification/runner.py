"""Reference reliability calculations and untrained replacement solvers."""

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from math import ceil, comb
from pathlib import Path
import platform
import random
import resource
import sys
import time

from audit import prepare_reference, audit_certificate, reject_corruptions


def generate_inputs(configuration):
    count = configuration["node_count"]
    backbone = [(index, index + 1) for index in range(count - 1)]
    alternatives = [(first, second) for first in range(count) for second in range(first + 2, count)]
    specifications = []
    for split in ["development", "evaluation"]:
        generator = random.Random(configuration[split + "_seed"])
        for index in range(configuration[split + "_count"]):
            edges = sorted(backbone + generator.sample(alternatives, configuration["additional_edges"]))
            stress = configuration["stress_numerators"][index % len(configuration["stress_numerators"])]
            failures = [stress * generator.choice(configuration["edge_multipliers"]) for _ in edges]
            specifications.append({"split": split, "index": index, "edges": list(map(list, edges)), "failures": failures})
    return specifications


def connected_nodes(edges, mask, count):
    parents = list(range(count))
    def representative(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node
    for index, (first, second) in enumerate(edges):
        if mask & (1 << index):
            parents[representative(second)] = representative(first)
    source = representative(0)
    return [representative(node) == source for node in range(1, count)]


def exact_probabilities(specification, configuration):
    denominator = configuration["failure_denominator"]
    edge_count = len(specification["edges"])
    totals = [0] * (configuration["node_count"] - 1)
    normalization = 0
    for mask in range(1 << edge_count):
        weight = 1
        for index, failure in enumerate(specification["failures"]):
            weight *= denominator - failure if mask & (1 << index) else failure
        normalization += weight
        for node, connected in enumerate(connected_nodes(specification["edges"], mask, configuration["node_count"])):
            totals[node] += weight * connected
    assert normalization == denominator ** edge_count
    return [Fraction(total, normalization) for total in totals]


def path_probabilities(specification, configuration):
    count = configuration["node_count"]
    adjacency = [[] for _ in range(count)]
    for (first, second), failure in zip(specification["edges"], specification["failures"]):
        availability = 1 - Fraction(failure, configuration["failure_denominator"])
        adjacency[first].append((second, availability))
        adjacency[second].append((first, availability))
    probabilities = [Fraction(0)] * count
    probabilities[0] = Fraction(1)
    pending = set(range(count))
    relaxations = 0
    while pending:
        selected = max(pending, key=lambda node: probabilities[node])
        pending.remove(selected)
        for neighbor, availability in adjacency[selected]:
            relaxations += 1
            if neighbor in pending:
                probabilities[neighbor] = max(probabilities[neighbor], probabilities[selected] * availability)
    return probabilities[1:], relaxations


def cut_probabilities(specification, configuration):
    count = configuration["node_count"]
    upper = [Fraction(1)] * (count - 1)
    checks = 0
    for subset in range((1 << (count - 1)) - 1):
        inside = (subset << 1) | 1
        cut_failure = Fraction(1)
        for (first, second), failure in zip(specification["edges"], specification["failures"]):
            checks += 1
            if bool(inside & (1 << first)) != bool(inside & (1 << second)):
                cut_failure *= Fraction(failure, configuration["failure_denominator"])
        for node in range(1, count):
            if not inside & (1 << node):
                upper[node - 1] = min(upper[node - 1], 1 - cut_failure)
    return upper, checks


def sample_probabilities(specification, configuration, sample_count, seed):
    generator = random.Random(seed)
    totals = [0] * (configuration["node_count"] - 1)
    for _ in range(sample_count):
        mask = sum(1 << index for index, failure in enumerate(specification["failures"])
                   if generator.randrange(configuration["failure_denominator"]) >= failure)
        for node, connected in enumerate(connected_nodes(specification["edges"], mask, configuration["node_count"])):
            totals[node] += connected
    return [Fraction(total, sample_count) for total in totals]


def alarm_probability(probability, sample_count, threshold):
    numerator, denominator = probability.numerator, probability.denominator
    maximum_connected = int(threshold * sample_count)
    total = sum(comb(sample_count, successes) * numerator ** successes * (denominator - numerator) ** (sample_count - successes)
                for successes in range(maximum_connected + 1))
    return Fraction(total, denominator ** sample_count)


def measure_rows(rows, predictions, alarms):
    targets = [list(map(Fraction, row["targets"])) for row in rows]
    labels = [row["hazard"] for row in rows]
    class_sizes = [labels.count(label) for label in range(2)]
    assert min(class_sizes) > 0, "Qualification requires both judgment classes"
    squared_loss = sum(sum((truth - forecast) ** 2 for truth, forecast in zip(target, prediction)) / len(target)
                       for target, prediction in zip(targets, predictions)) / len(rows)
    accuracy = sum(sum((alarm if label else 1 - alarm) for alarm, actual in zip(alarms, labels) if actual == label) / class_sizes[label]
                   for label in range(2)) / 2
    return {"mean_squared_error": str(squared_loss), "balanced_judgment_accuracy": str(accuracy)}


def timing_summary(values):
    ordered = sorted(values)
    return {"median_seconds": ordered[len(ordered) // 2], "p95_seconds": ordered[ceil(len(ordered) * .95) - 1], "maximum_seconds": max(ordered)}


def summarize(rows, configuration):
    development = [row for row in rows if row["split"] == "development"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    dimension = configuration["node_count"] - 1
    threshold = Fraction(configuration["connectivity_threshold"])
    development_mean = [sum(Fraction(row["targets"][node]) for row in development) / len(development) for node in range(dimension)]
    means = [sum(Fraction(row["targets"][node]) for row in evaluation) / len(evaluation) for node in range(dimension)]
    labels = [row["hazard"] for row in evaluation]
    sizes = [labels.count(label) for label in range(2)]
    assert min(sizes) > 0
    probability = Fraction(sizes[1], len(evaluation))
    class_means = [[sum(Fraction(row["targets"][node]) for row in evaluation if row["hazard"] == label) / sizes[label]
                    for node in range(dimension)] for label in range(2)]
    difference = [right - left for left, right in zip(*class_means)]
    upper_projection = sum(max(-direction * center, direction * (1 - center)) for direction, center in zip(difference, means))
    lower_projection = sum(min(-direction * center, direction * (1 - center)) for direction, center in zip(difference, means))
    bound = max(upper_projection, -lower_projection)
    between = probability * (1 - probability) * sum(value ** 2 for value in difference) / dimension
    variance = sum(sum((Fraction(row["targets"][node]) - means[node]) ** 2 for node in range(dimension)) / dimension for row in evaluation) / len(evaluation)
    floor = between - 8 * bound * probability * (1 - probability) * (Fraction(configuration["reader_advantage_cap"]) + Fraction(1, 1 << configuration["reader_bits"])) / dimension
    solvers = {}
    solvers["development_mean"] = measure_rows(evaluation, [development_mean] * len(evaluation), [Fraction(development_mean[-1] <= threshold)] * len(evaluation))
    for name in ["path", "cut", "midpoint"]:
        predictions = [list(map(Fraction, row[name])) for row in evaluation]
        solvers[name] = measure_rows(evaluation, predictions, [Fraction(prediction[-1] <= threshold) for prediction in predictions])
        timing_field = "path_seconds" if name == "path" else "bounds_seconds"
        solvers[name]["timing"] = timing_summary([row[timing_field] for row in evaluation])
    certified = [Fraction(row["certified_alarm_probability"]) for row in evaluation]
    solvers["certified_reader"] = measure_rows(evaluation, [list(map(Fraction, row["midpoint"])) for row in evaluation], certified)
    solvers["certified_reader"]["timing"] = timing_summary([row["bounds_seconds"] for row in evaluation])
    for sample_count in configuration["sample_counts"]:
        sample_rows = [row["sampling"][str(sample_count)] for row in evaluation]
        predictions = [list(map(Fraction, row["forecast"])) for row in sample_rows]
        actual = measure_rows(evaluation, predictions, [Fraction(prediction[-1] <= threshold) for prediction in predictions])
        expected = measure_rows(evaluation, predictions, [Fraction(row["expected_alarm_probability"]) for row in sample_rows])
        actual["expected_balanced_accuracy"] = expected["balanced_judgment_accuracy"]
        actual["expected_mean_squared_error"] = str(sum(Fraction(row["expected_squared_error"]) for row in sample_rows) / len(evaluation))
        actual["timing"] = timing_summary([row["elapsed_seconds"] for row in sample_rows])
        solvers["sampling_" + str(sample_count)] = actual
    ceiling = Fraction(1, 2) + Fraction(configuration["reader_advantage_cap"])
    witnesses = [name for name, metrics in solvers.items() if "timing" in metrics
                 and metrics["timing"]["maximum_seconds"] <= configuration["reader_deadline_seconds"]
                 and Fraction(metrics.get("expected_balanced_accuracy", metrics["balanced_judgment_accuracy"])) > ceiling]
    return {"development_mean": list(map(str, development_mean)), "evaluation_class_counts": sizes,
            "evaluation_mean": list(map(str, means)), "evaluation_class_means": [list(map(str, values)) for values in class_means],
            "reference_constant_variance": str(variance), "between_class_variance": str(between),
            "projection_bound": str(bound), "one_call_loss_floor": str(floor), "sufficient_loss_fraction": str(floor / variance),
            "solvers": solvers, "public_reader_rejection_witnesses": witnesses,
            "severity_screen_passed": floor / variance >= Fraction(configuration["severe_loss_fraction"]),
            "absolute_removal_screen_passed": not witnesses}


def main():
    source = Path(__file__).resolve().parent
    if platform.node() != "charon" or not (source / "plan-frozen.md").is_file():
        raise RuntimeError("Requires Charon and a frozen plan")
    configuration = json.loads((source / "config.json").read_text())
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (output / "machine.json").write_text(json.dumps({"hostname": platform.node(), "python": sys.version,
        "platform": platform.platform(), "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    try:
        rows = []
        threshold = Fraction(configuration["connectivity_threshold"])
        for specification in generate_inputs(configuration):
            oracle_start = time.perf_counter()
            targets = exact_probabilities(specification, configuration)
            oracle_seconds = time.perf_counter() - oracle_start
            bounds_start = time.perf_counter()
            lower, relaxations = path_probabilities(specification, configuration)
            path_alarm = lower[-1] <= threshold
            path_seconds = time.perf_counter() - bounds_start
            upper, cut_checks = cut_probabilities(specification, configuration)
            midpoint = [(first + second) / 2 for first, second in zip(lower, upper)]
            certified = Fraction(1) if upper[-1] <= threshold else Fraction(0) if lower[-1] > threshold else Fraction(1, 2)
            bounds_seconds = time.perf_counter() - bounds_start
            sampling = {}
            if specification["split"] == "evaluation":
                for sample_count in configuration["sample_counts"]:
                    seed = configuration["sampling_seed"] + 10000 * specification["index"] + sample_count
                    sample_start = time.perf_counter()
                    forecast = sample_probabilities(specification, configuration, sample_count, seed)
                    sampled_alarm = forecast[-1] <= threshold
                    elapsed = time.perf_counter() - sample_start
                    sampling[str(sample_count)] = {"seed": seed, "forecast": list(map(str, forecast)), "elapsed_seconds": elapsed,
                        "expected_squared_error": str(sum(probability * (1 - probability) / sample_count for probability in targets) / len(targets)),
                        "expected_alarm_probability": str(alarm_probability(targets[-1], sample_count, threshold)),
                        "edge_draws": sample_count * len(specification["edges"])}
            rows.append({**specification, "targets": list(map(str, targets)), "hazard": int(targets[-1] <= threshold),
                         "oracle_seconds": oracle_seconds, "path": list(map(str, lower)), "cut": list(map(str, upper)),
                         "midpoint": list(map(str, midpoint)), "certified_alarm_probability": str(certified),
                         "path_seconds": path_seconds, "bounds_seconds": bounds_seconds,
                         "path_relaxations": relaxations, "cut_edge_checks": cut_checks, "sampling": sampling})
        certificate = {"scope": configuration["interpretation"], "instances": rows, "summary": summarize(rows, configuration)}
        (output / "certificate.json").write_text(json.dumps(certificate, indent=2) + "\n")
        reference = prepare_reference(configuration)
        summary = audit_certificate(certificate, configuration, reference)
        corruptions = reject_corruptions(certificate, configuration, reference)
        (output / "audit.json").write_text(json.dumps({"summary": summary, "rejected_corruptions": corruptions}, indent=2) + "\n")
        size = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if size >= configuration["output_limit_bytes"]:
            raise RuntimeError("Output budget exceeded")
        receipt = {"status": "PASS", **summary, "corruptions_rejected": len(corruptions),
                   "elapsed_seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "output_bytes_before_receipt": size, "interpretation": configuration["interpretation"]}
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        manifest = {str(path.relative_to(source.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for directory in [source, output] for path in directory.iterdir()
                    if path.is_file() and not path.name.startswith("._")}
        (output / "sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(receipt))
        print(json.dumps({"class_counts": certificate["summary"]["evaluation_class_counts"],
                          "severe_loss_fraction": float(Fraction(certificate["summary"]["sufficient_loss_fraction"])),
                          "public_reader_witnesses": certificate["summary"]["public_reader_rejection_witnesses"]}))
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"error": repr(error)}) + "\n")
        raise


if __name__ == "__main__":
    main()
