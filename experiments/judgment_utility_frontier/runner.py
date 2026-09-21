"""Exact Bayes judgment-error versus useful-accuracy frontier; see frozen LN-245.

All numerical computation belongs on Charon. This is a finite state-information
experiment, not a learned controller or a claim about destruction of a procedure.
"""
import argparse
import hashlib
import itertools
import json
import platform
import resource
import time
from pathlib import Path


def serialize(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def action_words(measure):
    lengths = range(5) if measure == "length_0_through_4" else [int(measure.split("_")[1])]
    return [list(word) for length in lengths for word in itertools.product((0, 1), repeat=length)]


def outputs(state, word, linked):
    phase, useful, hazard = state % 4, state // 4 % 2, state // 8
    for action in word:
        useful ^= action ^ (hazard if linked and phase == 0 and action == 1 else 0)
        phase = (phase + 1) % 4
    return [useful, phase // 2, phase % 2]


def signatures(family, words, score):
    result = []
    for state in range(16):
        values = [outputs(state, word, family == "linked") for word in words]
        if score == "useful_bit":
            result.append([value[0] for value in values])
        elif score == "scalar_outputs":
            result.append([item for value in values for item in value])
        else:
            result.append([value[0] + 2 * value[1] + 4 * value[2] for value in values])
    return result


def minimum_matchings(distances):
    size = len(distances)
    layers = [[None] * (1 << size) for _ in range(size + 1)]
    layers[0][0] = 0
    for row in range(size):
        for mask, previous in enumerate(layers[row]):
            if previous is None:
                continue
            current = layers[row + 1][mask]
            if current is None or previous < current:
                layers[row + 1][mask] = previous
            for column in range(size):
                if mask & (1 << column):
                    continue
                target = mask | (1 << column)
                candidate = previous + distances[row][column]
                current = layers[row + 1][target]
                if current is None or candidate < current:
                    layers[row + 1][target] = candidate
    solutions = []
    for cardinality in range(size + 1):
        cost, mask = min((cost, mask) for mask, cost in enumerate(layers[size])
                         if mask.bit_count() == cardinality and cost is not None)
        pairs = []
        for row in range(size, 0, -1):
            if layers[row - 1][mask] == layers[row][mask]:
                continue
            for column in range(size):
                if not mask & (1 << column):
                    continue
                previous = layers[row - 1][mask ^ (1 << column)]
                if previous is not None and previous + distances[row - 1][column] == layers[row][mask]:
                    pairs.append([row - 1, column + size])
                    mask ^= 1 << column
                    break
            else:
                raise AssertionError("No matching predecessor")
        pairs.sort()
        classes = [[state] for state in range(2 * size)
                   if all(state not in pair for pair in pairs)] + pairs
        solutions.append({"cardinality": cardinality, "distance_sum": cost,
                          "pairs": pairs, "classes": classes})
    return layers, solutions


def run(source, destination):
    started = time.monotonic()
    destination.mkdir(exist_ok=False)
    configuration = json.loads((source / "config.json").read_text())
    if not (source / "plan-frozen.md").is_file():
        raise RuntimeError("Missing frozen research plan")
    cases = []
    for family, measure, score in itertools.product(configuration["families"],
            configuration["query_measures"], configuration["scores"]):
        words = action_words(measure)
        predictions = signatures(family, words, score)
        query_count = len(predictions[0])
        distances = [[sum(left != right for left, right in zip(predictions[row], predictions[column + 8]))
                      for column in range(8)] for row in range(8)]
        layers, frontier = minimum_matchings(distances)
        for point in frontier:
            point["judgment_error"] = [point["cardinality"], 16]
            point["useful_error"] = [point["distance_sum"], 16 * query_count]
            point["retained_state_bits"] = (len(point["classes"]) - 1).bit_length()
        cases.append({"family": family, "measure": measure, "score": score,
                      "words": words, "predictions": predictions, "query_count": query_count,
                      "distances": distances, "matching_layers": layers, "frontier": frontier})
    serialize(destination / "certificate.json", {"configuration": configuration, "cases": cases})
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source.iterdir()) if path.is_file() and not path.name.startswith("._")}
    serialize(destination / "receipt.json", {
        "schema": 1, "case_count": len(cases), "frontier_endpoint_count": sum(len(case["frontier"]) for case in cases),
        "host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
        "wall_seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source_hashes": source_hashes,
        "certificate_sha256": hashlib.sha256((destination / "certificate.json").read_bytes()).hexdigest(),
        "limits": "All stochastic encodings of the stated finite source, unrestricted readout, no fresh state information; not arbitrary online controllers."
    })
    total_bytes = sum(path.stat().st_size for path in destination.iterdir() if path.is_file())
    if total_bytes > configuration["output_limit_bytes"]:
        raise RuntimeError("Output exceeds frozen limit")
    print(json.dumps({"cases": len(cases), "output_bytes": total_bytes, "wall_seconds": time.monotonic() - started}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    run(Path(__file__).resolve().parent, arguments.output)
