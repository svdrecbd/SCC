"""Whole-trajectory judgment-removal certificates for the frozen LN-248 contract."""
import argparse
import hashlib
import json
import platform
import resource
import time
from fractions import Fraction
from pathlib import Path

from matching_reference import minimum_matchings
from parent_models import transition


def source_hashes(source):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.iterdir()) if path.is_file() and not path.name.startswith("._")}


def run(source, destination):
    started = time.monotonic()
    configuration = json.loads((source / "config.json").read_text())
    if not (source / "plan-frozen.md").is_file():
        raise ValueError("Frozen research plan is required")
    destination.mkdir(exist_ok=False)
    families = []
    for family in configuration["families"]:
        linked = family == "linked"
        agreement = [[[int(first % 8 == second % 8) for second in range(16)] for first in range(16)]]
        for horizon in range(1, max(configuration["horizons"]) + 1):
            previous = agreement[-1]
            agreement.append([[sum(previous[transition(first, action, linked)][transition(second, action, linked)]
                                       for action in range(2)) if first % 8 == second % 8 else 0
                               for second in range(16)] for first in range(16)])
        cases = []
        for horizon in configuration["horizons"]:
            word_count = 1 << horizon
            distances = [[word_count - agreement[horizon][first][second + 8] for second in range(8)] for first in range(8)]
            layers, frontier = minimum_matchings(distances)
            for point in frontier:
                point["judgment_error"] = [point["cardinality"], 16]
                point["trajectory_error"] = [point["distance_sum"], 16 * word_count]
            # No world observations: prediction is run with the discarded h set to zero.
            no_observation_accuracy = Fraction(1, 2) + Fraction(sum(agreement[horizon][state][state + 8] for state in range(8)), 16 * word_count)
            # With real useful-state feedback, the first exposing action identifies h.
            ambiguity = Fraction(sum(agreement[horizon][state][state + 8] for state in range(8)), 8 * word_count)
            feedback_judgment_accuracy = 1 - ambiguity / 2 if linked else Fraction(1, 2)
            feedback_bit_accuracy = 1 - (1 - ambiguity) / (2 * (horizon + 1)) if linked else Fraction(1)
            cases.append({"horizon": horizon, "word_count": word_count, "distances": distances,
                          "matching_layers": layers, "frontier": frontier,
                          "no_observation_trajectory_accuracy": str(no_observation_accuracy),
                          "feedback_judgment_accuracy": str(feedback_judgment_accuracy),
                          "feedback_useful_bit_accuracy": str(feedback_bit_accuracy)})
        families.append({"family": family, "agreement_counts": agreement, "cases": cases})
    certificate = {"configuration": configuration, "families": families}
    encoded = (json.dumps(certificate, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (destination / "certificate.json").write_bytes(encoded)
    receipt = {"host": platform.node(), "python": platform.python_version(), "platform": platform.platform(),
               "case_count": sum(len(family["cases"]) for family in families),
               "source_hashes": source_hashes(source), "certificate_sha256": hashlib.sha256(encoded).hexdigest(),
               "wall_seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "scope": "Exact complete-trajectory predictions and separately charged world-feedback restoration; no learned procedure deletion."}
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    total = sum(path.stat().st_size for path in destination.iterdir() if path.is_file())
    if total > configuration["output_limit_bytes"]:
        raise ValueError("Frozen output budget exceeded")
    print(json.dumps({"case_count": receipt["case_count"], "output_bytes": total, "wall_seconds": receipt["wall_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(Path(__file__).resolve().parent, arguments.output)
