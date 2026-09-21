"""Exact partial-transport controls. Research execution is restricted to Charon."""

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from math import lcm
from pathlib import Path
import platform
import resource
import sys
import time

from audit import audit_certificate, reject_corruptions


def squared_distance(first, second):
    return sum((left - right) ** 2 for left, right in zip(first, second))


def expand_points(points, count):
    return [tuple(map(Fraction, point)) for point in points for _ in range(count // len(points))]


def solve_partial_assignments(costs):
    count = len(costs)
    states = {0: (Fraction(0), [])}
    for lower_index in range(count):
        following = dict(states)
        for mask, (cost, pairs) in states.items():
            for upper_index in range(count):
                if mask & (1 << upper_index):
                    continue
                next_mask = mask | (1 << upper_index)
                candidate = cost + costs[lower_index][upper_index]
                if next_mask not in following or candidate < following[next_mask][0]:
                    following[next_mask] = (candidate, pairs + [[lower_index, upper_index]])
        states = following
    return [min((entry for mask, entry in states.items() if mask.bit_count() == mass),
                key=lambda entry: entry[0]) for mass in range(count + 1)]


def finite_case(specification):
    count = lcm(len(specification["lower_points"]), len(specification["upper_points"]))
    lower = expand_points(specification["lower_points"], count)
    upper = expand_points(specification["upper_points"], count)
    probability = Fraction(specification["hazard_probability"])
    dimension = len(lower[0])
    mean = tuple(sum((1 - probability) * lower[index][coordinate] + probability * upper[index][coordinate]
                     for index in range(count)) / count for coordinate in range(dimension))
    baseline = sum((1 - probability) * squared_distance(point, mean) for point in lower) / count
    baseline += sum(probability * squared_distance(point, mean) for point in upper) / count
    costs = [[squared_distance(first, second) for second in upper] for first in lower]
    frontier = []
    for matched_count, (cost, pairs) in enumerate(solve_partial_assignments(costs)):
        retained = []
        for lower_index, upper_index in pairs:
            forecast = [(1 - probability) * lower[lower_index][coordinate]
                        + probability * upper[upper_index][coordinate] for coordinate in range(dimension)]
            retained.append({"lower_index": lower_index, "upper_index": upper_index,
                             "forecast": list(map(str, forecast))})
        for lower_index in set(range(count)) - {pair[0] for pair in pairs}:
            retained.append({"lower_index": lower_index, "upper_index": None,
                             "forecast": list(map(str, lower[lower_index]))})
        for upper_index in set(range(count)) - {pair[1] for pair in pairs}:
            retained.append({"lower_index": None, "upper_index": upper_index,
                             "forecast": list(map(str, upper[upper_index]))})
        variation = 1 - Fraction(matched_count, count)
        loss = probability * (1 - probability) * cost / count
        frontier.append({"matched_count": matched_count, "conditional_variation": str(variation),
                         "transport_cost": str(cost / count), "squared_loss": str(loss),
                         "retained_advantage_fraction": str(1 - loss / baseline),
                         "balanced_judgment_accuracy": str((1 + variation) / 2),
                         "ordinary_judgment_accuracy": str(variation + (1 - variation) * max(probability, 1 - probability)),
                         "states": retained})
    return {"name": specification["name"], "hazard_probability": str(probability), "expanded_count": count,
            "mean": list(map(str, mean)), "baseline_variance": str(baseline), "frontier": frontier}


def continuous_case(specification, variation, noise_amplitude):
    probability = Fraction(specification["hazard_probability"])
    lower_start, lower_end = map(Fraction, specification["lower_interval"])
    upper_start, upper_end = map(Fraction, specification["upper_interval"])
    lower_width = lower_end - lower_start
    upper_width = upper_end - upper_start
    gap = upper_start - lower_end
    mass = 1 - variation
    mean = (1 - probability) * (lower_start + lower_end) / 2 + probability * (upper_start + upper_end) / 2
    baseline = ((1 - probability) * lower_width ** 2 + probability * upper_width ** 2) / 12
    baseline += probability * (1 - probability) * ((upper_start + upper_end - lower_start - lower_end) / 2) ** 2
    transport = mass * (gap + (lower_width + upper_width) * mass / 2) ** 2
    transport += (upper_width - lower_width) ** 2 * mass ** 3 / 12
    loss = probability * (1 - probability) * transport
    return {"name": specification["name"], "conditional_variation": str(variation), "common_mass": str(mass),
            "mean": str(mean), "baseline_variance": str(baseline), "transport_cost": str(transport),
            "squared_loss": str(loss), "retained_advantage_fraction": str(1 - loss / baseline),
            "balanced_judgment_accuracy": str((1 + variation) / 2),
            "noisy_intact_loss": str(noise_amplitude ** 2),
            "noisy_edited_loss": str(loss + noise_amplitude ** 2),
            "noisy_baseline_loss": str(baseline + noise_amplitude ** 2)}


def main():
    source = Path(__file__).resolve().parent
    if platform.node() != "charon" or not (source / "plan-frozen.md").is_file():
        raise RuntimeError("Requires Charon and a frozen plan")
    configuration = json.loads((source / "config.json").read_text())
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    machine = {"hostname": platform.node(), "platform": platform.platform(), "python": sys.version,
               "executable": sys.executable, "started_utc": datetime.now(timezone.utc).isoformat()}
    (output / "machine.json").write_text(json.dumps(machine, indent=2) + "\n")
    try:
        certificate = {
            "scope": configuration["interpretation"],
            "finite": [finite_case(case) for case in configuration["finite_families"]],
            "continuous": [continuous_case(case, Fraction(cap), Fraction(configuration["independent_noise_amplitude"]))
                           for case in configuration["continuous_families"] for cap in configuration["conditional_variation_caps"]],
        }
        (output / "certificate.json").write_text(json.dumps(certificate, indent=2) + "\n")
        summary = audit_certificate(certificate, configuration)
        corruptions = reject_corruptions(certificate, configuration)
        (output / "audit.json").write_text(json.dumps({"summary": summary, "rejected_corruptions": corruptions}, indent=2) + "\n")
        size = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if size >= configuration["output_limit_bytes"]:
            raise RuntimeError("Output budget exceeded")
        receipt = {"status": "PASS", **summary, "corruptions_rejected": len(corruptions),
                   "elapsed_seconds": time.monotonic() - started, "output_bytes_before_receipt": size,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "interpretation": configuration["interpretation"]}
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        manifest = {str(path.relative_to(source.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for directory in [source, output] for path in directory.iterdir()
                    if path.is_file() and not path.name.startswith("._")}
        (output / "sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(receipt))
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"error": repr(error)}) + "\n")
        raise


if __name__ == "__main__":
    main()
