"""Budgeted forecast-to-judgment controls, executed only on Charon."""

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import resource
import sys
import time

from audit import audit_certificate, reject_corruptions


def scalar_parameters(probability, targets):
    lower_mean = sum(targets[:2]) / 2
    upper_mean = sum(targets[2:]) / 2
    difference = upper_mean - lower_mean
    mean = (1 - probability) * lower_mean + probability * upper_mean
    bound = difference * max(mean, 1 - mean)
    weights = [(1 - probability) / 2] * 2 + [probability / 2] * 2
    residuals = [target - mean - difference * (int(index >= 2) - probability)
                 for index, target in enumerate(targets)]
    within = sum(weight * residual ** 2 for weight, residual in zip(weights, residuals))
    between = probability * (1 - probability) * difference ** 2
    return mean, difference, bound, weights, residuals, within, between


def evaluate_forecasts(forecasts, parameters, targets, bits):
    mean, difference, bound, weights, residuals, within, between = parameters
    probabilities = [Fraction(1, 2) + difference * (forecast - mean) / (2 * bound) for forecast in forecasts]
    coin_counts = [int(probability * (1 << bits)) for probability in probabilities]
    ideal_accuracy = (sum(1 - value for value in probabilities[:2]) + sum(probabilities[2:])) / 4
    integer_accuracy = Fraction(sum((1 << bits) - value for value in coin_counts[:2]) + sum(coin_counts[2:]), 4 * (1 << bits))
    loss = sum(weight * (target - forecast) ** 2 for weight, target, forecast in zip(weights, targets, forecasts))
    remainder = sum(weight * (residual - (forecast - mean)) ** 2
                    for weight, residual, forecast in zip(weights, residuals, forecasts))
    return {"forecasts": list(map(str, forecasts)), "squared_loss": str(loss),
            "reader_probabilities": list(map(str, probabilities)), "reader_coin_counts": coin_counts,
            "ideal_balanced_accuracy": str(ideal_accuracy), "finite_balanced_accuracy": str(integer_accuracy),
            "square_remainder": str(remainder)}


def scalar_case(probability, configuration):
    targets = list(map(Fraction, configuration["scalar_targets"]))
    parameters = scalar_parameters(probability, targets)
    mean, difference, bound, weights, residuals, within, between = parameters
    grid = list(map(Fraction, configuration["forecast_grid"]))
    tables = [evaluate_forecasts(list(forecasts), parameters, targets, configuration["reader_bits"])
              for forecasts in product(grid, repeat=len(targets))]
    controls = []
    for name, forecasts, states in [
        ("intact", targets, [0, 1, 2, 3]),
        ("native_judgment_deleted", targets, [0, 1, 2, 3]),
        ("constant_mean", [mean] * 4, [0, 0, 0, 0]),
        ("within_class_coordinate", [mean + residual for residual in residuals], [0, 1, 0, 1]),
    ]:
        row = evaluate_forecasts(forecasts, parameters, targets, configuration["reader_bits"])
        native = [0] * 4 if name == "native_judgment_deleted" else [int(forecast > Fraction(1, 2)) for forecast in forecasts]
        row.update(name=name, retained_states=states,
                   native_balanced_accuracy=str(Fraction(sum(1 - value for value in native[:2]) + sum(native[2:]), 4)))
        controls.append(row)
    probability_product = probability * (1 - probability)
    cap = Fraction(configuration["judgment_advantage_cap"])
    return {"hazard_probability": str(probability), "mean": str(mean), "class_mean_difference": str(difference),
            "projection_bound": str(bound), "within_variance": str(within), "between_variance": str(between),
            "baseline_variance": str(within + between),
            "finite_reader_loss_floor": str(between - 8 * bound * probability_product * (cap + Fraction(1, 1 << configuration["reader_bits"]))),
            "tables": tables, "controls": controls}


def vector_cases(configuration):
    targets = list(map(Fraction, configuration["scalar_targets"]))
    grid = list(map(Fraction, configuration["forecast_grid"]))
    rows = []
    mean = (Fraction(1, 2), Fraction(1, 2))
    difference = (Fraction(4, 5), Fraction(0))
    for index, target in enumerate(targets):
        hazard = int(index >= 2)
        for nuisance in map(Fraction, configuration["vector_second_coordinate"]):
            point = (target, nuisance)
            residual = tuple(value - center - direction * (hazard - Fraction(1, 2))
                             for value, center, direction in zip(point, mean, difference))
            for forecast in product(grid, repeat=2):
                displacement = tuple(value - center for value, center in zip(forecast, mean))
                gain = sum((value - center) ** 2 - (value - prediction) ** 2
                           for value, center, prediction in zip(point, mean, forecast))
                projection = sum(direction * value for direction, value in zip(difference, displacement))
                remainder = sum((noise - value) ** 2 for noise, value in zip(residual, displacement))
                rows.append({"source_index": index, "nuisance": str(nuisance), "forecast": list(map(str, forecast)),
                             "gain": str(gain), "projection": str(projection), "remainder": str(remainder)})
    return rows


def public_sensor_cases(configuration):
    targets = list(map(Fraction, configuration["scalar_targets"]))
    rows = []
    for error in map(Fraction, configuration["sensor_errors"]):
        records = []
        for index, target in enumerate(targets):
            hazard = int(index >= 2)
            rank = index % 2
            for observation in range(2):
                weight = (1 - error if observation == hazard else error) / 4
                posterior = 1 - error if observation else error
                public_forecast = Fraction(1, 10) + Fraction(4, 5) * posterior
                edited_forecast = public_forecast + (Fraction(1, 20) if rank else -Fraction(1, 20))
                records.append({"hazard": hazard, "rank": rank, "observation": observation,
                                "weight": str(weight), "target": str(target),
                                "public_forecast": str(public_forecast), "edited_forecast": str(edited_forecast)})
        public_loss = sum(Fraction(row["weight"]) * (Fraction(row["target"]) - Fraction(row["public_forecast"])) ** 2 for row in records)
        edited_loss = sum(Fraction(row["weight"]) * (Fraction(row["target"]) - Fraction(row["edited_forecast"])) ** 2 for row in records)
        rows.append({"sensor_error": str(error), "public_balanced_accuracy": str(1 - error),
                     "public_baseline_loss": str(public_loss), "conditional_removal_loss": str(edited_loss),
                     "absolute_removal_cap_feasible_from_public_input": 1 - error <= Fraction(1, 2) + Fraction(configuration["judgment_advantage_cap"]),
                     "records": records})
    return rows


def vector_retention_control(configuration):
    records = []
    for index, target in enumerate(map(Fraction, configuration["scalar_targets"])):
        for nuisance in map(Fraction, configuration["vector_second_coordinate"]):
            forecast = Fraction(1, 2) + (Fraction(1, 20) if index % 2 else -Fraction(1, 20))
            records.append({"source_index": index, "nuisance": str(nuisance),
                            "retained_state": [index % 2, str(nuisance)],
                            "forecast": [str(forecast), str(nuisance)]})
    return records


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
        certificate = {"scope": configuration["interpretation"],
                       "scalar": [scalar_case(Fraction(probability), configuration) for probability in configuration["scalar_probabilities"]],
                       "vector": vector_cases(configuration), "vector_retention": vector_retention_control(configuration),
                       "public_sensors": public_sensor_cases(configuration),
                       "masked_bit": [{"hazard": hazard, "retained_bit": retained, "public_bit": hazard ^ retained}
                                      for hazard in range(2) for retained in range(2)]}
        (output / "certificate.json").write_text(json.dumps(certificate, indent=2) + "\n")
        summary = audit_certificate(certificate, configuration)
        corruptions = reject_corruptions(certificate, configuration)
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
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"error": repr(error)}) + "\n")
        raise


if __name__ == "__main__":
    main()
