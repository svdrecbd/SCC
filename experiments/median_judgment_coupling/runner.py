"""Exact arithmetic controls for median-judgment coupling; execute on Charon."""

from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time

from audit import audit_certificate, reject_corruptions


def serialize_fraction(value):
    return str(Fraction(value))


def measure_forecasts(name, predictions, states, target_count):
    absolute_loss = Fraction(0)
    squared_loss = Fraction(0)
    reader_accuracy = Fraction(0)
    native_correct = 0
    classes = defaultdict(lambda: [0, 0])
    for index, prediction in enumerate(predictions):
        target = Fraction(2 * index + 1, 2 * target_count)
        judgment = int(target > Fraction(1, 2))
        absolute_loss += abs(target - prediction) / target_count
        squared_loss += (target - prediction) ** 2 / target_count
        reader_accuracy += (prediction if judgment else 1 - prediction) / target_count
        native = 0 if name == "deleted_judgment_output" else int(prediction > Fraction(1, 2))
        native_correct += native == judgment
        classes[str(states[index])][judgment] += 1
    optimal_accuracy = Fraction(sum(max(counts) for counts in classes.values()), target_count)
    return {
        "name": name,
        "forecasts": list(map(serialize_fraction, predictions)),
        "retained_states": list(map(str, states)),
        "absolute_loss": serialize_fraction(absolute_loss),
        "squared_loss": serialize_fraction(squared_loss),
        "useful_gain": serialize_fraction(Fraction(1, 4) - absolute_loss),
        "randomized_judgment_accuracy": serialize_fraction(reader_accuracy),
        "native_judgment_accuracy": serialize_fraction(Fraction(native_correct, target_count)),
        "optimal_state_judgment_accuracy": serialize_fraction(optimal_accuracy),
    }


def public_key_recovery(prime, generator, public_key, order):
    value = 1
    for exponent in range(order):
        if value == public_key:
            return {"recovered_exponent": exponent, "comparisons": exponent + 1,
                    "modular_multiplications": exponent}
        value = value * generator % prime
    raise ValueError("public key not in the declared subgroup")


def construct_certificate(configuration):
    target_count = configuration["target_count"]
    denominator = configuration["forecast_denominator"]
    points = []
    for index in range(target_count):
        target = Fraction(2 * index + 1, 2 * target_count)
        sign = 1 if target > Fraction(1, 2) else -1
        for numerator in range(denominator + 1):
            prediction = Fraction(numerator, denominator)
            gain = abs(target - Fraction(1, 2)) - abs(target - prediction)
            advantage = (prediction - Fraction(1, 2)) * sign
            points.append({"target_index": index, "forecast_numerator": numerator,
                           "useful_gain": str(gain), "judgment_advantage": str(advantage)})

    targets = [Fraction(2 * index + 1, 2 * target_count) for index in range(target_count)]
    half = Fraction(1, 2)
    forecast_tables = {
        "intact": targets,
        "deleted_judgment_output": targets,
        "median": [half] * target_count,
        "inverted": [1 - value for value in targets],
        "quantized": [Fraction(4 * (index // 4) + 2, target_count)
                      for index in range(target_count)],
        "boundary_reversal": [
            value if index < target_count // 4 or index >= 3 * target_count // 4
            else half + (Fraction(1, denominator) if value < half else -Fraction(1, denominator))
            for index, value in enumerate(targets)],
        "paired_state": [Fraction(2 * (index % (target_count // 2)) + 1, 2 * target_count)
                         + Fraction(1, 4) for index in range(target_count)],
        "extreme_recovery": [value if index < target_count // 8 or index >= 7 * target_count // 8
                             else half for index, value in enumerate(targets)],
    }
    tables = []
    for name, predictions in forecast_tables.items():
        states = ([index % (target_count // 2) for index in range(target_count)]
                  if name == "paired_state" else predictions)
        tables.append(measure_forecasts(name, predictions, states, target_count))

    encryption = []
    key_recoveries = []
    for prime, order, generator in configuration["group_parameters"]:
        for secret_exponent in range(1, order):
            public_key = pow(generator, secret_exponent, prime)
            recovery = public_key_recovery(prime, generator, public_key, order)
            key_recoveries.append({"prime": prime, "order": order, "generator": generator,
                                   "public_key": public_key, **recovery})
            for bit in range(2):
                for randomness in range(order):
                    first = pow(generator, randomness, prime)
                    second = pow(public_key, randomness, prime) * (generator if bit else 1) % prime
                    message = second * pow(pow(first, secret_exponent, prime), -1, prime) % prime
                    recovered_message = second * pow(pow(first, recovery["recovered_exponent"], prime), -1, prime) % prime
                    encryption.append({"prime": prime, "order": order, "generator": generator,
                                       "public_key": public_key, "secret_exponent": secret_exponent,
                                       "bit": bit, "randomness": randomness,
                                       "ciphertext": [first, second], "message_element": message,
                                       "public_recovery_message_element": recovered_message})
    return {"scope": configuration["interpretation"], "pointwise": points, "forecast_tables": tables,
            "encryption_cases": encryption, "public_key_recoveries": key_recoveries,
            "baseline_absolute_loss": "1/4", "removal_advantage_cap": "1/100",
            "minimum_absolute_loss": "6/25", "maximum_retained_advantage_fraction": "1/25"}


def main():
    source = Path(__file__).resolve().parent
    output = Path(sys.argv[1]).resolve()
    if platform.node() != "charon" or not (source / "plan-frozen.md").is_file():
        raise RuntimeError("Requires Charon and a frozen plan")
    configuration = json.loads((source / "config.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    machine = {"hostname": platform.node(), "platform": platform.platform(), "python": sys.version,
               "executable": sys.executable, "started_utc": datetime.now(timezone.utc).isoformat()}
    (output / "machine.json").write_text(json.dumps(machine, indent=2) + "\n")
    try:
        certificate = construct_certificate(configuration)
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
