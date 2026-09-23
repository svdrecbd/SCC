"""Validate and measure recovery from signed and ordinal spatial relations."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import sys
import time
import numpy as np
from data import read_frame
from analyze import recovery_prefixes, integrated_loss


def clip_fraction(value):
    return min(Fraction(1), max(Fraction(0), value))


def exact_risk_gain(baseline, prediction, truth):
    boundaries = sorted({Fraction(0), Fraction(1), baseline, truth})
    loss = Fraction(0)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        threshold = (left+right)/2
        reference = Fraction(threshold < baseline)
        recovered = clip_fraction(reference+prediction-baseline)
        target = Fraction(threshold < truth)
        loss += (right-left)*(recovered-target)**2
    return abs(baseline-truth)-loss


def exact_validation():
    grid = [Fraction(index, 4) for index in range(5)]
    differences = [Fraction(index, 8) for index in range(-8, 9)]
    checks = 0
    reversed_failures = 0
    for first, second, first_baseline, second_baseline in product(grid, repeat=4):
        truth = first-second
        reference = first_baseline-second_baseline
        for prediction in differences:
            displacement = prediction-reference
            first_unclipped = first_baseline+displacement/2
            second_unclipped = second_baseline-displacement/2
            useful_gain = (reference-truth)**2-(prediction-truth)**2
            depth_gain = ((first_baseline-first)**2-(first_unclipped-first)**2
                          +(second_baseline-second)**2-(second_unclipped-second)**2)/2
            assert depth_gain == useful_gain/4
            protected_gain = (exact_risk_gain(first_baseline, clip_fraction(first_unclipped), first)
                              +exact_risk_gain(second_baseline, clip_fraction(second_unclipped), second))/2
            assert protected_gain >= useful_gain/4
            reversed_gain = (exact_risk_gain(first_baseline, clip_fraction(first_baseline-displacement/2), first)
                             +exact_risk_gain(second_baseline, clip_fraction(second_baseline+displacement/2), second))/2
            reversed_failures += int(reversed_gain < useful_gain/4)
            checks += 1
    assert reversed_failures > 0
    return {"exact_checks": checks, "reversed_sign_failures": reversed_failures}


def main(directory, root):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    exact = exact_validation()
    (directory / "exact_validation.json").write_text(json.dumps(exact, indent=2)+"\n")
    selected = json.loads((directory / "selection.json").read_text())
    count = configuration["calibration_scenes"]
    maximum = configuration["depth_range_metres"]
    minimum = configuration["valid_depth_minimum"]
    depths = np.stack([read_frame(root/"acquisition01", record)[1].astype(np.float64)
                       for record in selected])
    crop = np.zeros((480, 640), dtype=bool)
    row_start, row_end, column_start, column_end = configuration["crop"]
    crop[row_start:row_end, column_start:column_end] = True
    masks = crop[None] & (depths > minimum) & (depths < maximum) & np.isfinite(depths)
    calibration = depths[:count].copy()
    calibration_mean = float(calibration[masks[:count]].mean())
    valid_calibration = (calibration > minimum) & (calibration < maximum) & np.isfinite(calibration)
    calibration[~valid_calibration] = calibration_mean
    sorted_calibration = np.sort(calibration/maximum, axis=0)
    baseline_means = sorted_calibration.mean(axis=0)
    rows = []
    for index in range(len(selected)):
        prediction = np.load(root/f"inference{index//4+1:02d}"/f"prediction_{index:03d}.npy",
                             allow_pickle=False).astype(np.float64)
        prediction = np.clip(prediction, minimum, maximum)/maximum
        centered = prediction-prediction[crop].mean()
        valid = masks[index]
        locations = np.flatnonzero(valid)
        generator = np.random.default_rng(configuration["pair_seed"]+index)
        pairs = generator.choice(locations, size=(2, configuration["order_pairs"]), replace=True)
        first, second = pairs
        target_first = depths[index].ravel()[first]/maximum
        target_second = depths[index].ravel()[second]/maximum
        difference = prediction.ravel()[first]-prediction.ravel()[second]
        centered_difference = centered.ravel()[first]-centered.ravel()[second]
        assert np.max(np.abs(difference-centered_difference)) < 1e-12
        rows.append({"selection_index": index, "locations": pairs,
                     "first": target_first, "second": target_second,
                     "truth": target_first-target_second, "native": difference,
                     "centered": centered_difference, "sign": np.sign(difference),
                     "first_baseline": baseline_means.ravel()[first],
                     "second_baseline": baseline_means.ravel()[second],
                     "reference": baseline_means.ravel()[first]-baseline_means.ravel()[second]})
    calibration_started = time.perf_counter()
    signs = np.concatenate([row["sign"] for row in rows[:count]])
    targets = np.concatenate([row["truth"] for row in rows[:count]])
    references = np.concatenate([row["reference"] for row in rows[:count]])
    magnitude = float(np.clip(np.dot(signs, targets)/np.dot(signs, signs), 0, 1))
    coefficients = np.linalg.lstsq(np.stack((signs, references), axis=1), targets, rcond=None)[0]
    parameters = {"ordinal_magnitude": magnitude, "ordinal_reference_coefficients": coefficients.tolist(),
                  "calibration_seconds": time.perf_counter()-calibration_started,
                  "calibration_pairs": len(signs)}
    (directory / "reader_parameters.json").write_text(json.dumps(parameters, indent=2)+"\n")
    records = []
    for row in rows[count:]:
        first_calibration = sorted_calibration.reshape(count, -1)[:, row["locations"][0]]
        second_calibration = sorted_calibration.reshape(count, -1)[:, row["locations"][1]]
        first_prefixes = recovery_prefixes(row["first"], first_calibration)
        second_prefixes = recovery_prefixes(row["second"], second_calibration)
        predictions = {"public_reference": row["reference"], "native": row["native"],
                       "centered": row["centered"], "ordinal": magnitude*row["sign"],
                       "ordinal_with_reference": coefficients[0]*row["sign"]+coefficients[1]*row["reference"]}
        record = {"selection_index": row["selection_index"],
                  "group": selected[row["selection_index"]]["group"], "readers": {}}
        for name, prediction in predictions.items():
            displacement = prediction-row["reference"]
            first_unclipped = row["first_baseline"]+displacement/2
            second_unclipped = row["second_baseline"]-displacement/2
            first_mean = np.clip(first_unclipped, 0, 1)
            second_mean = np.clip(second_unclipped, 0, 1)
            useful_gain = (row["reference"]-row["truth"])**2-(prediction-row["truth"])**2
            unclipped_gain = ((row["first_baseline"]-row["first"])**2-(first_unclipped-row["first"])**2
                              +(row["second_baseline"]-row["second"])**2-(second_unclipped-row["second"])**2)/2
            assert np.max(np.abs(unclipped_gain-useful_gain/4)) < 1e-12
            first_recovery = integrated_loss(first_mean, row["first"], first_calibration, first_prefixes)
            second_recovery = integrated_loss(second_mean, row["second"], second_calibration, second_prefixes)
            protected_gain = (first_recovery["protected_gain"]+second_recovery["protected_gain"])/2
            assert protected_gain >= useful_gain.mean()/4-1e-10
            record["readers"][name] = {
                "contrast_mse_normalized": float(np.mean((prediction-row["truth"])**2)),
                "contrast_mse_metres_squared": float(np.mean((prediction-row["truth"])**2)*maximum**2),
                "contrast_gain_normalized": float(useful_gain.mean()),
                "protected_gain": protected_gain,
                "baseline_brier": (first_recovery["baseline_brier"]+second_recovery["baseline_brier"])/2,
                "recovered_brier": (first_recovery["recovered_brier"]+second_recovery["recovered_brier"])/2,
            }
        records.append(record)
    generator = np.random.default_rng(configuration["bootstrap_seed"])
    indices = generator.integers(0, len(records), (configuration["bootstrap_replicates"], len(records)))
    summaries = {}
    for name in records[0]["readers"]:
        summaries[name] = {metric: float(np.mean([record["readers"][name][metric] for record in records]))
                           for metric in records[0]["readers"][name]}
        for metric in ("contrast_gain_normalized", "protected_gain"):
            vector = np.array([record["readers"][name][metric] for record in records])
            summaries[name][metric+"_interval"] = np.quantile(vector[indices].mean(axis=1), [.025, .975]).tolist()
    (directory / "cases.jsonl").write_text("".join(json.dumps(record)+"\n" for record in records))
    summary = {"status": "complete", **exact, "evaluation_scenes": len(records),
               "pairs_per_scene": configuration["order_pairs"], "readers": summaries,
               "reader_parameters": parameters, "neural_training": False,
               "actual_all_reader_removal_established": False, "training_admitted": False,
               "wall_seconds": time.perf_counter()-started}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
