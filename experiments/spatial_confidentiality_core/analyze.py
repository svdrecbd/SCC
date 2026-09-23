"""Compare spatial utility, public replacements and exact judgment recovery."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from PIL import Image
from data import read_frame


def features(rgb, coordinates):
    horizontal, vertical = coordinates
    red, green, blue = np.asarray(rgb, dtype=np.float64).transpose(2, 0, 1) / 255
    values = [np.ones_like(horizontal), horizontal, vertical, horizontal**2,
              horizontal * vertical, vertical**2, red, green, blue,
              red**2, green**2, blue**2, red*green, red*blue, green*blue,
              horizontal*red, horizontal*green, horizontal*blue,
              vertical*red, vertical*green, vertical*blue]
    return np.stack(values, axis=-1)


def thumbnail(rgb):
    return np.asarray(Image.fromarray(rgb).resize((16, 12), Image.Resampling.BILINEAR),
                      dtype=np.float64).reshape(-1) / 255


def utility_metrics(prediction, truth, pair_indices):
    difference = prediction - truth
    log_difference = np.log(prediction) - np.log(truth)
    left, right = pair_indices
    difference_truth = truth[left] - truth[right]
    difference_prediction = prediction[left] - prediction[right]
    order = np.where(difference_prediction == 0, 0.5,
                     np.sign(difference_prediction) == np.sign(difference_truth))
    return {"mse": float(np.mean(difference**2)),
            "absolute_relative_error": float(np.mean(np.abs(difference) / truth)),
            "ratio_accuracy": float(np.mean(np.maximum(prediction/truth, truth/prediction) < 1.25)),
            "centred_log_error": float(np.var(log_difference)),
            "order_accuracy": float(np.mean(order)) if len(order) else None}


def direct_integrated_loss(prediction, truth, sorted_calibration):
    count = len(sorted_calibration)
    baseline_mean = sorted_calibration.mean(axis=0)
    shift = prediction - baseline_mean
    baseline_loss = np.zeros_like(truth)
    recovered_loss = np.zeros_like(truth)
    left = np.zeros_like(truth)
    for interval in range(count + 1):
        right = sorted_calibration[interval] if interval < count else np.ones_like(truth)
        width = right - left
        positive_width = np.clip(truth - left, 0, width)
        negative_width = width - positive_width
        baseline_probability = (count - interval) / count
        recovered_probability = np.clip(baseline_probability + shift, 0, 1)
        baseline_loss += positive_width*(1-baseline_probability)**2 + negative_width*baseline_probability**2
        recovered_loss += positive_width*(1-recovered_probability)**2 + negative_width*recovered_probability**2
        left = right
    return baseline_loss, recovered_loss


def recovery_prefixes(truth, sorted_calibration):
    count = len(sorted_calibration)
    boundaries = np.concatenate((np.zeros((1, len(truth))), sorted_calibration,
                                 np.ones((1, len(truth)))), axis=0)
    widths = np.diff(boundaries, axis=0)
    positive_widths = np.clip(truth[None]-boundaries[:-1], 0, widths)
    probabilities = (1-np.arange(count+1)/count)[:, None]
    def prefix(values):
        return np.concatenate((np.zeros((1, len(truth))), np.cumsum(values, axis=0)), axis=0)
    return {"width": prefix(widths), "first": prefix(widths*probabilities),
            "second": prefix(widths*probabilities**2),
            "positive_width": prefix(positive_widths),
            "positive_first": prefix(positive_widths*probabilities),
            "mean": sorted_calibration.mean(axis=0), "count": count}


def integrated_loss(prediction, truth, sorted_calibration, prefixes):
    count = prefixes["count"]
    baseline_mean = prefixes["mean"]
    shift = prediction-baseline_mean
    lower = np.clip(np.floor(count*shift).astype(int)+1, 0, count+1)
    upper = np.clip(np.ceil(count*(1+shift)).astype(int), 0, count+1)
    columns = np.arange(len(truth))
    def selected(name, indices):
        return prefixes[name][indices, columns]
    square_integral = (selected("width", lower)
                       + selected("second", upper)-selected("second", lower)
                       + 2*shift*(selected("first", upper)-selected("first", lower))
                       + shift**2*(selected("width", upper)-selected("width", lower)))
    positive_integral = (selected("positive_width", lower)
                         + selected("positive_first", upper)-selected("positive_first", lower)
                         + shift*(selected("positive_width", upper)-selected("positive_width", lower)))
    baseline_loss = prefixes["second"][-1]-2*prefixes["positive_first"][-1]+truth
    recovered_loss = square_integral-2*positive_integral+truth
    checks = np.linspace(0, len(truth)-1, min(64, len(truth)), dtype=int)
    direct_baseline, direct_recovered = direct_integrated_loss(prediction[checks], truth[checks], sorted_calibration[:, checks])
    assert np.max(np.abs(direct_baseline-baseline_loss[checks])) < 1e-12
    assert np.max(np.abs(direct_recovered-recovered_loss[checks])) < 1e-12
    useful_gain = (baseline_mean - truth)**2 - (prediction - truth)**2
    protected_gain = baseline_loss - recovered_loss
    minimum_slack = float(np.min(protected_gain - useful_gain))
    assert minimum_slack >= -1e-10, minimum_slack
    return {"baseline_brier": float(baseline_loss.mean()),
            "recovered_brier": float(recovered_loss.mean()),
            "point_depth_brier": float(np.mean(np.abs(prediction-truth))),
            "useful_gain": float(useful_gain.mean()),
            "protected_gain": float(protected_gain.mean()),
            "minimum_pointwise_slack": minimum_slack}


def main(directory, root):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    selection = json.loads((directory / "selection.json").read_text())
    acquisition = root / "acquisition01"
    rgb_images, depths = [], []
    for record in selection:
        rgb, depth, _ = read_frame(acquisition, record)
        rgb_images.append(rgb)
        depths.append(depth.astype(np.float64))
    depths = np.stack(depths)
    row_start, row_end, column_start, column_end = configuration["crop"]
    crop = np.zeros((480, 640), dtype=bool)
    crop[row_start:row_end, column_start:column_end] = True
    maximum = configuration["depth_range_metres"]
    minimum = configuration["valid_depth_minimum"]
    masks = crop[None] & (depths > minimum) & (depths < maximum) & np.isfinite(depths)
    count = configuration["calibration_scenes"]
    calibration = depths[:count].copy()
    global_mean = float(np.mean(calibration[masks[:count]]))
    imputed = ~((calibration > minimum) & (calibration < maximum) & np.isfinite(calibration))
    calibration[imputed] = global_mean
    spatial_mean = calibration.mean(axis=0)
    sorted_calibration = np.sort(calibration / maximum, axis=0)
    coordinates = np.meshgrid(np.linspace(-1, 1, 640), np.linspace(-1, 1, 480))
    calibration_features, calibration_targets = [], []
    generator = np.random.default_rng(configuration["selection_seed"])
    preparation_started = time.perf_counter()
    for index in range(count):
        valid = np.flatnonzero(masks[index])
        positions = generator.choice(valid, size=min(configuration["calibration_pixels_per_scene"], len(valid)), replace=False)
        calibration_features.append(features(rgb_images[index], coordinates).reshape(-1, 21)[positions])
        calibration_targets.append(depths[index].ravel()[positions])
    design = np.concatenate(calibration_features)
    target = np.concatenate(calibration_targets)
    position_coefficients = np.linalg.lstsq(design[:, :6], target, rcond=None)[0]
    regularizer = np.eye(design.shape[1]) * configuration["ridge_regularization"]
    regularizer[0, 0] = 0
    rgb_coefficients = np.linalg.solve(design.T @ design / len(design) + regularizer, design.T @ target / len(design))
    thumbnails = np.stack([thumbnail(rgb) for rgb in rgb_images[:count]])
    quantile_positions = np.linspace(0, 1, 4097)
    quantile_values = np.quantile(calibration[masks[:count]], quantile_positions)
    preparation_seconds = time.perf_counter() - preparation_started
    np.savez(directory / "public_parameters.npz", global_mean=global_mean, spatial_mean=spatial_mean,
             position_coefficients=position_coefficients, rgb_coefficients=rgb_coefficients,
             thumbnails=thumbnails, quantile_positions=quantile_positions, quantile_values=quantile_values)
    public_names = ("global_mean", "spatial_mean", "quadratic_position", "position_rgb", "nearest_image", "three_nearest_images")
    records = []
    expected_digest = None
    for index in range(count, len(selection)):
        inference = root / f"inference{index//4+1:02d}"
        inference_summary = json.loads((inference / "summary.json").read_text())
        assert inference_summary["status"] == "complete"
        digest = inference_summary["state_sha256_after"]
        assert digest == inference_summary["state_sha256_before"]
        expected_digest = digest if expected_digest is None else expected_digest
        assert digest == expected_digest
        prediction_raw = np.load(inference / f"prediction_{index:03d}.npy", allow_pickle=False).astype(np.float64)
        prediction = np.clip(prediction_raw, minimum, maximum)
        feature_array = features(rgb_images[index], coordinates)
        distances = np.mean((thumbnails-thumbnail(rgb_images[index]))**2, axis=1)
        neighbors = np.argsort(distances, kind="stable")[:3]
        weights = 1/(distances[neighbors]+1e-8)
        weights /= weights.sum()
        sorted_prediction = np.sort(prediction[crop])
        lower_rank = np.searchsorted(sorted_prediction, prediction, side="left")
        upper_rank = np.searchsorted(sorted_prediction, prediction, side="right")
        rank = (lower_rank+upper_rank)/(2*len(sorted_prediction))
        affine = 0.5*prediction + 1
        restored = 2*(affine-1)
        assert np.max(np.abs(restored-prediction)) < 1e-12
        predictors = {
            "native": prediction,
            "global_mean": np.full_like(prediction, global_mean),
            "spatial_mean": spatial_mean,
            "quadratic_position": feature_array[:, :, :6] @ position_coefficients,
            "position_rgb": feature_array @ rgb_coefficients,
            "nearest_image": calibration[neighbors[0]],
            "three_nearest_images": np.einsum("n,nhw->hw", weights, calibration[neighbors]),
            "affine": affine,
            "affine_restored": restored,
            "mean_normalized": prediction * global_mean / prediction[crop].mean(),
            "rank_quantile": np.interp(rank, quantile_positions, quantile_values),
        }
        mask = masks[index]
        truth = depths[index][mask]
        local_generator = np.random.default_rng(configuration["selection_seed"] + index)
        pairs = local_generator.integers(0, len(truth), (2, configuration["order_pairs"]))
        separated = np.abs(truth[pairs[0]]-truth[pairs[1]]) > configuration["order_minimum_separation_metres"]
        pairs = pairs[:, separated]
        scene_record = {"selection_index": index, "scene": selection[index]["scene"],
                        "group": selection[index]["group"], "valid_pixels": int(mask.sum()),
                        "excluded_pixels": int(mask.size-mask.sum()), "evaluated_order_pairs": int(separated.sum()),
                        "native_clipping_fraction": float(np.mean(prediction_raw[mask] != prediction[mask])),
                        "readers": {}}
        calibration_at_pixels = sorted_calibration[:, mask]
        prefixes = recovery_prefixes(truth/maximum, calibration_at_pixels)
        for name, values in predictors.items():
            values = np.clip(values[mask], minimum, maximum)
            scene_record["readers"][name] = utility_metrics(values, truth, pairs)
            scene_record["readers"][name]["recovery"] = integrated_loss(values/maximum, truth/maximum, calibration_at_pixels, prefixes)
        records.append(scene_record)
        (directory / "cases.jsonl").write_text("".join(json.dumps(record)+"\n" for record in records))
        print(json.dumps({"selection_index": index, "native_mse": scene_record["readers"]["native"]["mse"]}), flush=True)
    readers = {}
    for name in records[0]["readers"]:
        readers[name] = {metric: float(np.mean([record["readers"][name][metric] for record in records]))
                         for metric in ("mse", "absolute_relative_error", "ratio_accuracy", "centred_log_error", "order_accuracy")}
        readers[name]["recovery"] = {metric: float(np.mean([record["readers"][name]["recovery"][metric] for record in records]))
                                     for metric in records[0]["readers"][name]["recovery"]}
    native_errors = np.array([record["readers"]["native"]["mse"] for record in records])
    public_errors = np.array([[record["readers"][name]["mse"] for name in public_names] for record in records])
    generator = np.random.default_rng(configuration["bootstrap_seed"])
    indices = generator.integers(0, len(records), (configuration["bootstrap_replicates"], len(records)))
    advantages = public_errors[indices].mean(axis=1).min(axis=1)-native_errors[indices].mean(axis=1)
    interval = np.quantile(advantages, [.025, .975]).tolist()
    best_public = min(public_names, key=lambda name: readers[name]["mse"])
    reduction = 1-readers["native"]["mse"]/readers[best_public]["mse"]
    capability_pass = (reduction >= configuration["minimum_mse_reduction"] and interval[0] > 0
                       and readers["native"]["order_accuracy"] >= configuration["minimum_order_accuracy"])
    summary = {"status": "complete", "evaluation_scenes": len(records), "readers": readers,
               "best_public_reader": best_public, "relative_mse_reduction": reduction,
               "selection_aware_mse_advantage_interval": interval, "development_capability_pass": capability_pass,
               "public_preparation_seconds": preparation_seconds,
               "reference_maps_retained_bytes": int(calibration.nbytes),
               "invalid_calibration_values_imputed": int(imputed.sum()),
               "parameter_count": inference_summary["parameter_count"], "state_sha256": expected_digest,
               "actual_all_reader_removal_established": False, "training_admitted": False,
               "neural_training": False, "wall_seconds": time.perf_counter()-started}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "readers"}))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
