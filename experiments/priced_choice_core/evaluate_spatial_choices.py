"""Evaluate risk recovery separately from the action-only reader process."""

from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    parent = directory.parent
    construction = json.loads((parent / "policy01/results.json").read_text())
    reader = json.loads((parent / "reader01/results.json").read_text())
    assert reader["status"] == construction["status"] == "complete"
    path = parent / "policy01/evaluation_data.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == construction["evaluation_sha256"]
    data = np.load(path, allow_pickle=False)
    probabilities, outcomes = data["probabilities"], data["outcomes"][:, None, :]
    recovered = np.load(parent / "reader01/recovered_probabilities.npy", allow_pickle=False)
    assert recovered.shape == probabilities.shape
    failures = 1-outcomes.astype(float)
    estimates = 1-probabilities
    costs = failures*(1-estimates) + estimates**2/2
    losses = (probabilities-outcomes)**2
    identity_error = np.max(np.abs((costs[:, 1:, :] - costs[:, :1, :]) -
                                  (losses[:, 1:, :] - losses[:, :1, :])/2))
    assert identity_error < 1e-14
    expected_variance = probabilities*(1-probabilities)/settings["reader_calls"]
    recovered_losses = (recovered-outcomes)**2
    scene_losses = losses.mean(axis=-1)
    recovered_scene_losses = recovered_losses.mean(axis=-1)
    generator = np.random.default_rng(settings["bootstrap_seed"])
    draws = generator.integers(0, len(scene_losses),
                              (settings["bootstrap_replicates"], len(scene_losses)))
    aggregate = scene_losses[draws].mean(axis=1)
    differences = aggregate[:, 1:].min(axis=1)-aggregate[:, 0]
    recovered_aggregate = recovered_scene_losses[draws].mean(axis=1)
    recovered_differences = aggregate[:, 1:].min(axis=1)-recovered_aggregate[:, 0]
    means = scene_losses.mean(axis=0)
    best_index = int(np.argmin(means[1:]))+1
    reduction = 1-means[0]/means[best_index]
    interval = np.quantile(differences, [.025, .975]).tolist()
    records = {}
    for index, name in enumerate(construction["methods"]):
        records[name] = {"brier": float(means[index]),
                         "mean_routing_cost": float(costs[:, index].mean()),
                         "recovered_brier": float(recovered_losses[:, index].mean()),
                         "expected_recovered_brier": float((losses+expected_variance)[:, index].mean()),
                         "reconstruction_mse": float(((recovered-probabilities)**2)[:, index].mean()),
                         "exact_expected_reconstruction_mse": float(expected_variance[:, index].mean())}
    result = {"status": "complete", "methods": records,
              "best_public_method": construction["methods"][best_index],
              "native_relative_brier_reduction": float(reduction),
              "native_selection_aware_interval": interval,
              "recovered_native_against_direct_public_interval": np.quantile(recovered_differences, [.025, .975]).tolist(),
              "development_signal_pass": bool(reduction >= settings["minimum_relative_brier_reduction"] and interval[0] > 0),
              "risk_cost_identity_maximum_error": float(identity_error),
              "finite_reader_expected_brier_allowance": 1/(4*settings["reader_calls"]),
              "scene_losses": scene_losses.tolist(), "independent_confirmation": False,
              "genuine_removal_established": False, "neural_training": False,
              "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "scene_losses"}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
