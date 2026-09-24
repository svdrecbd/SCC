"""Construct reusable priced choices from frozen spatial predictions."""

from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

from analyze import features, thumbnail
from data import read_frame


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    prior = Path(settings["prior_root"])
    assert digest(prior / "transfer_manifest.json") == settings["prior_manifest_sha256"]
    manifest = json.loads((prior / "transfer_manifest.json").read_text())
    expected = {record["path"]: record["sha256"] for record in manifest["files"]}
    used = {}
    def checked(relative):
        path = prior / relative
        used[relative] = digest(path)
        assert used[relative] == expected[relative]
        return path
    configuration = json.loads(checked("analysis01/config.json").read_text())
    selection = json.loads(checked("selection01/selection.json").read_text())
    parameters = np.load(checked("analysis01/public_parameters.npz"), allow_pickle=False)
    calibration = []
    image_hashes = []
    for record in selection[:16]:
        _, depth, image_hash = read_frame(prior / "acquisition01", record)
        values = depth.astype(np.float64)
        invalid = ~((values > .1) & (values < 10) & np.isfinite(values))
        values[invalid] = parameters["global_mean"]
        calibration.append(values)
        image_hashes.append(image_hash)
    calibration = np.stack(calibration)
    coordinates = np.meshgrid(np.linspace(-1, 1, 640), np.linspace(-1, 1, 480))
    crop = np.zeros((480, 640), dtype=bool)
    row_start, row_end, column_start, column_end = configuration["crop"]
    crop[row_start:row_end, column_start:column_end] = True
    names = ["native", "global_mean", "spatial_mean", "quadratic_position", "position_rgb",
             "nearest_image", "three_nearest_images"]
    generator = np.random.default_rng(settings["seed"])
    action_generator = np.random.default_rng(settings["seed"] + 1)
    probabilities, outcomes, packed_choices, queries = [], [], [], []
    inherited_forward_seconds = 0
    for index in range(16, 32):
        rgb, depth, image_hash = read_frame(prior / "acquisition01", selection[index])
        image_hashes.append(image_hash)
        truth = depth.astype(np.float64)
        valid = crop & (truth > .1) & (truth < 10) & np.isfinite(truth)
        positions = generator.choice(np.flatnonzero(valid), settings["queries_per_scene"], replace=False)
        placements = generator.random(settings["queries_per_scene"])
        prediction = np.load(checked(f"inference{index//4+1:02d}/prediction_{index:03d}.npy"), allow_pickle=False).astype(np.float64)
        prediction = np.clip(prediction, .1, 10)
        summary = json.loads(checked(f"inference{index//4+1:02d}/summary.json").read_text())
        assert summary["state_sha256_before"] == summary["state_sha256_after"]
        if index % 4 == 0:
            inherited_forward_seconds += summary["wall_seconds"]
        design = features(rgb, coordinates)
        distances = np.mean((parameters["thumbnails"] - thumbnail(rgb))**2, axis=1)
        neighbors = np.argsort(distances, kind="stable")[:3]
        weights = 1 / (distances[neighbors] + 1e-8)
        weights /= weights.sum()
        estimates = [prediction,
                     np.full_like(prediction, parameters["global_mean"]),
                     parameters["spatial_mean"],
                     design[:, :, :6] @ parameters["position_coefficients"],
                     design @ parameters["rgb_coefficients"],
                     calibration[neighbors[0]],
                     np.einsum("n,nhw->hw", weights, calibration[neighbors])]
        empirical = calibration.reshape(16, -1)[:, positions] / 10
        baseline = np.mean(empirical > placements, axis=0)
        mean = empirical.mean(axis=0)
        risks = np.stack([np.clip(baseline + np.clip(estimate.ravel()[positions], .1, 10)/10 - mean, 0, 1)
                          for estimate in estimates])
        prices = action_generator.random((settings["queries_per_scene"], settings["reader_calls"]))
        choices = prices[None, :, :] < (1 - risks[:, :, None])
        packed_choices.append(np.packbits(choices, axis=-1, bitorder="little"))
        probabilities.append(risks)
        outcomes.append(truth.ravel()[positions]/10 > placements)
        queries.append({"selection_index": index, "positions": positions.tolist(),
                        "placements": placements.tolist(), "scene": selection[index]["scene"]})
    np.savez_compressed(directory / "policy_actions.npz", actions=np.stack(packed_choices))
    np.savez_compressed(directory / "evaluation_data.npz", probabilities=np.stack(probabilities),
                        outcomes=np.stack(outcomes))
    (directory / "queries.json").write_text(json.dumps(queries) + "\n")
    metadata = {"status": "complete", "methods": names, "scene_count": 16,
                "queries_per_scene": settings["queries_per_scene"], "reader_calls": settings["reader_calls"],
                "actions_sha256": digest(directory / "policy_actions.npz"),
                "evaluation_sha256": digest(directory / "evaluation_data.npz"),
                "inherited_input_hashes": used, "image_member_hashes": image_hashes,
                "inherited_four_inference_block_seconds": inherited_forward_seconds,
                "new_neural_forward_calls": 0, "neural_training": False,
                "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metadata.items()
                      if key not in ("inherited_input_hashes", "image_member_hashes")}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
