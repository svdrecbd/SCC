"""Measure retained inference and disclosure assessment in a public replacement."""
import json
from pathlib import Path
import sys


def main(directory, output):
    import numpy as np

    configuration = json.loads((directory / "config.json").read_text())
    rows = []
    for family in configuration["families"]:
        for seed in configuration["seeds"]:
            location = directory / f"{family}_{seed}"
            result = json.loads((location / "result.json").read_text())
            with np.load(location / "task.npz") as task, np.load(location / "predictions.npz") as predictions:
                outcome = task["outcomes"]
                indicator = outcome[:, None] > np.arange(configuration["class_count"] - 1)[None, :]
                losses = {}
                for name in ("context_frequency", "contextual_model", "linear_discriminant", "public_selected"):
                    risk = 1 - np.cumsum(predictions[name], axis=1)[:, :-1]
                    losses[name] = float(np.mean(((risk - indicator)**2).sum(axis=1)))
                rows.append({"family": family, "seed": seed, "risk_loss": losses,
                             "useful_loss": {name: result["scores"][name] for name in losses},
                             "linear_discriminant_seconds": result["public_model_seconds"]["linear_discriminant"]})
    useful = {name: float(np.mean([row["useful_loss"][name] for row in rows])) for name in rows[0]["useful_loss"]}
    risk = {name: float(np.mean([row["risk_loss"][name] for row in rows])) for name in rows[0]["risk_loss"]}
    retained = {name: (useful["context_frequency"]-useful[name])/(useful["context_frequency"]-useful["contextual_model"])
                for name in ("linear_discriminant", "public_selected")}
    result = {"useful_brier": useful, "summed_risk_brier": risk, "retained_useful_gain_over_frequency": retained,
              "mean_linear_discriminant_seconds_including_validation": float(np.mean([row["linear_discriminant_seconds"] for row in rows])),
              "risk_removed": False, "replacement_uses_neural_parameters": False, "rows": rows}
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
