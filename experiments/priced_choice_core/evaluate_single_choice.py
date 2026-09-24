"""Evaluate exact one-call expectations on the saved spatial development law."""

from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

from recover_shrunk_risk import recover_shrunk_risk


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    path = directory.parent / "policy01/evaluation_data.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == settings["input_sha256"]
    data = np.load(path, allow_pickle=False)
    reference = json.loads((directory.parent / "policy01/results.json").read_text())
    levels = settings["price_levels"]
    weight = settings["mixing_weight"]
    native = data["probabilities"][:, 0, :]
    baseline_index = reference["methods"].index("spatial_mean")
    baseline = np.rint(data["probabilities"][:, baseline_index, :]*levels)/levels
    outcome = data["outcomes"].astype(float)
    failure_mass = np.clip(np.ceil((1-native)*levels-.5), 0, levels)/levels
    risk_mass = 1-failure_mass
    baseline_failure = 1-baseline
    failure = 1-outcome
    cost = failure*(1-failure_mass)+failure_mass**2/2
    reference_cost = failure*(1-baseline_failure)+baseline_failure**2/2
    useful_gain = reference_cost-cost
    mean = (1-weight)*baseline+weight*risk_mass
    expected_loss = (mean-outcome)**2+weight**2*risk_mass*(1-risk_mass)
    reference_loss = (baseline-outcome)**2
    expected_gain = reference_loss-expected_loss
    bound = 2*weight*useful_gain-weight**2/4
    assert np.min(expected_gain-bound) >= -1e-12
    generator = np.random.default_rng(settings["seed"])
    prices = (generator.integers(0, levels, size=native.shape)+.5)/levels
    actions = prices < 1-native
    sampled_probability = (1-weight)*baseline+weight*(1-actions)
    sampled_gain = reference_loss-(sampled_probability-outcome)**2
    scalar_checks = []
    for index in np.linspace(0, native.size-1, 64, dtype=int):
        calls = []
        probability = float(native.ravel()[index])
        def policy(price):
            action = int(price < 1-probability)
            calls.append({"price": price, "action": action})
            return action
        result = recover_shrunk_risk(policy, float(baseline.ravel()[index]), weight,
                                    calls=1, price_levels=levels, seed=settings["seed"]+int(index),
                                    complement=True)
        assert len(calls) == 1
        expected = (1-weight)*float(baseline.ravel()[index])+weight*(1-calls[0]["action"])
        assert result == expected
        scalar_checks.append({"index": int(index), "result": result, "calls": calls})
    result = {"status": "complete", "queries": int(native.size), "price_levels": levels,
              "mixing_weight": weight, "expected_useful_gain": float(useful_gain.mean()),
              "expected_one_call_protected_gain": float(expected_gain.mean()),
              "guaranteed_protected_gain": float(bound.mean()),
              "sampled_one_call_protected_gain": float(sampled_gain.mean()),
              "minimum_bound_slack": float(np.min(expected_gain-bound)),
              "finite_price_probability_change_maximum": float(np.max(np.abs(risk_mass-native))),
              "scalar_implementation_checks": scalar_checks,
              "neural_training": False, "genuine_removal_established": False,
              "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "scalar_implementation_checks"}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
