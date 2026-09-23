"""Exact implementation controls for mean-to-threshold-risk score recovery."""
from fractions import Fraction
import itertools
import json
from pathlib import Path
import sys
import time


def clip_probability(value):
    return min(Fraction(1), max(Fraction(0), value))


def main(directory):
    configuration = json.loads((directory / "config.json").read_text())
    started = time.monotonic()
    grid = [Fraction(index, configuration["denominator"]) for index in range(configuration["denominator"] + 1)]
    checks = positive = clipped = corrupt_sign_failures = 0
    smallest_margin = None
    for count in configuration["threshold_counts"]:
        for weighting in ("uniform", "geometric"):
            raw_weights = [1] * count if weighting == "uniform" else [2 ** index for index in range(count)]
            weights = [Fraction(value, sum(raw_weights)) for value in raw_weights]
            for baseline in itertools.product(grid, repeat=count):
                baseline_mean = sum(weight * risk for weight, risk in zip(weights, baseline))
                for forecast in grid:
                    shift = forecast - baseline_mean
                    repaired = tuple(clip_probability(risk + shift) for risk in baseline)
                    corrupted = tuple(clip_probability(risk - shift) for risk in baseline)
                    if any(value != risk + shift for value, risk in zip(repaired, baseline)):
                        clipped += 1
                    for tier in range(count + 1):
                        labels = [int(index < tier) for index in range(count)]
                        outcome = sum(weight * label for weight, label in zip(weights, labels))
                        useful_gain = (baseline_mean - outcome)**2 - (forecast - outcome)**2
                        baseline_loss = sum(weight * (risk-label)**2 for weight, risk, label in zip(weights, baseline, labels))
                        risk_gain = baseline_loss - sum(weight*(risk-label)**2 for weight,risk,label in zip(weights,repaired,labels))
                        raw_gain = baseline_loss - sum(weight*(risk+shift-label)**2 for weight,risk,label in zip(weights,baseline,labels))
                        assert raw_gain == useful_gain
                        assert risk_gain >= useful_gain
                        if all(baseline[index] >= baseline[index+1] for index in range(count-1)):
                            assert all(repaired[index] >= repaired[index+1] for index in range(count-1))
                        corrupted_gain = baseline_loss - sum(weight*(risk-label)**2 for weight,risk,label in zip(weights,corrupted,labels))
                        corrupt_sign_failures += corrupted_gain < useful_gain
                        checks += 1
                        positive += useful_gain > 0
                        margin = risk_gain - useful_gain
                        smallest_margin = margin if smallest_margin is None else min(smallest_margin, margin)
    # A mean forecast improves while its degenerate-distribution risk estimate worsens.
    prevalence = Fraction(1, 10)
    mean_forecast = prevalence
    baseline_mse = prevalence
    new_mse = (1-prevalence)*mean_forecast**2 + prevalence*(1-mean_forecast)**2
    baseline_risk = prevalence
    point_mass_risk = (1-prevalence)*mean_forecast + prevalence*(1-mean_forecast)
    recovered_risk = new_mse
    assert new_mse < baseline_mse and point_mass_risk > baseline_risk
    assert baseline_risk - recovered_risk == baseline_mse - new_mse
    assert corrupt_sign_failures > 0
    result = {"exact_pointwise_checks": checks, "positive_useful_gains": positive,
              "clipped_forecast_configurations": clipped, "corrupt_sign_rejections": corrupt_sign_failures,
              "minimum_recovery_margin": str(smallest_margin),
              "control": {"baseline_mean_mse": str(baseline_mse), "new_mean_mse": str(new_mse),
                          "baseline_risk_loss": str(baseline_risk), "point_mass_risk_loss": str(point_mass_risk),
                          "recovered_risk_loss": str(recovered_risk)},
              "seconds": time.monotonic()-started, "neural_training": False}
    (directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
