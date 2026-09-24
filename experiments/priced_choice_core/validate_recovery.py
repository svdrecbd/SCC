"""Exact rational controls for recovery from arbitrary priced-choice policies."""

from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import time

from recover_risk import recover_probability, variance_allowance


def integrated_policy(probability, choices):
    width = Fraction(1, len(choices))
    mass = sum(choices, Fraction(0)) * width
    cost = Fraction(0)
    for index, choice in enumerate(choices):
        left, right = index * width, (index + 1) * width
        cost += choice * (right**2 - left**2) / 2
        cost += (1 - choice) * probability * width
    return mass, cost


def threshold_cost(probability, threshold):
    return threshold**2 / 2 + probability * (1 - threshold)


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    denominator = settings["probability_denominator"]
    probabilities = [Fraction(index, denominator) for index in range(denominator + 1)]
    baseline_denominator = settings["baseline_denominator"]
    baselines = [Fraction(index, baseline_denominator)
                 for index in range(baseline_denominator + 1)]
    counts = {"policy_probability_pairs": 0, "baseline_gain_checks": 0,
              "sampling_bound_checks": 0, "sharp_threshold_checks": 0}
    maximum_rearrangement_gap = Fraction(0)
    policy_families = [
        product([Fraction(0), Fraction(1)], repeat=settings["binary_intervals"]),
        product([Fraction(str(value)) for value in settings["stochastic_action_values"]],
                repeat=settings["stochastic_intervals"]),
    ]
    for policies in policy_families:
        for choices in policies:
            for probability in probabilities:
                mass, cost = integrated_policy(probability, choices)
                ordered_cost = threshold_cost(probability, mass)
                assert cost >= ordered_cost
                maximum_rearrangement_gap = max(maximum_rearrangement_gap, cost - ordered_cost)
                optimal_cost = probability - probability**2 / 2
                assert (mass - probability)**2 <= 2 * (cost - optimal_cost)
                counts["policy_probability_pairs"] += 1
                for baseline in baselines:
                    utility_gain = threshold_cost(probability, baseline) - cost
                    protected_gain = (baseline - probability)**2 - (mass - probability)**2
                    assert protected_gain >= 2 * utility_gain
                    counts["baseline_gain_checks"] += 1
                for calls in settings["sampling_calls"]:
                    variance = mass * (1 - mass) / calls
                    allowance = Fraction(1, 4 * calls)
                    assert variance <= allowance
                    assert (mass - probability)**2 + variance <= 2 * (cost - optimal_cost) + allowance
                    counts["sampling_bound_checks"] += 1
    for probability in probabilities:
        for threshold in baselines:
            assert threshold_cost(probability, threshold) - (probability - probability**2/2) == (threshold-probability)**2/2
            counts["sharp_threshold_checks"] += 1

    # Fixed-price decisions cannot identify probabilities on either side.
    assert int(Fraction(1, 2) < Fraction(3, 4)) == int(Fraction(1, 2) < Fraction(7, 8))
    # Any finite, known deterministic query set can be changed at measure zero.
    query_prices = {Fraction(2 * index + 1, 32) for index in range(16)}
    reference_probability = Fraction(3, 4)
    grid_actions = [0 if price in query_prices else int(price < reference_probability)
                    for price in query_prices]
    assert sum(grid_actions) == 0
    grid_counterexample = {"true_probability": str(reference_probability),
                          "uniform_price_regret": "0",
                          "deterministic_grid_estimate": 0,
                          "explanation": "Changed finitely many price points only; uniform measure is unchanged."}
    # A one-time stateful response is not an independent repeatable endpoint.
    state = {"calls": 0}
    def changing_policy(price):
        state["calls"] += 1
        return int(price < .75) if state["calls"] == 1 else 0
    stateful_estimate = recover_probability(changing_policy, 1024, settings["seed"])
    assert stateful_estimate <= 1 / 1024
    implementation_checks = []
    for probability in (0, .125, .5, .875, 1):
        estimate = recover_probability(lambda price: int(price < probability), 4096,
                                       settings["seed"])
        complement = recover_probability(lambda price: int(price < probability), 4096,
                                         settings["seed"], complement=True)
        assert estimate + complement == 1
        assert abs(estimate-probability) <= .04
        implementation_checks.append({"probability": probability, "estimate": estimate})
    for invalid in (0, -1, True, 1.5):
        try:
            recover_probability(lambda price: 0, invalid, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid query budget accepted")
    try:
        recover_probability(lambda price: 2, 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid action accepted")
    assert variance_allowance(1024) == 1/4096
    result = {"status": "complete", "counts": counts,
              "maximum_rearrangement_gap": str(maximum_rearrangement_gap),
              "deterministic_query_counterexample": grid_counterexample,
              "stateful_without_reset_counterexample_estimate": stateful_estimate,
              "reader_implementation_checks": implementation_checks,
              "neural_training": False, "mechanism_demonstrated": False,
              "wall_seconds": time.perf_counter()-started,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
