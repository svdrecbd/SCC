"""Exact finite expected-score validation, including nonmonotone policies."""

from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import sys
import time

from recover_shrunk_risk import recover_shrunk_risk


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    weights = [Fraction(value) for value in settings["mixing_weights"]]
    counts = {"mixture_bounds": 0, "optimized_bounds": 0,
              "positive_useful_cases": 0, "nonpositive_useful_cases": 0}
    families = [(settings["binary_price_levels"], (Fraction(0), Fraction(1))),
                (settings["stochastic_price_levels"], (Fraction(0), Fraction(1, 2), Fraction(1)))]
    for levels, values in families:
        prices = [Fraction(2*index+1, 2*levels) for index in range(levels)]
        for actions in product(values, repeat=levels):
            mass = sum(actions) / levels
            for risk_index in range(settings["probability_denominator"]+1):
                risk = Fraction(risk_index, settings["probability_denominator"])
                cost = sum(price*action+risk*(1-action) for price, action in zip(prices, actions)) / levels
                assert cost >= risk*(1-mass)+mass**2/2
                for reference_index in range(levels+1):
                    baseline = Fraction(reference_index, levels)
                    reference_cost = sum(price if index < reference_index else risk
                                         for index, price in enumerate(prices)) / levels
                    assert reference_cost == risk*(1-baseline)+baseline**2/2
                    utility_gain = reference_cost-cost
                    counts["positive_useful_cases" if utility_gain > 0 else "nonpositive_useful_cases"] += 1
                    for calls in settings["sampling_calls"]:
                        variance = mass*(1-mass)/calls
                        for weight in weights:
                            mean = (1-weight)*baseline+weight*mass
                            expected_score = (mean-risk)**2+weight**2*variance
                            score_gain = (baseline-risk)**2-expected_score
                            bound = 2*weight*utility_gain-weight**2/Fraction(4*calls)
                            assert score_gain >= bound
                            complement_mean = (1-weight)*(1-baseline)+weight*(1-mass)
                            assert (complement_mean-(1-risk))**2+weight**2*variance == expected_score
                            counts["mixture_bounds"] += 1
                        if utility_gain >= 0:
                            weight = min(Fraction(1), 4*calls*utility_gain)
                            bound = 2*weight*utility_gain-weight**2/Fraction(4*calls)
                            expected = (4*calls*utility_gain**2 if utility_gain <= Fraction(1, 4*calls)
                                        else 2*utility_gain-Fraction(1, 4*calls))
                            assert bound == expected
                            counts["optimized_bounds"] += 1
    implementation = []
    for probability in (0, .25, .5, .75, 1):
        for weight in (0, .125, .5, 1):
            result = recover_shrunk_risk(lambda price: int(price < probability), .5, weight,
                                        calls=16, seed=settings["seed"])
            complement = recover_shrunk_risk(lambda price: int(price < probability), .5, weight,
                                            calls=16, seed=settings["seed"], complement=True)
            assert result+complement == 1
            assert 0 <= result <= 1
            implementation.append({"probability": probability, "weight": weight, "result": result})
    invalid_arguments = [{"calls": 0}, {"calls": True}, {"price_levels": 3},
                         {"price_levels": 1}, {"price_levels": 2**53},
                         {"baseline": .3}, {"baseline": float("nan")},
                         {"mixing_weight": -1}, {"mixing_weight": float("inf")}]
    for invalid in invalid_arguments:
        arguments = {"baseline": .5, "mixing_weight": .5, "seed": 0}
        arguments.update(invalid)
        try:
            recover_shrunk_risk(lambda price: 0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid argument accepted")
    result = {"status": "complete", "counts": counts, "implementation_checks": implementation,
              "invalid_argument_checks": len(invalid_arguments), "neural_training": False,
              "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "implementation_checks"}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
