"""Independent injection enumeration and direct channel scoring; no runner imports."""

from copy import deepcopy
from fractions import Fraction
from itertools import combinations, permutations
from math import lcm


def enumerate_minima(lower, upper):
    count = len(lower)
    rational_costs = [[sum((first - second) ** 2 for first, second in zip(left, right))
                       for right in upper] for left in lower]
    denominator = lcm(*(value.denominator for row in rational_costs for value in row))
    costs = [[int(value * denominator) for value in row] for row in rational_costs]
    minima = []
    enumerations = 0
    for size in range(count + 1):
        best = None
        for left_indices in combinations(range(count), size):
            for right_indices in permutations(range(count), size):
                value = sum(costs[left][right] for left, right in zip(left_indices, right_indices))
                best = value if best is None else min(best, value)
                enumerations += 1
        minima.append(Fraction(best, denominator * count))
    return minima, enumerations


def integrate_quadratic(function, lower, upper):
    return (upper - lower) * (function(lower) + 4 * function((lower + upper) / 2) + function(upper)) / 6


def audit_certificate(certificate, configuration):
    assert certificate["scope"] == configuration["interpretation"]
    assert [case["name"] for case in certificate["finite"]] == [case["name"] for case in configuration["finite_families"]]
    endpoint_count = 0
    enumeration_count = 0
    state_count = 0
    for case, specification in zip(certificate["finite"], configuration["finite_families"]):
        probability = Fraction(specification["hazard_probability"])
        count = lcm(len(specification["lower_points"]), len(specification["upper_points"]))
        lower = [tuple(map(Fraction, specification["lower_points"][index * len(specification["lower_points"]) // count]))
                 for index in range(count)]
        upper = [tuple(map(Fraction, specification["upper_points"][index * len(specification["upper_points"]) // count]))
                 for index in range(count)]
        assert case["hazard_probability"] == str(probability) and case["expanded_count"] == count
        means = [sum((1 - probability) * point[coordinate] / count for point in lower)
                 + sum(probability * point[coordinate] / count for point in upper) for coordinate in range(len(lower[0]))]
        baseline = sum((1 - probability) * sum((point[index] - means[index]) ** 2 for index in range(len(means))) / count
                       for point in lower)
        baseline += sum(probability * sum((point[index] - means[index]) ** 2 for index in range(len(means))) / count
                        for point in upper)
        assert case["mean"] == list(map(str, means)) and Fraction(case["baseline_variance"]) == baseline
        minima, enumerations = enumerate_minima(lower, upper)
        enumeration_count += enumerations
        assert [row["matched_count"] for row in case["frontier"]] == list(range(count + 1))
        for row in case["frontier"]:
            endpoint_count += 1
            seen_lower, seen_upper = [], []
            loss = Fraction(0)
            balanced = Fraction(0)
            ordinary = Fraction(0)
            variation = Fraction(0)
            matched = 0
            for state in row["states"]:
                state_count += 1
                left, right = state["lower_index"], state["upper_index"]
                forecast = tuple(map(Fraction, state["forecast"]))
                assert len(forecast) == len(means)
                assert left is not None or right is not None
                conditional_zero = Fraction(left is not None, count)
                conditional_one = Fraction(right is not None, count)
                if left is not None:
                    assert 0 <= left < count
                    seen_lower.append(left)
                    loss += (1 - probability) / count * sum((value - prediction) ** 2 for value, prediction in zip(lower[left], forecast))
                if right is not None:
                    assert 0 <= right < count
                    seen_upper.append(right)
                    loss += probability / count * sum((value - prediction) ** 2 for value, prediction in zip(upper[right], forecast))
                balanced += max(conditional_zero, conditional_one) / 2
                ordinary += max((1 - probability) * conditional_zero, probability * conditional_one)
                variation += abs(conditional_zero - conditional_one) / 2
                matched += left is not None and right is not None
            assert sorted(seen_lower) == sorted(seen_upper) == list(range(count))
            assert matched == row["matched_count"]
            assert Fraction(row["conditional_variation"]) == variation == 1 - Fraction(matched, count)
            assert Fraction(row["balanced_judgment_accuracy"]) == balanced
            assert Fraction(row["ordinary_judgment_accuracy"]) == ordinary
            assert Fraction(row["transport_cost"]) == minima[matched]
            assert Fraction(row["squared_loss"]) == loss == probability * (1 - probability) * minima[matched]
            assert Fraction(row["retained_advantage_fraction"]) == 1 - loss / baseline

    expected_inventory = [(case["name"], cap) for case in configuration["continuous_families"]
                          for cap in configuration["conditional_variation_caps"]]
    assert [(case["name"], case["conditional_variation"]) for case in certificate["continuous"]] == expected_inventory
    specifications = {case["name"]: case for case in configuration["continuous_families"]}
    noise = Fraction(configuration["independent_noise_amplitude"])
    for case in certificate["continuous"]:
        specification = specifications[case["name"]]
        probability = Fraction(specification["hazard_probability"])
        lower_start, lower_end = map(Fraction, specification["lower_interval"])
        upper_start, upper_end = map(Fraction, specification["upper_interval"])
        assert lower_start < lower_end <= upper_start < upper_end
        mass = 1 - Fraction(case["conditional_variation"])
        assert Fraction(case["common_mass"]) == mass
        lower_quantile = lambda rank: lower_start + (lower_end - lower_start) * rank
        upper_quantile = lambda rank: upper_start + (upper_end - upper_start) * rank
        expectation = ((1 - probability) * integrate_quadratic(lower_quantile, Fraction(0), Fraction(1))
                       + probability * integrate_quadratic(upper_quantile, Fraction(0), Fraction(1)))
        baseline = ((1 - probability) * integrate_quadratic(lambda rank: (lower_quantile(rank) - expectation) ** 2, Fraction(0), Fraction(1))
                    + probability * integrate_quadratic(lambda rank: (upper_quantile(rank) - expectation) ** 2, Fraction(0), Fraction(1)))
        assert Fraction(case["mean"]) == expectation and Fraction(case["baseline_variance"]) == baseline
        transport = integrate_quadratic(lambda rank: (upper_quantile(rank) - lower_quantile(1 - mass + rank)) ** 2, Fraction(0), mass)
        def matched_loss(rank, perturbation):
            left, right = lower_quantile(1 - mass + rank), upper_quantile(rank)
            prediction = (1 - probability) * left + probability * right
            return (1 - probability) * (left + perturbation - prediction) ** 2 + probability * (right + perturbation - prediction) ** 2
        loss = integrate_quadratic(lambda rank: matched_loss(rank, Fraction(0)), Fraction(0), mass)
        noisy_loss = integrate_quadratic(lambda rank: (matched_loss(rank, noise) + matched_loss(rank, -noise)) / 2, Fraction(0), mass)
        noisy_loss += (1 - mass) * noise ** 2
        noisy_baseline = sum(((1 - probability) * integrate_quadratic(lambda rank: (lower_quantile(rank) + perturbation - expectation) ** 2, Fraction(0), Fraction(1))
                              + probability * integrate_quadratic(lambda rank: (upper_quantile(rank) + perturbation - expectation) ** 2, Fraction(0), Fraction(1))) / 2
                             for perturbation in [-noise, noise])
        assert Fraction(case["transport_cost"]) == transport
        assert Fraction(case["squared_loss"]) == loss == probability * (1 - probability) * transport
        assert Fraction(case["balanced_judgment_accuracy"]) == 1 - mass / 2
        assert Fraction(case["retained_advantage_fraction"]) == 1 - loss / baseline
        assert Fraction(case["noisy_intact_loss"]) == noise ** 2
        assert Fraction(case["noisy_edited_loss"]) == noisy_loss
        assert Fraction(case["noisy_baseline_loss"]) == noisy_baseline
        assert (noisy_baseline - noisy_loss) / (noisy_baseline - noise ** 2) == 1 - loss / baseline
    return {"finite_families": len(certificate["finite"]), "frontier_endpoints": endpoint_count,
            "partial_injections_enumerated": enumeration_count, "retained_states_scored": state_count,
            "continuous_cases": len(certificate["continuous"]), "independent_noise_cases": len(certificate["continuous"])}


def reject_corruptions(certificate, configuration):
    mutations = {
        "transport_cost": lambda changed: changed["finite"][0]["frontier"][-1].update(transport_cost="0"),
        "matched_mass": lambda changed: changed["finite"][0]["frontier"][-1].update(conditional_variation="1"),
        "retained_witness": lambda changed: changed["finite"][0]["frontier"][-1]["states"][0].update(forecast=["0"]),
        "hazard_prevalence": lambda changed: changed["finite"][0].update(hazard_probability="1/10"),
        "constant_baseline": lambda changed: changed["finite"][0].update(baseline_variance="1"),
        "judgment_score": lambda changed: changed["finite"][0]["frontier"][-1].update(balanced_judgment_accuracy="1"),
        "continuous_moment": lambda changed: changed["continuous"][0].update(squared_loss="0"),
        "noise_accounting": lambda changed: changed["continuous"][0].update(noisy_edited_loss="0"),
        "case_inventory": lambda changed: changed["continuous"].pop(),
    }
    rejected = []
    for name, mutate in mutations.items():
        changed = deepcopy(certificate)
        mutate(changed)
        try:
            audit_certificate(changed, configuration)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError("Corruption accepted: " + name)
    return rejected
