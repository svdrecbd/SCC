"""Exact signal and variance of the two-batch parity Brier derivative."""

from fractions import Fraction
from itertools import product
from math import comb
from pathlib import Path
import json
import sys
import time


def binomial_law(count, probability):
    return [(value, Fraction(comb(count, value)) * probability**value
             * (1 - probability)**(count - value)) for value in range(count + 1)]


def score_moments(count, probability):
    law = binomial_law(count, probability)
    event_probability = sum(weight for value, weight in law if value % 2)
    derivative = sum(weight * (value % 2) * (value - count * probability)
                     for value, weight in law)
    score_second = count * probability * (1 - probability)
    event_score_second = sum(weight * (value % 2) * (value - count * probability)**2
                             for value, weight in law)
    assert event_probability == (1 - (1 - 2 * probability)**count) / 2
    assert derivative == count * probability * (1 - probability) * (1 - 2 * probability)**(count - 1)
    return event_probability, derivative, score_second, event_score_second


def mean_binomial_moments(count, probability):
    # Stirling numbers express ordinary powers through falling factorial powers.
    stirling = ((1,), (0, 1), (0, 1, 1), (0, 1, 3, 1), (0, 1, 7, 6, 1))
    moments = []
    for degree in range(5):
        falling = 1
        value = Fraction(0)
        for order in range(degree + 1):
            if order:
                falling *= count - order + 1
            value += stirling[degree][order] * falling * probability**order
        moments.append(value / count**degree)
    return moments


def gradient_statistics(moments, target, sample_count):
    probability, derivative, score_second, event_score_second = moments
    powers = mean_binomial_moments(sample_count, probability)
    # Conditional on A's frequency t, B has mean 2(t-y)d and second
    # single-sample moment 4(t-y)^2[(1-2t) E(f S^2)+t^2 E(S^2)].
    squared = powers[2] - 2 * target * powers[1] + target**2
    squared_linear = powers[3] - 2 * target * powers[2] + target**2 * powers[1]
    squared_quadratic = powers[4] - 2 * target * powers[3] + target**2 * powers[2]
    expected_conditional_variance = 4 * (
        event_score_second * (squared - 2 * squared_linear)
        + score_second * squared_quadratic - derivative**2 * squared) / sample_count
    variance_conditional_mean = 4 * derivative**2 * probability * (1 - probability) / sample_count
    variance = expected_conditional_variance + variance_conditional_mean
    expected = 2 * (probability - target) * derivative
    assert variance >= 0
    return expected, variance


def enumerate_estimator(count, probability, target, sample_count):
    probability_event, _, _, _ = score_moments(count, probability)
    first = binomial_law(sample_count, probability_event)
    second = binomial_law(count, probability)
    expectation = Fraction(0)
    second_moment = Fraction(0)
    cases = 0
    for successes, first_weight in first:
        frequency = Fraction(successes, sample_count)
        for observations in product(second, repeat=sample_count):
            weight = first_weight
            gradient = Fraction(0)
            for value, observation_weight in observations:
                weight *= observation_weight
                gradient += 2 * (frequency - target) * ((value % 2) - frequency) * (value - count * probability) / sample_count
            expectation += weight * gradient
            second_moment += weight * gradient**2
            cases += 1
    return expectation, second_moment - expectation**2, cases


def sufficient_batch_size(moments, target, ratio):
    def meets_requirement(count):
        expected, variance = gradient_statistics(moments, target, count)
        return expected**2 >= ratio**2 * variance
    upper = 1
    while not meets_requirement(upper):
        upper *= 2
    lower = upper // 2
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if meets_requirement(middle):
            upper = middle
        else:
            lower = middle
    # This is a found sufficient size with an adjacent failing size. We do not
    # assert global minimality without a separate monotonicity proof.
    assert meets_requirement(upper)
    if upper > 1:
        assert not meets_requirement(upper - 1)
    return upper


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    exact_checks = 0
    enumerated_cases = 0
    for count, sample_count, target in product(range(1, 5), (1, 2), (0, 1)):
        probability = Fraction(3, 4)
        expected, variance, cases = enumerate_estimator(count, probability, target, sample_count)
        assert (expected, variance) == gradient_statistics(score_moments(count, probability), target, sample_count)
        exact_checks += 1
        enumerated_cases += cases
    records = []
    for probability_text, count, target in product(configuration["probabilities"], configuration["lengths"], (0, 1)):
        probability = Fraction(probability_text)
        moments = score_moments(count, probability)
        expected, variance = gradient_statistics(moments, target, configuration["samples_per_batch"])
        sufficient = sufficient_batch_size(moments, target, configuration["signal_to_noise_requirement"])
        records.append({"predicate_probability": float(probability), "program_length": count,
                        "target": target, "risk": float(moments[0]),
                        "shared_logit_brier_derivative": float(expected),
                        "gradient_variance": float(variance),
                        "gradient_standard_deviation": float(variance)**0.5,
                        "signal_to_noise": abs(float(expected)) / float(variance)**0.5,
                        "sufficient_samples_per_batch": sufficient,
                        "total_joint_scenes_at_sufficient_size": 2 * sufficient,
                        "exact_integration_gradient_variance": 0})
    result = {"status": "complete", "exact_moment_checks": exact_checks,
              "enumerated_estimator_cases": enumerated_cases, "records": records,
              "neural_training": False, "wall_seconds": time.perf_counter() - started}
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
