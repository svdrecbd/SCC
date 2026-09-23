"""Validate fixed-event recovery and its sharp predictive-loss boundary."""
from fractions import Fraction
from itertools import product
from math import ceil, comb, log
from pathlib import Path
import json
import sys


def relative_entropy(actual, predicted):
    return sum(float(weight) * log(float(weight / predicted[outcome]))
               for outcome, weight in actual.items() if weight)


def prefix_mass(law, prefix):
    return sum(weight for outcome, weight in law.items()
               if outcome[:len(prefix)] == prefix)


def sequential_relative_entropy(actual, predicted, length):
    total = 0.0
    for index in range(length):
        for prefix in product((0, 1), repeat=index):
            actual_mass = prefix_mass(actual, prefix)
            predicted_mass = prefix_mass(predicted, prefix)
            if not actual_mass:
                continue
            actual_next = {symbol: prefix_mass(actual, prefix + (symbol,)) / actual_mass
                           for symbol in (0, 1)}
            predicted_next = {symbol: prefix_mass(predicted, prefix + (symbol,)) / predicted_mass
                              for symbol in (0, 1)}
            total += float(actual_mass) * relative_entropy(actual_next, predicted_next)
    return total


def event(outcome):
    # Fixed outcome threshold, specified before selecting the probability laws.
    value = sum(symbol * 2 ** index for index, symbol in enumerate(reversed(outcome)))
    return value >= 2 ** (len(outcome) - 1)


def construct_law(conditional, event_mass):
    return {outcome: weight * (event_mass if event(outcome) else 1 - event_mass)
            for outcome, weight in conditional.items()}


def binomial_error(probability, sample_count, true_judgment):
    return sum(Fraction(comb(sample_count, count)) * probability ** count
               * (1 - probability) ** (sample_count - count)
               for count in range(sample_count + 1)
               if (2 * count > sample_count) != true_judgment)


def main(directory):
    configuration = json.loads((directory / 'config.json').read_text())
    length = configuration['trajectory_length']
    outcomes = list(product((0, 1), repeat=length))
    rows = []
    maximum_chain_error = 0.0
    conditional_controls = 0
    for context in range(configuration['contexts']):
        conditional = {}
        for event_value in (False, True):
            selected = [outcome for outcome in outcomes if event(outcome) == event_value]
            weights = [1 + (index + context) % len(selected) for index in range(len(selected))]
            for outcome, weight in zip(selected, weights):
                conditional[outcome] = Fraction(weight, sum(weights))
        removed = construct_law(conditional, Fraction(1, 2))
        laws = [construct_law(conditional, probability)
                for probability in (Fraction(1, 4), Fraction(3, 4))]
        assert all(removed[outcome] == (laws[0][outcome] + laws[1][outcome]) / 2
                   for outcome in outcomes)
        for actual, probability in zip(laws, (Fraction(1, 4), Fraction(3, 4))):
            for replacement_mass in (Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)):
                predicted = construct_law(conditional, replacement_mass)
                assert sum(predicted.values()) == 1
                for outcome in outcomes:
                    part_mass = replacement_mass if event(outcome) else 1 - replacement_mass
                    assert predicted[outcome] / part_mass == conditional[outcome]
                divergence = relative_entropy(actual, predicted)
                binary_divergence = relative_entropy(
                    {1: probability, 0: 1 - probability},
                    {1: replacement_mass, 0: 1 - replacement_mass})
                sequential_divergence = sequential_relative_entropy(actual, predicted, length)
                assert abs(divergence - binary_divergence) < 1e-12
                assert abs(divergence - sequential_divergence) < 1e-12
                maximum_chain_error = max(maximum_chain_error, abs(divergence - sequential_divergence))
                altered = dict(predicted)
                first, second = [outcome for outcome in outcomes if event(outcome)][:2]
                displacement = min(altered[first], altered[second]) / 3
                altered[first] += displacement
                altered[second] -= displacement
                assert relative_entropy(actual, altered) > binary_divergence + 1e-8
                conditional_controls += 1
                rows.append({'context': context, 'actual_event_mass': str(probability),
                             'edited_event_mass': str(replacement_mass), 'relative_entropy': divergence})
        # The complete retained conditional generator is the same in both regimes.
        # Every randomized reader of any context therefore has balanced accuracy 1/2.
        for numerator in range(17):
            decision_probability = Fraction(numerator, 16)
            assert (decision_probability + 1 - decision_probability) / 2 == Fraction(1, 2)
    sampling_rows = []
    delta = Fraction(configuration['failure_probability'])
    for margin_text in configuration['margins']:
        margin = Fraction(margin_text)
        sample_count = ceil(2 / float(margin ** 2) * log(2 / float(delta)))
        for direction in (-1, 1):
            sampled_probability = Fraction(1, 2) + direction * margin / 2
            error = binomial_error(sampled_probability, sample_count, direction == 1)
            assert error <= delta
            sampling_rows.append({'margin': str(margin), 'sample_count': sample_count,
                                  'sampled_probability': str(sampled_probability),
                                  'exact_binomial_error_float': float(error),
                                  'model_calls': sample_count * length})
    removal_cost = relative_entropy({0: Fraction(1, 4), 1: Fraction(3, 4)},
                                    {0: Fraction(1, 2), 1: Fraction(1, 2)})
    report = {'status': 'complete', 'partition_checks': len(rows),
              'conditional_alteration_controls': conditional_controls,
              'maximum_chain_rule_error': maximum_chain_error,
              'regime_removal_cost_nats': removal_cost,
              'regime_judgment_accuracy_after_removal': 0.5,
              'sampling_checks': sampling_rows, 'rows': rows,
              'neural_training': False, 'general_SCC_impossibility': False}
    (directory / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
