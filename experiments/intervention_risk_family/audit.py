"""Independent path enumeration and regression checks for intervention controls."""
import copy
import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


def posterior_by_paths(history):
    weights = {}
    for numerator, initial, increments in itertools.product((1, 3), range(3), itertools.product((0, 1), repeat=6)):
        states = [initial]
        for increment in increments:
            states.append(min(8, states[-1] + increment))
        weight = numerator ** sum(increments) * (8 - numerator) ** (6 - sum(increments))
        for state, observed in zip(states, history):
            weight *= sum(min(8, max(0, state + noise)) == observed for noise in (-1, 0, 0, 1))
        key = numerator, states[-1]
        weights[key] = weights.get(key, 0) + weight
    normalizer = sum(weights.values())
    return {key: Fraction(value, normalizer) for key, value in weights.items() if value}


def future_by_paths():
    table = {}
    for numerator, initial in itertools.product((1, 3), range(9)):
        forecasts, risks = [], []
        for load in (1, 2, 3):
            damage_totals, production_totals = [0] * 8, [0] * 8
            failed_total = 0
            for increments in itertools.product((0, 1), repeat=8):
                weight = numerator ** sum(increments) * (8 - numerator) ** (8 - sum(increments))
                damage = initial
                for index, increment in enumerate(increments):
                    damage = min(8, damage + load * increment)
                    damage_totals[index] += weight * damage
                    production_totals[index] += weight * load * (damage < 8)
                failed_total += weight * (damage == 8)
            for damage, production in zip(damage_totals, production_totals):
                forecasts.extend((Fraction(damage, 8 * 8 ** 8), Fraction(production, 3 * 8 ** 8)))
            risks.append(Fraction(failed_total, 8 ** 8))
        table[numerator, initial] = forecasts, risks
    return table


def close(actual, expected, tolerance=1e-10):
    assert np.all(np.isfinite(np.asarray(actual, dtype=float)))
    assert np.allclose(actual, expected, rtol=tolerance, atol=tolerance)


def audit_family(targets, labels, diagnostic, caps):
    centered = labels - np.mean(labels, axis=0)
    response = targets - np.mean(targets, axis=0)
    coefficients = np.linalg.lstsq(centered, response, rcond=1e-12)[0].T
    projection = centered @ coefficients.T
    close(diagnostic['coefficients'], coefficients)
    close(diagnostic['prevalence'], labels.mean(axis=0))
    close(diagnostic['covariance'], centered.T @ centered / len(labels))
    explained = float(np.mean(projection ** 2))
    variance = float(np.mean(response ** 2))
    close(diagnostic['explained_variance'], explained)
    close(diagnostic['constant_variance'], variance)
    if variance:
        close(diagnostic['complete_exact_fraction'], explained / variance)
    mean = targets.mean(axis=0)
    ranges = []
    for column in coefficients.T:
        high = np.dot(column, (column > 0).astype(float) - mean)
        low = np.dot(column, (column < 0).astype(float) - mean)
        ranges.append(max(abs(high), abs(low)))
    close(diagnostic['ranges'], ranges)
    prevalence = labels.mean(axis=0)
    penalty = sum(8 * scale * rate * (1 - rate) for scale, rate in zip(ranges, prevalence)) / targets.shape[1]
    assert len(diagnostic['bounds']) == len(caps)
    for cap, result in zip(caps, diagnostic['bounds']):
        assert result['cap'] == cap
        floor = explained - penalty * (cap - .5 + 2 ** -16)
        close(result['raw_loss_floor'], floor)
        close(result['loss_floor'], max(0, floor))
        if variance:
            close(result['fraction_constant_advantage_lost'], max(0, floor) / variance)
    return explained, variance


def verify(certificate, expected, configuration):
    assert certificate['config'] == configuration
    for name in ('development', 'evaluation'):
        assert len(certificate['splits'][name]) == configuration[name + '_count']
        for index, record in enumerate(certificate['splits'][name]):
            assert len(record['history']) == 7 and all(0 <= value <= 8 for value in record['history'])
            if (name, index) not in expected:
                continue
            forecasts, risks = expected[name, index]
            close(record['reference'], [float(value) for value in forecasts], 1e-12)
            assert [Fraction(value) for value in record['exact_risk']] == risks
            close(record['risk'], [float(value) for value in risks], 1e-12)
            assert record['labels'] == [int(value >= Fraction(1, 4)) for value in risks]
    evaluation = certificate['splits']['evaluation']
    targets = np.array([record['reference'] for record in evaluation])
    labels = np.array([record['labels'] for record in evaluation])
    for name, result in certificate['solvers'].items():
        predictions = np.array(result['forecasts'])
        risks = np.array(result['risks'])
        assert predictions.shape == targets.shape and risks.shape == labels.shape
        assert (predictions >= -1e-12).all() and (predictions <= 1 + 1e-12).all()
        if name == 'filtered':
            close(predictions, targets, 1e-12)
            close(risks, [record['risk'] for record in evaluation], 1e-12)
        else:
            for index, record in enumerate(evaluation):
                forecasts, probabilities = expected['last_observation', record['history'][-1]]
                close(predictions[index], [float(value) for value in forecasts], 1e-12)
                close(risks[index], [float(value) for value in probabilities], 1e-12)
        decisions = risks >= .25
        close(result['mean_squared_error'], np.mean((targets - predictions) ** 2), 1e-14)
        close(result['risk_accuracy'], np.mean(decisions == labels))
        for index, accuracy in enumerate(result['balanced_accuracy']):
            classes = [labels[:, index] == value for value in (0, 1)]
            if not all(group.any() for group in classes):
                assert accuracy is None
            else:
                close(accuracy, sum(float(np.mean(decisions[group, index] == value)) for value, group in enumerate(classes)) / 2)
        durations = np.asarray(result['query_seconds'])
        assert durations.shape == (256,) and np.isfinite(durations).all() and (durations >= 0).all()
        assert result['median_seconds'] == float(np.median(durations))
        assert result['maximum_seconds'] == float(np.max(durations))
    development_mean = np.mean([record['reference'] for record in certificate['splits']['development']], axis=0)
    baseline = float(np.mean((targets - development_mean) ** 2))
    close(certificate['development_baseline_loss'], baseline)
    audit_family(targets, labels, certificate['family'], configuration['balanced_accuracy_caps'])
    for result in certificate['family']['bounds']:
        close(result['fraction_development_advantage_lost'], result['loss_floor'] / baseline)
    for index in range(3):
        audit_family(targets, labels[:, [index]], certificate['individual'][index], configuration['balanced_accuracy_caps'])
    histories = {name: [tuple(record['history']) for record in rows] for name, rows in certificate['splits'].items()}
    assert certificate['duplicates'] == {name: len(rows) - len(set(rows)) for name, rows in histories.items()}
    assert certificate['cross_split_shared_histories'] == len(set(histories['development']) & set(histories['evaluation']))
    fractions = [Fraction(value) for record in evaluation for value in record['exact_risk']]
    assert certificate['minimum_threshold_margin'] == min(abs(float(value - Fraction(1, 4))) for value in fractions)
    assert certificate['threshold_ties'] == sum(value == Fraction(1, 4) for value in fractions)
    pairs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    for name, control in certificate['controls'].items():
        if name == 'independent':
            control_targets, control_labels, fraction = pairs, pairs, 1.
        elif name == 'duplicate':
            control_targets = np.column_stack((pairs[:, 0], 1 - pairs[:, 0]))
            control_labels = np.column_stack((pairs[:, 0], pairs[:, 0]))
            fraction = 1.
        else:
            control_targets = np.logical_xor(pairs[:, 0], pairs[:, 1]).astype(float)[:, None]
            control_labels, fraction = pairs, 0.
        audit_family(control_targets, control_labels, control['diagnostic'], [.5, .51])
        close(control['diagnostic']['complete_exact_fraction'], fraction)
        assert control['decomposition_checks'] == 20 and control['maximum_error'] < 1e-12


def main():
    directory = Path(sys.argv[1])
    certificate = json.loads((directory / 'output/certificate.json').read_text())
    configuration = json.loads((directory / 'source/config.json').read_text())
    table = future_by_paths()
    expected = {}
    for damage in range(9):
        expected['last_observation', damage] = (
            [(table[1, damage][0][index] + table[3, damage][0][index]) / 2 for index in range(48)],
            [(table[1, damage][1][index] + table[3, damage][1][index]) / 2 for index in range(3)])
    for name, records in certificate['splits'].items():
        for index, record in enumerate(records):
            if name == 'development' and index >= 8:
                continue
            posterior = posterior_by_paths(record['history'])
            forecasts = [sum((weight * table[state][0][coordinate] for state, weight in posterior.items()), Fraction(0)) for coordinate in range(48)]
            risks = [sum((weight * table[state][1][coordinate] for state, weight in posterior.items()), Fraction(0)) for coordinate in range(3)]
            expected[name, index] = forecasts, risks
    verify(certificate, expected, configuration)
    modifications = [
        ('reference', lambda value: value['splits']['evaluation'][0]['reference'].__setitem__(0, -1)),
        ('label', lambda value: value['splits']['evaluation'][0]['labels'].__setitem__(0, 2)),
        ('family_variance', lambda value: value['family'].__setitem__('explained_variance', -1)),
        ('reader_accuracy', lambda value: value['solvers']['filtered'].__setitem__('risk_accuracy', -1)),
        ('timing', lambda value: value['solvers']['filtered'].__setitem__('maximum_seconds', -1)),
        ('duplicates', lambda value: value['duplicates'].__setitem__('evaluation', -1)),
    ]
    rejected = []
    for name, modification in modifications:
        altered = copy.deepcopy(certificate)
        modification(altered)
        try:
            verify(altered, expected, configuration)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Corruption accepted: ' + name)
    result = {'passed': True, 'evaluation_histories_enumerated': 256,
              'development_histories_enumerated': 8, 'historical_paths_per_history': 384,
              'future_paths_per_start_and_action': 256, 'future_start_action_combinations': 54,
              'forecast_coordinates_checked': 264 * 48, 'risk_probabilities_checked': 264 * 3,
              'decomposition_controls': 60, 'corruptions_rejected': rejected}
    (directory / 'output/audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
