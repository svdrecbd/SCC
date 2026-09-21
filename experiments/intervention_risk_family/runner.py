"""Finite intervention forecasts and simultaneous risk-recovery diagnostics."""
import hashlib
import json
import math
import platform
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np


def observation_weight(damage, observed):
    return sum(weight for error, weight in ((-1, 1), (0, 2), (1, 1))
               if min(8, max(0, damage + error)) == observed)


def generate(count, seed):
    generator = random.Random(seed)
    records = []
    for _ in range(count):
        numerator = generator.choice((1, 3))
        damage = generator.choice((0, 1, 2))
        initial_damage = damage
        observations = [min(8, max(0, damage + generator.choice((-1, 0, 0, 1))))]
        states = [damage]
        for _ in range(6):
            damage = min(8, damage + int(generator.randrange(8) < numerator))
            states.append(damage)
            observations.append(min(8, max(0, damage + generator.choice((-1, 0, 0, 1)))))
        records.append({'history': observations, 'hidden_numerator': numerator,
                        'hidden_initial_damage': initial_damage, 'hidden_states': states})
    return records


def posterior_exact(history):
    weights = {(numerator, damage): observation_weight(damage, history[0])
               for numerator in (1, 3) for damage in range(3)}
    for observed in history[1:]:
        successor = {}
        for (numerator, damage), weight in weights.items():
            for increment, factor in ((0, 8 - numerator), (1, numerator)):
                target = min(8, damage + increment)
                key = (numerator, target)
                successor[key] = successor.get(key, 0) + weight * factor * observation_weight(target, observed)
        weights = {key: value for key, value in successor.items() if value}
    total = sum(weights.values())
    assert total > 0
    return {key: Fraction(value, total) for key, value in weights.items()}


def conditional_table():
    table = {}
    for numerator in (1, 3):
        for damage in range(9):
            forecasts, risks = [], []
            for load in (1, 2, 3):
                for horizon in range(1, 9):
                    expected_damage = Fraction(0)
                    survival = Fraction(0)
                    for increments in range(horizon + 1):
                        probability = Fraction(math.comb(horizon, increments) * numerator ** increments * (8 - numerator) ** (horizon - increments), 8 ** horizon)
                        target = min(8, damage + load * increments)
                        expected_damage += probability * Fraction(target, 8)
                        survival += probability * int(target < 8)
                    forecasts.extend((expected_damage, survival * Fraction(load, 3)))
                risks.append(1 - survival)
            table[numerator, damage] = (forecasts, risks)
    return table


def reference(history, table):
    posterior = posterior_exact(history)
    forecasts = [sum((weight * table[state][0][index] for state, weight in posterior.items()), Fraction(0)) for index in range(48)]
    risks = [sum((weight * table[state][1][index] for state, weight in posterior.items()), Fraction(0)) for index in range(3)]
    return forecasts, risks


def forecast_distribution(distribution):
    forecasts, risks = [], []
    for load in (1, 2, 3):
        current = distribution.copy()
        for _ in range(8):
            successor = {}
            for (numerator, damage), weight in current.items():
                for increment, factor in ((0, 1 - numerator / 8), (load, numerator / 8)):
                    key = (numerator, min(8, damage + increment))
                    successor[key] = successor.get(key, 0.) + weight * factor
            current = successor
            forecasts.append(sum(weight * damage / 8 for (_, damage), weight in current.items()))
            forecasts.append(sum(weight * load / 3 for (_, damage), weight in current.items() if damage < 8))
        risks.append(sum(weight for (_, damage), weight in current.items() if damage == 8))
    return forecasts, risks


def public_solver(history, use_history=True):
    if not use_history:
        return forecast_distribution({(1, history[-1]): .5, (3, history[-1]): .5})
    distribution = {(numerator, damage): observation_weight(damage, history[0]) / 24
                    for numerator in (1, 3) for damage in range(3)}
    for observed in history[1:]:
        successor = {}
        for (numerator, damage), weight in distribution.items():
            for increment, factor in ((0, 1 - numerator / 8), (1, numerator / 8)):
                target = min(8, damage + increment)
                key = (numerator, target)
                successor[key] = successor.get(key, 0.) + weight * factor * observation_weight(target, observed) / 4
        normalizer = sum(successor.values())
        distribution = {key: value / normalizer for key, value in successor.items() if value}
    return forecast_distribution(distribution)


def family_diagnostic(targets, labels, caps):
    mean = targets.mean(axis=0)
    prevalence = labels.mean(axis=0)
    centered_labels = labels - prevalence
    centered_targets = targets - mean
    covariance = centered_labels.T @ centered_labels / len(targets)
    cross = centered_targets.T @ centered_labels / len(targets)
    coefficients = cross @ np.linalg.pinv(covariance, rcond=1e-12)
    projection = centered_labels @ coefficients.T
    residual = centered_targets - projection
    dimension = targets.shape[1]
    explained = float(np.mean(np.sum(projection ** 2, axis=1)))
    maximum = np.maximum(coefficients, 0).sum(axis=0) - mean @ coefficients
    minimum = np.minimum(coefficients, 0).sum(axis=0) - mean @ coefficients
    ranges = np.maximum(abs(maximum), abs(minimum))
    variance = float(np.mean(centered_targets ** 2))
    penalties = 8 * ranges * prevalence * (1 - prevalence)
    bounds = []
    for cap in caps:
        raw = (explained - float(penalties.sum()) * (cap - .5 + 2 ** -16)) / dimension
        bounds.append({'cap': cap, 'raw_loss_floor': raw, 'loss_floor': max(0., raw),
                       'fraction_constant_advantage_lost': max(0., raw) / variance if variance else None})
    return {'prevalence': prevalence.tolist(), 'covariance': covariance.tolist(),
            'coefficients': coefficients.tolist(), 'ranges': ranges.tolist(),
            'explained_variance': explained / dimension, 'constant_variance': variance,
            'complete_exact_fraction': explained / dimension / variance if variance else None,
            'bounds': bounds, 'projection_orthogonality_maximum': float(np.max(abs(projection.T @ residual / len(targets))))}


def algebra_controls():
    pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    duplicate = np.column_stack((pairs[:, 0], pairs[:, 0]))
    cases = [('independent', pairs, pairs),
             ('duplicate', np.column_stack((pairs[:, 0], 1 - pairs[:, 0])), duplicate),
             ('interaction', np.logical_xor(pairs[:, 0], pairs[:, 1]).astype(float)[:, None], pairs)]
    results = {}
    generator = np.random.default_rng(26303)
    for name, targets, labels in cases:
        diagnostic = family_diagnostic(targets, labels, [.5, .51])
        mean = targets.mean(axis=0)
        centered = labels - labels.mean(axis=0)
        coefficients = np.asarray(diagnostic['coefficients'])
        projection = centered @ coefficients.T
        residual = targets - mean - projection
        errors = []
        for forecast in [np.zeros_like(targets), np.full_like(targets, .5), targets, 1 - targets] + [generator.random(targets.shape) for _ in range(16)]:
            loss = np.mean(np.sum((targets - forecast) ** 2, axis=1))
            decomposed = np.mean(np.sum(projection ** 2, axis=1)) - 2 * np.mean(np.sum(projection * (forecast - mean), axis=1)) + np.mean(np.sum((residual - (forecast - mean)) ** 2, axis=1))
            errors.append(abs(float(loss - decomposed)))
        results[name] = {'diagnostic': diagnostic, 'decomposition_checks': len(errors), 'maximum_error': max(errors)}
    return results


def main():
    directory = Path(sys.argv[1])
    config = json.loads((directory / 'source/config.json').read_text())
    table = conditional_table()
    splits = {}
    for name in ('development', 'evaluation'):
        records = generate(config[name + '_count'], config[name + '_seed'])
        for record in records:
            forecasts, risks = reference(record['history'], table)
            record.update(reference=[float(value) for value in forecasts],
                          risk=[float(value) for value in risks],
                          exact_risk=[str(value) for value in risks],
                          labels=[int(value >= Fraction(1, 4)) for value in risks])
        splits[name] = records
    records = splits['evaluation']
    targets = np.asarray([record['reference'] for record in records])
    labels = np.asarray([record['labels'] for record in records])
    development_mean = np.mean([record['reference'] for record in splits['development']], axis=0)
    results = {}
    for name, use_history in (('filtered', True), ('last_observation', False)):
        durations, predictions, risk_predictions = [], [], []
        public_solver(records[0]['history'], use_history)
        for record in records:
            started = time.perf_counter_ns()
            prediction, risk = public_solver(record['history'], use_history)
            durations.append((time.perf_counter_ns() - started) * 1e-9)
            predictions.append(prediction)
            risk_predictions.append(risk)
        decisions = np.asarray(risk_predictions) >= .25
        accuracies = []
        for index in range(3):
            present = [labels[:, index] == value for value in (0, 1)]
            accuracies.append(float(np.mean([np.mean(decisions[mask, index] == value) for value, mask in enumerate(present)])) if all(mask.any() for mask in present) else None)
        results[name] = {'forecasts': predictions, 'risks': risk_predictions, 'query_seconds': durations,
                         'mean_squared_error': float(np.mean((targets - predictions) ** 2)),
                         'risk_accuracy': float(np.mean(decisions == labels)), 'balanced_accuracy': accuracies,
                         'median_seconds': float(np.median(durations)), 'maximum_seconds': max(durations)}
    diagnostic = family_diagnostic(targets, labels, config['balanced_accuracy_caps'])
    development_loss = float(np.mean((targets - development_mean) ** 2))
    for bound in diagnostic['bounds']:
        bound['fraction_development_advantage_lost'] = bound['loss_floor'] / development_loss
    histories = {name: [tuple(record['history']) for record in rows] for name, rows in splits.items()}
    certificate = {'config': config, 'splits': splits, 'solvers': results,
                   'development_baseline_loss': development_loss, 'family': diagnostic,
                   'individual': [family_diagnostic(targets, labels[:, [index]], config['balanced_accuracy_caps']) for index in range(3)],
                   'controls': algebra_controls(),
                   'duplicates': {name: len(rows) - len(set(rows)) for name, rows in histories.items()},
                   'cross_split_shared_histories': len(set(histories['development']) & set(histories['evaluation'])),
                   'minimum_threshold_margin': min(abs(float(Fraction(value) - Fraction(1, 4))) for record in records for value in record['exact_risk']),
                   'threshold_ties': sum(Fraction(value) == Fraction(1, 4) for record in records for value in record['exact_risk'])}
    output = directory / 'output'
    output.mkdir()
    (output / 'certificate.json').write_text(json.dumps(certificate, indent=2) + '\n')
    (output / 'machine.json').write_text(json.dumps({'platform': platform.platform(), 'python': sys.version, 'numpy': np.__version__}, indent=2) + '\n')
    print(json.dumps({'solvers': {name: {key: value for key, value in row.items() if key not in ('forecasts', 'risks', 'query_seconds')} for name, row in results.items()},
                      'family': diagnostic, 'duplicates': certificate['duplicates'], 'cross_split_shared_histories': certificate['cross_split_shared_histories'],
                      'margin': certificate['minimum_threshold_margin']}))


if __name__ == '__main__':
    main()
