"""Validate asymmetric readers with public bounds on physical forecasts."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from audit import future_by_paths
from runner import conditional_table


def main():
    directory = Path(sys.argv[1])
    parent_content = (directory / 'parent_certificate.json').read_bytes()
    certificate = json.loads(parent_content)
    exact_table = conditional_table()
    independent_table = future_by_paths()
    assert exact_table == independent_table
    vertices = np.array([[float(value) for value in row[0]] for row in exact_table.values()])
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    targets = np.array([row['reference'] for row in certificate['splits']['evaluation']])
    labels = np.array([row['labels'] for row in certificate['splits']['evaluation']])
    assert np.all(targets >= lower - 1e-12) and np.all(targets <= upper + 1e-12)
    centered_targets = targets - targets.mean(axis=0)
    variance = np.mean(centered_targets ** 2)
    generator = np.random.default_rng(26401)
    successors = [targets, np.broadcast_to(targets.mean(axis=0), targets.shape), np.zeros_like(targets), 1 - targets]
    successors.extend(generator.random(targets.shape) for _ in range(16))
    output = {'parent_sha256': hashlib.sha256(parent_content).hexdigest(), 'physical_lower': lower.tolist(),
              'physical_upper': upper.tolist(), 'public_vertices_checked': len(vertices), 'cases': {}}
    for box_name, box_lower, box_upper in (('unit', np.zeros(48), np.ones(48)), ('physical', lower, upper)):
        box_results = {}
        for family_name, subset in [('family', [0, 1, 2])] + [('action_' + str(index + 1), [index]) for index in range(3)]:
            selected = labels[:, subset]
            prevalence = selected.mean(axis=0)
            centered_labels = selected - prevalence
            coefficients = np.linalg.lstsq(centered_labels, centered_targets, rcond=1e-12)[0].T
            projection = centered_labels @ coefficients.T
            explained = np.mean(projection ** 2)
            minimum = (np.maximum(coefficients, 0) * box_lower[:, None] + np.minimum(coefficients, 0) * box_upper[:, None]).sum(axis=0)
            maximum = (np.maximum(coefficients, 0) * box_upper[:, None] + np.minimum(coefficients, 0) * box_lower[:, None]).sum(axis=0)
            widths = maximum - minimum
            assert (widths > 0).all()
            multiplier = 4 * widths * prevalence * (1 - prevalence) / 48
            rows = []
            for cap in certificate['config']['balanced_accuracy_caps']:
                floor = max(0., float(explained - multiplier.sum() * (cap - .5 + 2 ** -16)))
                rows.append({'cap': cap, 'loss_floor': floor, 'fraction_constant_advantage_lost': floor / float(variance),
                             'fraction_development_advantage_lost': floor / certificate['development_baseline_loss']})
            maximum_rounding_error = 0.
            for forecast in successors:
                clipped = np.clip(forecast, box_lower, box_upper)
                original_loss = np.mean((targets - forecast) ** 2)
                clipped_loss = np.mean((targets - clipped) ** 2)
                assert clipped_loss <= original_loss + 1e-12
                probabilities = (clipped @ coefficients - minimum) / widths
                assert (probabilities >= -1e-12).all() and (probabilities <= 1 + 1e-12).all()
                probabilities = np.clip(probabilities, 0, 1)
                rounded = np.floor(probabilities * 65536) / 65536
                error = float(np.max(probabilities - rounded))
                assert error <= 2 ** -16 + 1e-12
                maximum_rounding_error = max(maximum_rounding_error, error)
                accuracies = np.array([.5 * (probabilities[selected[:, index] == 1, index].mean() + (1 - probabilities[selected[:, index] == 0, index]).mean()) for index in range(len(subset))])
                covariance = np.mean(centered_labels * ((clipped - targets.mean(axis=0)) @ coefficients), axis=0)
                assert np.allclose(accuracies - .5, covariance / (2 * widths * prevalence * (1 - prevalence)), atol=1e-12, rtol=1e-12)
                rounded_accuracies = np.array([.5 * (rounded[selected[:, index] == 1, index].mean() + (1 - rounded[selected[:, index] == 0, index]).mean()) for index in range(len(subset))])
                recovered_floor = explained - np.dot(multiplier, rounded_accuracies - .5 + 2 ** -16)
                assert original_loss >= recovered_floor - 1e-12
            box_results[family_name] = {'widths': widths.tolist(), 'bounds': rows, 'successors_checked': len(successors),
                                        'maximum_probability_rounding_error': maximum_rounding_error}
        output['cases'][box_name] = box_results
    output['passed'] = True
    (directory / 'output').mkdir()
    (directory / 'output/refinement.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({name: {family: result['bounds'] for family, result in rows.items()} for name, rows in output['cases'].items()}))


if __name__ == '__main__':
    main()
