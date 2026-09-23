"""Validate scene-relative confidentiality semantics and ordinal recovery."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from analyze import features, thumbnail
from data import read_frame


def optical_observation(secret, reference_depth, destination_depth):
    return secret if reference_depth < destination_depth else 0


def exact_validation():
    examples = []
    for exponent in [2, 4, 8, 16, 24]:
        epsilon = Fraction(1, 2**exponent)
        depths = [Fraction(1, 2)-epsilon, Fraction(1, 2)+epsilon]
        boundaries = [Fraction(0), *depths, Fraction(1)]
        baseline_loss = Fraction(0)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            threshold = (left+right)/2
            labels = [int(depth > threshold) for depth in depths]
            baseline = Fraction(sum(labels), len(labels))
            baseline_loss += (right-left)*sum((baseline-label)**2 for label in labels)/2
        assert baseline_loss == epsilon/2
        relative_labels = [optical_observation(1, *depths), optical_observation(1, *depths[::-1])]
        assert sorted(relative_labels) == [0, 1]
        examples.append({'epsilon': str(epsilon), 'absolute_brier_gain': str(baseline_loss),
                         'relative_brier_gain': '1/4', 'ordinal_accuracy': 1})
    checks = 0
    incorrect_sign_rejections = 0
    equality_cases = 0
    for first, second in product([Fraction(index, 4) for index in range(1, 5)], repeat=2):
        observation_zero = optical_observation(0, first, second)
        observation_one = optical_observation(1, first, second)
        violation = observation_zero != observation_one
        assert violation == (first < second)
        incorrect_sign_rejections += violation != (first > second)
        equality_cases += first == second
        checks += 1
    assert incorrect_sign_rejections == 12 and equality_cases == 4
    return {'two_world_controls': examples, 'exact_depth_pair_checks': checks,
            'secret_observations_checked': 2*checks,
            'incorrect_sign_rejections': incorrect_sign_rejections,
            'equal_depth_cases': equality_cases}


def score_values(values, first, second, labels, physical_labels):
    comparison = np.sign(values.ravel()[second]-values.ravel()[first])
    probability = .5+.25*comparison
    classification_probability = .5+.5*comparison
    utility = np.where(labels, classification_probability, 1-classification_probability)
    protected = np.where(physical_labels, classification_probability, 1-classification_probability)
    assert np.array_equal(utility, protected)
    return {'order_accuracy': float(utility.mean()),
            'violation_accuracy': float(protected.mean()),
            'fixed_confidence_brier': float(np.mean((probability-labels)**2)),
            'prediction_ties': int((comparison == 0).sum())}


def main(directory, prior_root):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    selection = json.loads((directory/'selection.json').read_text())
    exact = exact_validation()
    (directory/'exact_validation.json').write_text(json.dumps(exact, indent=2)+'\n')
    parameters = np.load(directory/'public_parameters.npz', allow_pickle=False)
    count = configuration['calibration_scenes']
    calibration = np.stack([read_frame(prior_root/'acquisition01', record)[1].astype(np.float64)
                            for record in selection[:count]])
    invalid = ~((calibration > .1) & (calibration < 10) & np.isfinite(calibration))
    calibration[invalid] = parameters['global_mean']
    crop = np.zeros((480, 640), dtype=bool)
    crop[45:471, 41:601] = True
    coordinates = np.meshgrid(np.linspace(-1, 1, 640), np.linspace(-1, 1, 480))
    public_names = ['global_mean', 'spatial_mean', 'quadratic_position', 'position_rgb',
                    'nearest_image', 'three_nearest_images']
    records = []
    for index in range(count, len(selection)):
        rgb, depth, input_digest = read_frame(prior_root/'acquisition01', selection[index])
        depth = depth.astype(np.float64)
        valid = crop & (depth > .1) & (depth < 10) & np.isfinite(depth)
        locations = np.flatnonzero(valid)
        generator = np.random.default_rng(configuration['pair_seed']+index)
        pairs = generator.choice(locations, size=(2, configuration['pairs_per_scene']), replace=True)
        unequal = depth.ravel()[pairs[0]] != depth.ravel()[pairs[1]]
        first_initial, second_initial = pairs[:, unequal]
        first = np.concatenate([first_initial, second_initial])
        second = np.concatenate([second_initial, first_initial])
        reference_depth = depth.ravel()[first]
        destination_depth = depth.ravel()[second]
        labels = reference_depth < destination_depth
        observation_zero = np.zeros_like(reference_depth)
        observation_one = np.where(reference_depth < destination_depth, 1, 0)
        physical_labels = observation_zero != observation_one
        assert np.array_equal(labels, physical_labels) and labels.mean() == .5
        prediction_file = prior_root/f'inference{index//4+1:02d}'/f'prediction_{index:03d}.npy'
        native = np.clip(np.load(prediction_file, allow_pickle=False).astype(np.float64), .1, 10)
        design = features(rgb, coordinates)
        distances = np.mean((parameters['thumbnails']-thumbnail(rgb))**2, axis=1)
        neighbors = np.argsort(distances, kind='stable')[:3]
        weights = 1/(distances[neighbors]+1e-8)
        weights /= weights.sum()
        sorted_native = np.sort(native[crop])
        ranks = (np.searchsorted(sorted_native, native, side='left')
                 +np.searchsorted(sorted_native, native, side='right'))/(2*len(sorted_native))
        predictors = {
            'native': native, 'affine': .5*native+1, 'centered': native-native[crop].mean(),
            'rank': ranks, 'reversed': -native, 'reversed_restored': -(-native),
            'constant_output': np.zeros_like(native), 'constant_output_recovered': native,
            'global_mean': np.full_like(native, parameters['global_mean']),
            'spatial_mean': parameters['spatial_mean'],
            'quadratic_position': design[..., :6] @ parameters['position_coefficients'],
            'position_rgb': design @ parameters['rgb_coefficients'],
            'nearest_image': calibration[neighbors[0]],
            'three_nearest_images': np.einsum('n,nhw->hw', weights, calibration[neighbors]),
        }
        for name in public_names:
            predictors[name] = np.clip(predictors[name], .1, 10)
        scores = {name: score_values(values, first, second, labels, physical_labels)
                  for name, values in predictors.items()}
        for name in ['affine', 'centered', 'rank', 'reversed_restored', 'constant_output_recovered']:
            assert scores[name] == scores['native'], name
        assert abs(scores['native']['order_accuracy']+scores['reversed']['order_accuracy']-1) < 1e-12
        assert scores['constant_output']['order_accuracy'] == .5
        records.append({'selection_index': index, 'group': selection[index]['group'],
                        'input_sha256': input_digest,
                        'prediction_sha256': hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
                        'requested_pairs': configuration['pairs_per_scene'],
                        'excluded_truth_ties': int((~unequal).sum()), 'oriented_pairs': len(first),
                        'label_prevalence': float(labels.mean()), 'readers': scores})
    generator = np.random.default_rng(configuration['bootstrap_seed'])
    indices = generator.integers(0, len(records), (configuration['bootstrap_replicates'], len(records)))
    summaries = {}
    for name in records[0]['readers']:
        accuracies = np.array([record['readers'][name]['order_accuracy'] for record in records])
        summaries[name] = {'order_accuracy': float(accuracies.mean()),
                           'violation_accuracy': float(accuracies.mean()),
                           'accuracy_interval': np.quantile(accuracies[indices].mean(axis=1), [.025, .975]).tolist(),
                           'fixed_confidence_brier': float(np.mean([record['readers'][name]['fixed_confidence_brier'] for record in records]))}
    native_scores = np.array([record['readers']['native']['order_accuracy'] for record in records])
    public_scores = np.array([[record['readers'][name]['order_accuracy'] for name in public_names] for record in records])
    differences = native_scores[indices].mean(axis=1)-public_scores[indices].mean(axis=1).max(axis=1)
    summary = {'status': 'complete', 'evaluation_scenes': len(records),
               'oriented_pairs': sum(record['oriented_pairs'] for record in records),
               'excluded_truth_ties': sum(record['excluded_truth_ties'] for record in records),
               'readers': summaries,
               'selection_aware_accuracy_advantage_interval': np.quantile(differences, [.025, .975]).tolist(),
               'best_public_reader': max(public_names, key=lambda name: summaries[name]['order_accuracy']),
               'reference_map_bytes': int(calibration.nbytes),
               'invalid_calibration_values_imputed': int(invalid.sum()),
               'neural_execution': False, 'neural_training': False,
               'actual_all_reader_removal_established': False,
               'physical_actuation_demonstrated': False, 'wall_seconds': time.perf_counter()-started}
    (directory/'cases.jsonl').write_text(''.join(json.dumps(record)+'\n' for record in records))
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
