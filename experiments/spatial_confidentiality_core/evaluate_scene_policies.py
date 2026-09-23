"""Evaluate fixed scene-dependent controllers without new neural execution."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from analyze import features, thumbnail
from data import read_frame


def terminal_register(observations, kind, length):
    counts = observations[:, :length].sum(axis=1)
    if kind == 'xor':
        return (counts % 2).astype(bool)
    if kind == 'majority':
        return counts > length/2
    raise ValueError('Unrecognized controller specification.')


def observations_from_depth(depth, pairs):
    values = depth.ravel()
    return values[pairs[:, :, 0]] < values[pairs[:, :, 1]]


def main(directory, prior_root):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    selection = json.loads((directory/'selection.json').read_text())
    parameters = np.load(directory/'public_parameters.npz', allow_pickle=False)
    count = configuration['calibration_scenes']
    calibration = np.stack([read_frame(prior_root/'acquisition01', row)[1].astype(np.float64)
                            for row in selection[:count]])
    invalid = ~((calibration > .1) & (calibration < 10) & np.isfinite(calibration))
    calibration[invalid] = parameters['global_mean']
    crop = np.zeros((480, 640), dtype=bool)
    crop[45:471, 41:601] = True
    coordinates = np.meshgrid(np.linspace(-1, 1, 640), np.linspace(-1, 1, 480))
    specifications = [(kind, length) for kind in ['xor', 'majority'] for length in configuration['lengths']]
    calibration_frequencies = {f'{kind}_{length:02d}': [] for kind, length in specifications}
    records = []
    public_names = ['global_mean', 'spatial_mean', 'quadratic_position', 'position_rgb', 'nearest_image', 'three_nearest_images', 'calibrated_constant']
    for index, selected in enumerate(selection):
        rgb, truth, input_digest = read_frame(prior_root/'acquisition01', selected)
        truth = truth.astype(np.float64)
        valid = crop & (truth > .1) & (truth < 10) & np.isfinite(truth)
        locations = np.flatnonzero(valid)
        generator = np.random.default_rng(configuration['program_seed']+index)
        pairs = generator.choice(locations, (configuration['programs_per_scene'], max(configuration['lengths']), 2), replace=True).astype(np.int32)
        truth_observations = observations_from_depth(truth, pairs)
        np.savez_compressed(directory/f'programs_{index:03d}.npz', pixel_pairs=pairs)
        truths = {f'{kind}_{length:02d}': terminal_register(truth_observations, kind, length)
                  for kind, length in specifications}
        if index < count:
            for name, values in truths.items():
                calibration_frequencies[name].append(float(values.mean()))
            continue
        if index == count:
            calibration_frequencies = {name: float(np.mean(values)) for name, values in calibration_frequencies.items()}
            (directory/'calibration_frequencies.json').write_text(json.dumps(calibration_frequencies, indent=2)+'\n')
        prediction_file = prior_root/f'inference{index//4+1:02d}'/f'prediction_{index:03d}.npy'
        native = np.clip(np.load(prediction_file, allow_pickle=False).astype(np.float64), .1, 10)
        design = features(rgb, coordinates)
        distances = np.mean((parameters['thumbnails']-thumbnail(rgb))**2, axis=1)
        neighbors = np.argsort(distances, kind='stable')[:3]
        weights = 1/(distances[neighbors]+1e-8)
        weights /= weights.sum()
        predictors = {
            'native': native, 'global_mean': np.full_like(native, parameters['global_mean']),
            'spatial_mean': parameters['spatial_mean'],
            'quadratic_position': design[..., :6] @ parameters['position_coefficients'],
            'position_rgb': design @ parameters['rgb_coefficients'],
            'nearest_image': calibration[neighbors[0]],
            'three_nearest_images': np.einsum('n,nhw->hw', weights, calibration[neighbors]),
        }
        comparisons = {name: observations_from_depth(np.clip(values, .1, 10), pairs) for name, values in predictors.items()}
        record = {'selection_index': index, 'group': selected['group'], 'input_sha256': input_digest,
                  'prediction_sha256': hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
                  'program_pairs_sha256': hashlib.sha256(pairs.tobytes()).hexdigest(), 'policies': {}}
        for kind, length in specifications:
            name = f'{kind}_{length:02d}'
            target = truths[name]
            # Two counterfactual executions of the declared conditional disclosure.
            observation_secret_zero = np.where(target, 0, 0)
            observation_secret_one = np.where(target, 1, 0)
            violation = observation_secret_zero != observation_secret_one
            assert np.array_equal(violation, target)
            scores = {}
            for reader, values in comparisons.items():
                prediction = terminal_register(values, kind, length)
                useful_accuracy = float(np.mean(prediction == target))
                protected_accuracy = float(np.mean(prediction == violation))
                assert useful_accuracy == protected_accuracy
                scores[reader] = useful_accuracy
            constant_prediction = calibration_frequencies[name] > .5
            scores['calibrated_constant'] = float(np.mean(target == constant_prediction))
            record['policies'][name] = {'label_prevalence': float(target.mean()), 'reader_accuracies': scores}
        records.append(record)
    generator = np.random.default_rng(configuration['bootstrap_seed'])
    indices = generator.integers(0, len(records), (configuration['bootstrap_replicates'], len(records)))
    summaries = {}
    for kind, length in specifications:
        name = f'{kind}_{length:02d}'
        native_scores = np.array([record['policies'][name]['reader_accuracies']['native'] for record in records])
        public_scores = np.array([[record['policies'][name]['reader_accuracies'][reader] for reader in public_names] for record in records])
        differences = native_scores[indices].mean(axis=1)-public_scores[indices].mean(axis=1).max(axis=1)
        averages = {reader: float(np.mean([record['policies'][name]['reader_accuracies'][reader] for record in records]))
                    for reader in ['native', *public_names]}
        summaries[name] = {'useful_and_violation_accuracies': averages,
                           'label_prevalence': float(np.mean([record['policies'][name]['label_prevalence'] for record in records])),
                           'calibration_prevalence': calibration_frequencies[name],
                           'native_accuracy_interval': np.quantile(native_scores[indices].mean(axis=1), [.025, .975]).tolist(),
                           'selection_aware_advantage_interval': np.quantile(differences, [.025, .975]).tolist(),
                           'best_public_reader': max(public_names, key=averages.get),
                           'world_depth_observations_if_executed': 2*length,
                           'controller_updates_if_executed': length}
    summary = {'status': 'complete', 'evaluation_scenes': len(records),
               'programs_per_scene_and_specification': configuration['programs_per_scene'],
               'specifications': summaries, 'reference_map_bytes': int(calibration.nbytes),
               'truth_depth_values_given_to_reader': False,
               'neural_execution': False, 'neural_training': False,
               'actual_all_reader_removal_established': False, 'training_admitted': False,
               'wall_seconds': time.perf_counter()-started}
    (directory/'cases.jsonl').write_text(''.join(json.dumps(record)+'\n' for record in records))
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'status': 'complete', 'wall_seconds': summary['wall_seconds'],
                      'policies': {name: {'native': row['useful_and_violation_accuracies']['native'],
                                          'best_public': row['useful_and_violation_accuracies'][row['best_public_reader']],
                                          'best_public_reader': row['best_public_reader'],
                                          'interval': row['selection_aware_advantage_interval'],
                                          'prevalence': row['label_prevalence']}
                                   for name, row in summaries.items()}}, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
