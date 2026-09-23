"""Measure same-scene recovery from a reflected-input retained procedure."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from analyze import features, integrated_loss, recovery_prefixes
from data import read_frame


def main(directory, root, inference_root):
    started = time.perf_counter()
    configuration = json.loads((directory / 'config.json').read_text())
    selection = json.loads((directory / 'selection.json').read_text())
    parameters = np.load(directory / 'public_parameters.npz', allow_pickle=False)
    calibration = []
    for record in selection[:16]:
        _, depth, _ = read_frame(root / 'acquisition01', record)
        depth = depth.astype(np.float64)
        depth[~((depth > .1) & (depth < 10) & np.isfinite(depth))] = float(parameters['global_mean'])
        calibration.append(depth)
    sorted_calibration = np.sort(np.stack(calibration)/10, axis=0)
    coordinates = np.meshgrid(np.linspace(-1, 1, 640), np.linspace(-1, 1, 480))
    crop = np.zeros((480, 640), dtype=bool)
    crop[45:471, 41:601] = True
    missing_half = np.broadcast_to(np.arange(640)[None] >= 320, (480, 640))
    records = []
    for index in configuration['selection_indices']:
        image, truth, digest = read_frame(root / 'acquisition01', selection[index])
        truth = truth.astype(np.float64)
        original = np.clip(np.load(root / f'inference{index//4+1:02d}' / f'prediction_{index:03d}.npy').astype(np.float64), .1, 10)
        transformed = np.clip(np.load(inference_root / f'inference{(index-16)//4+1:02d}' / f'reflected_prediction_{index:03d}.npy').astype(np.float64), .1, 10)
        public = np.clip(features(image, coordinates)[:, :, :6] @ parameters['position_coefficients'], .1, 10)
        mask = crop & (truth > .1) & (truth < 10) & np.isfinite(truth)
        missing_mask = mask & missing_half
        recovered = np.where(missing_half, transformed, original)
        filled = np.where(missing_half, public, original)
        calibration_at_pixels = sorted_calibration[:, mask]
        prefixes = recovery_prefixes(truth[mask]/10, calibration_at_pixels)
        missing_calibration = sorted_calibration[:, missing_mask]
        missing_prefixes = recovery_prefixes(truth[missing_mask]/10, missing_calibration)
        readers = {}
        for name, prediction in {'native': original, 'reflected': transformed,
                                 'reconstructed': recovered, 'public_fill': filled}.items():
            readers[name] = {'mse': float(np.mean((prediction[mask]-truth[mask])**2)),
                             'missing_half_mse': float(np.mean((prediction[missing_mask]-truth[missing_mask])**2)),
                             'recovery': integrated_loss(prediction[mask]/10, truth[mask]/10,
                                                         calibration_at_pixels, prefixes),
                             'missing_half_recovery': integrated_loss(prediction[missing_mask]/10, truth[missing_mask]/10,
                                                                      missing_calibration, missing_prefixes)}
        # The exact reconstruction error to the original is solely the measured
        # equivariance defect on the missing half, not an assumed zero.
        reconstruction_error = np.mean((recovered[mask]-original[mask])**2)
        selected_error = np.mean(np.where(missing_half, (transformed-original)**2, 0)[mask])
        assert abs(reconstruction_error-selected_error) < 1e-12
        records.append({'selection_index': index, 'input_sha256': digest,
                        'equivariance_mse': float(np.mean((transformed[mask]-original[mask])**2)),
                        'reconstruction_to_native_mse': float(reconstruction_error), 'readers': readers})
    native = np.mean([record['readers']['native']['mse'] for record in records])
    old = json.loads((root / 'analysis01' / 'summary.json').read_text())
    assert abs(native-old['readers']['native']['mse']) < 1e-8
    summary_readers = {name: {'mse': float(np.mean([r['readers'][name]['mse'] for r in records])),
                              'missing_half_mse': float(np.mean([r['readers'][name]['missing_half_mse'] for r in records])),
                              'protected_gain': float(np.mean([r['readers'][name]['recovery']['protected_gain'] for r in records])),
                              'missing_half_protected_gain': float(np.mean([r['readers'][name]['missing_half_recovery']['protected_gain'] for r in records]))}
                      for name in records[0]['readers']}
    advantage = np.array([r['readers']['public_fill']['missing_half_mse']-r['readers']['reconstructed']['missing_half_mse'] for r in records])
    generator = np.random.default_rng(configuration['bootstrap_seed'])
    indices = generator.integers(0, len(records), (configuration['bootstrap_replicates'], len(records)))
    interval = np.quantile(advantage[indices].mean(axis=1), [.025, .975]).tolist()
    passed = bool(advantage.mean() > 0 and interval[0] > 0 and summary_readers['reconstructed']['protected_gain'] > 0)
    summary = {'status': 'complete', 'readers': summary_readers,
               'missing_half_advantage_interval': interval,
               'mean_equivariance_mse': float(np.mean([r['equivariance_mse'] for r in records])),
               'mean_reconstruction_to_native_mse': float(np.mean([r['reconstruction_to_native_mse'] for r in records])),
               'recovery_screen_pass': passed, 'genuine_removal_test': False,
               'neural_training': False, 'evaluation_scenes': len(records),
               'wall_seconds': time.perf_counter()-started}
    (directory / 'cases.json').write_text(json.dumps(records, indent=2)+'\n')
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
