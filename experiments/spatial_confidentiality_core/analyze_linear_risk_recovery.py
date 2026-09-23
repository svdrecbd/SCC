"""Recover threshold-risk predictions from invariant spatial summaries."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from analyze import integrated_loss, recovery_prefixes
from data import read_frame


def project(values, name):
    if name == 'even':
        return (values+values[:, ::-1])/2
    if name == 'odd':
        return (values-values[:, ::-1])/2
    if name == 'mean':
        return np.full_like(values, values.mean())
    if name == 'centered':
        return values-values.mean()
    raise ValueError(name)


def main(directory, root, reflection_root):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    selection = json.loads((directory/'selection.json').read_text())
    manifests = {}
    for name in ('original', 'reflection'):
        payload = (directory/f'{name}_manifest.json').read_bytes()
        assert hashlib.sha256(payload).hexdigest() == configuration[f'{name}_manifest_sha256']
        manifests[name] = {record['path']: record['sha256'] for record in json.loads(payload)['files']}
    def prediction_array(source_root, relative, kind):
        payload = (source_root/relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == manifests[kind][relative]
        return np.load(source_root/relative, allow_pickle=False).astype(np.float64)
    parameters = np.load(directory/'public_parameters.npz', allow_pickle=False)
    # The rectangle is fixed from the original crop and its reflection, before
    # any labels are read. Its horizontal reflection maps it onto itself.
    crop = np.s_[45:471, 41:599]
    calibration = []
    for record in selection[:16]:
        _, depth, _ = read_frame(root/'acquisition01', record)
        depth = depth.astype(np.float64)
        valid = (depth > .1) & (depth < 10) & np.isfinite(depth)
        depth[~valid] = float(parameters['global_mean'])
        calibration.append(depth[crop]/10)
    sorted_calibration = np.sort(np.stack(calibration), axis=0)
    baseline = sorted_calibration.mean(axis=0)
    records = []
    for index in configuration['selection_indices']:
        _, truth, digest = read_frame(root/'acquisition01', selection[index])
        truth = truth.astype(np.float64)[crop]/10
        valid = (truth > .01) & (truth < 1) & np.isfinite(truth)
        pair_valid = valid & valid[:, ::-1]
        native = prediction_array(root, f'inference{index//4+1:02d}/prediction_{index:03d}.npy', 'original')
        reflected = prediction_array(reflection_root, f'inference{(index-16)//4+1:02d}/reflected_prediction_{index:03d}.npy', 'reflection')
        native = np.clip(native[crop]/10, .01, 1)
        reflected = np.clip(reflected[crop]/10, .01, 1)
        complete = (native+reflected)/2
        calibration_at_pixels = sorted_calibration[:, pair_valid]
        prefixes = recovery_prefixes(truth[pair_valid], calibration_at_pixels)
        reference = integrated_loss(complete[pair_valid], truth[pair_valid], calibration_at_pixels, prefixes)
        record = {'selection_index': index, 'input_sha256': digest,
                  'paired_valid_pixels': int(pair_valid.sum()), 'crop_pixels': int(truth.size),
                  'fully_valid_crop': bool(valid.all()), 'reference': reference, 'projections': {}}
        for name in configuration['projections']:
            if name in ('mean', 'centered') and not valid.all():
                record['projections'][name] = {'status': 'excluded_invalid_labels'}
                continue
            predicted_summary = project(complete, name)
            baseline_summary = project(baseline, name)
            truth_summary = project(truth, name)
            raw_recovered = baseline+predicted_summary-baseline_summary
            recovered = np.clip(raw_recovered, 0, 1)
            useful_gain = np.mean(((baseline_summary-truth_summary)**2-(predicted_summary-truth_summary)**2)[pair_valid])
            raw_gain = np.mean(((baseline-truth)**2-(raw_recovered-truth)**2)[pair_valid])
            clipped_gain = np.mean(((baseline-truth)**2-(recovered-truth)**2)[pair_valid])
            assert abs(useful_gain-raw_gain) < 1e-12, (name, useful_gain, raw_gain)
            assert clipped_gain >= raw_gain-1e-12
            risk = integrated_loss(recovered[pair_valid], truth[pair_valid], calibration_at_pixels, prefixes)
            assert risk['protected_gain'] >= useful_gain-1e-12
            record['projections'][name] = {'status': 'complete',
                'summary_baseline_mse_metres_squared': float(np.mean((baseline_summary-truth_summary)[pair_valid]**2)*100),
                'summary_prediction_mse_metres_squared': float(np.mean((predicted_summary-truth_summary)[pair_valid]**2)*100),
                'summary_normalized_gain': float(useful_gain), 'raw_full_field_gain': float(raw_gain),
                'clipped_full_field_gain': float(clipped_gain), 'protected_gain': risk['protected_gain'],
                'protected_baseline_brier': risk['baseline_brier'],
                'protected_gain_ratio': risk['protected_gain']/reference['protected_gain'] if reference['protected_gain'] > 0 else None,
                'clipping_fraction': float(np.mean(recovered[pair_valid] != raw_recovered[pair_valid]))}
        records.append(record)
        (directory/'cases.json').write_text(json.dumps(records, indent=2)+'\n')
    summaries = {}
    for name in configuration['projections']:
        cases = [record['projections'][name] for record in records if record['projections'][name]['status'] == 'complete']
        if not cases:
            summaries[name] = {'status': 'no_valid_scenes', 'evaluated_scenes': 0}
            continue
        generator = np.random.default_rng(configuration['bootstrap_seed'])
        indices = generator.integers(0, len(cases), (configuration['bootstrap_replicates'], len(cases)))
        gains = np.array([case['protected_gain'] for case in cases])
        interval = np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist()
        summaries[name] = {'status': 'complete', 'evaluated_scenes': len(cases),
                          'excluded_scenes': len(records)-len(cases),
                          'mean_summary_gain': float(np.mean([case['summary_normalized_gain'] for case in cases])),
                          'mean_protected_gain': float(gains.mean()),
                          'protected_gain_interval': interval,
                          'positive_recovery_screen': bool(gains.mean() > 0 and interval[0] > 0),
                          'mean_clipping_fraction': float(np.mean([case['clipping_fraction'] for case in cases]))}
    summary = {'status': 'complete', 'projections': summaries,
               'scene_count': len(records), 'crop': [45,471,41,599],
               'original_parameters_unchanged': True, 'neural_training': False,
               'new_neural_executions': 0, 'genuine_removal_established': False,
               'wall_seconds': time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
