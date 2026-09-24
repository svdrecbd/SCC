"""Validate finite designs, arbitrary-reader projection and saved neural predictions."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from finite_permission_design import permission_design, recover_probabilities


def verified_array(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    return np.load(path, allow_pickle=False)


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / 'config.json').read_text())
    generator = np.random.default_rng(settings['seed'])
    tolerance = settings['tolerance']
    records = []
    projected_cases = 0
    for count in settings['orders']:
        masks = permission_design(count)
        queries = len(masks)
        assert queries == 2*(count-1)
        assert np.all(masks.sum(-1) == count//2)
        assert np.all(masks.sum(0) == count-1)
        disagreement = ((masks[:, :, None]-masks[:, None, :])**2).sum(0)
        assert np.array_equal(disagreement, count*(np.ones((count, count), dtype=np.int64)-np.eye(count, dtype=np.int64)))
        centered = masks-.5
        coefficient = count/(4*(count-1))
        moment_error = float(np.abs(centered.T @ centered/queries-coefficient*(np.eye(count)-np.ones((count,count))/count)).max())
        assert moment_error <= tolerance
        probabilities = generator.dirichlet(np.ones(count), settings['random_cases'])
        reports = probabilities @ masks.T
        recovered, raw = recover_probabilities(reports, masks)
        coherent_error = float(np.abs(recovered-probabilities).max())
        assert coherent_error <= tolerance
        direct = ((reports[:, :, None]-masks[None, :, :])**2).mean(1)
        class_loss = coefficient*((probabilities[:, None, :]-np.eye(count)[None, :, :])**2).sum(-1)
        loss_error = float(np.abs(direct-class_loss).max())
        assert loss_error <= tolerance
        corrupted = reports[:1].copy()
        corrupted[0, 0] = 1-corrupted[0, 0]
        arbitrary = np.concatenate([generator.random((settings['random_cases'], queries)),
                                    np.zeros((1, queries)), np.ones((1, queries)),
                                    np.full((1, queries), .5), 1-reports[:1], corrupted], axis=0)
        recovered, raw = recover_probabilities(arbitrary, masks)
        assert np.all(recovered >= 0) and np.allclose(recovered.sum(-1), 1, atol=tolerance, rtol=0)
        direct = ((arbitrary[:, :, None]-masks[None, :, :])**2).mean(1)
        recovered_loss = coefficient*((recovered[:, None, :]-np.eye(count)[None, :, :])**2).sum(-1)
        maximum_violation = float(np.max(recovered_loss-direct))
        assert maximum_violation <= tolerance
        projected = int(np.sum(np.any(raw < 0, axis=1)))
        projected_cases += projected
        residual = arbitrary-(raw @ masks.T)
        orthogonality_error = float(np.abs(residual @ centered/queries).max())
        assert orthogonality_error <= tolerance
        np.save(directory / f'permission_masks_{count}.npy', masks.astype(np.uint8), allow_pickle=False)
        records.append({'classes': count, 'queries': queries, 'category_pairs_checked': count*count,
                        'moment_error': moment_error, 'coherent_reconstruction_error': coherent_error,
                        'loss_identity_error': loss_error, 'arbitrary_report_vectors': len(arbitrary),
                        'all_label_inequalities_checked': len(arbitrary)*count,
                        'maximum_projection_inequality_violation': maximum_violation,
                        'orthogonality_error': orthogonality_error, 'simplex_projection_cases': projected})
    assert projected_cases > 0
    root = directory.parent
    data = root / 'data_validation01'
    data_record = json.loads((data/'results.json').read_text())
    labels = verified_array(data/'evaluation_labels.npz', data_record['evaluation_labels_sha256'])
    logits = []
    for stage, expected_start, expected_end in [('initialization01', 0, 100), ('inference01', 100, 1000)]:
        record = json.loads((root/stage/'results.json').read_text())
        assert record['start_index'] == expected_start and record['end_index'] == expected_end
        assert record['state_before'] == record['state_after'] == 'f682a4d5419ca404b8461551d1f1828e644bab04049ffb654b05b2b915ee9abd'
        assert record['input_images_sha256'] == data_record['images_sha256']
        logits.append(verified_array(root/stage/'logits.npy', record['logits_sha256']))
    logits = np.concatenate(logits).astype(np.float64)
    assert len(logits) == settings['native_cases']
    probabilities = np.exp(logits-logits.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    coarse = np.stack([probabilities[:, labels['mapping'] == label].sum(-1) for label in range(20)], axis=1)
    native = []
    for values, targets in [(probabilities, labels['fine']), (coarse, labels['coarse'])]:
        count = values.shape[1]
        masks = permission_design(count)
        reports = values @ masks.T
        # Numerical probability sums may exceed one by floating-point rounding.
        assert reports.min() >= -tolerance and reports.max() <= 1+tolerance
        recovered, _ = recover_probabilities(np.clip(reports, 0, 1), masks)
        probability_error = float(np.abs(recovered-values).max())
        assert probability_error <= tolerance
        assert np.array_equal(recovered.argmax(-1), values.argmax(-1))
        correct = masks[:, targets].T
        loss = float(((reports-correct)**2).mean())
        class_loss = float(((values-np.eye(count)[targets])**2).sum(-1).mean())
        assert abs(loss-count/(4*(count-1))*class_loss) <= tolerance
        predictions = values.argmax(-1)
        useful_accuracy = float(np.mean(predictions == targets))
        protected_accuracy = float(np.mean(masks[:, predictions].T == correct))
        expected_accuracy = .5+(count*useful_accuracy-1)/(2*(count-1))
        assert abs(protected_accuracy-expected_accuracy) <= tolerance
        native.append({'classes': count, 'queries_per_image': len(masks), 'risk_values_evaluated': len(values)*len(masks),
                       'useful_accuracy': useful_accuracy, 'protected_accuracy': protected_accuracy,
                       'expected_protected_accuracy': expected_accuracy, 'permission_brier': loss,
                       'multiclass_brier': class_loss, 'probability_reconstruction_error': probability_error})
    result = {'status': 'complete', 'design_checks': records, 'native_checks': native,
              'total_simplex_projection_cases': projected_cases, 'native_forwards': 0,
              'neural_training': False, 'new_query_contract': True,
              'all_reader_removal_certified': False, 'wall_seconds': time.perf_counter()-started}
    (directory/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
