"""Measure an explicitly all-source reference on the frozen public image workload."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import io
import json
import sys
import time
import numpy as np
import pyarrow.parquet as parquet
from PIL import Image
from finite_permission_design import permission_design


def build_reference(table):
    labels = table['fine_label'].to_pylist()
    coarse = table['coarse_label'].to_pylist()
    counts = defaultdict(Counter)
    mapping = {}
    for item, label, parent in zip(table['img'].to_pylist(), labels, coarse):
        with Image.open(io.BytesIO(item['bytes'])) as image:
            assert image.mode == 'RGB' and image.size == (32, 32)
            pixels = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
        key = hashlib.sha256(pixels.tobytes()).digest()
        counts[key][label] += 1
        assert label not in mapping or mapping[label] == parent
        mapping[label] = parent
    reference = {key: min(values, key=lambda label: (-values[label], label)) for key, values in counts.items()}
    packed = b''.join(key+bytes([reference[key], mapping[reference[key]]]) for key in sorted(reference))
    return reference, mapping, packed, counts


def predict_images(reference, images):
    return np.array([reference.get(hashlib.sha256(image.tobytes()).digest(), -1) for image in images], dtype=np.int64)


def main(directory):
    started = time.perf_counter()
    root = directory.parent
    source = root/'acquisition02/cifar100_test.parquet'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68'
    preparation_start = time.perf_counter()
    table = parquet.read_table(source)
    reference, mapping, packed, counts = build_reference(table)
    (directory/'public_image_reference.bin').write_bytes(packed)
    # Reconstruct the actual predictor solely from the charged serialized table.
    restored = {packed[offset:offset+32]: packed[offset+32] for offset in range(0, len(packed), 34)}
    assert restored == reference
    preparation_seconds = time.perf_counter()-preparation_start
    data = root/'data_validation01'
    record = json.loads((data/'results.json').read_text())
    for name, field in [('selected_images.npy','images_sha256'),('evaluation_labels.npz','evaluation_labels_sha256')]:
        assert hashlib.sha256((data/name).read_bytes()).hexdigest() == record[field]
    images = np.load(data/'selected_images.npy', allow_pickle=False)
    prediction_start = time.perf_counter()
    predictions = predict_images(restored, images)
    prediction_seconds = time.perf_counter()-prediction_start
    assert np.all(predictions >= 0)
    unknown = np.zeros((1, 3, 32, 32), dtype=np.uint8)
    assert hashlib.sha256(unknown[0].tobytes()).digest() not in restored
    assert predict_images(restored, unknown)[0] == -1
    labels = np.load(data/'evaluation_labels.npz', allow_pickle=False)
    coarse_predictions = np.array([mapping[label] for label in predictions])
    outcomes = []
    for count, predicted, targets in [(100, predictions, labels['fine']), (20, coarse_predictions, labels['coarse'])]:
        accuracy = float(np.mean(predicted == targets))
        masks = permission_design(count)
        risk_accuracy = float(np.mean(masks[:, predicted] == masks[:, targets]))
        expected = .5+(count*accuracy-1)/(2*(count-1))
        assert abs(risk_accuracy-expected) < 1e-12
        outcomes.append({'classes':count, 'recognition_accuracy':accuracy, 'finite_permission_accuracy':risk_accuracy,
                         'expected_uniform_half_set_accuracy':expected, 'permission_cases':int(len(masks)*len(predicted))})
    np.savez(directory/'predictions.npz', fine=predictions, coarse=coarse_predictions)
    result = {'status':'complete', 'preparation_source_records':table.num_rows, 'evaluated_records':len(images),
              'source_bytes':source.stat().st_size, 'table_bytes':len(packed), 'unique_keys':len(reference),
              'repeated_keys':sum(sum(value.values()) > 1 for value in counts.values()),
              'conflicting_label_keys':sum(len(value) > 1 for value in counts.values()),
              'reference_sha256':hashlib.sha256(packed).hexdigest(),
              'preparation_seconds':preparation_seconds, 'prediction_seconds':prediction_seconds,
              'outcomes':outcomes, 'uses_public_evaluation_annotations':True,
              'independent_generalization_evidence':False, 'genuine_removal':False,
              'neural_training':False, 'unknown_key_control':'passed', 'wall_seconds':time.perf_counter()-started}
    (directory/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
