"""Independent data, physical-reference, decision and accounting checks."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def physical_reference():
    # Solve Q = UA * logarithmic mean temperature difference by bisection.
    lower, upper = 0.0, 55 * 3800.0
    for _ in range(100):
        heat = (lower + upper) / 2
        first_difference = 55 - heat / 3800
        second_difference = 55 - heat / 7000
        mean_difference = (second_difference - first_difference) / math.log(second_difference / first_difference)
        if heat < 40000 * mean_difference:
            lower = heat
        else:
            upper = heat
    return 25 + (lower + upper) / (2 * 3800)


def verify(certificate, directory):
    config = json.loads((directory / 'source/config.json').read_text())
    assert certificate['config'] == config
    content = (directory / 'upstream/data/observations.npz').read_bytes()
    identity = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    assert identity == config['observation_git_blob'] == certificate['observation_git_blob']
    assert len(content) == config['observation_bytes']
    assert certificate['observation_sha256'] == hashlib.sha256(content).hexdigest()
    reference = physical_reference()
    flow_limit = config['standard_error_multiplier'] * math.sqrt(2 * config['meter_noise'] ** 2 / config['time_steps'])
    temperature_limit = config['standard_error_multiplier'] * config['temperature_noise'] / math.sqrt(config['time_steps'])
    assert abs(certificate['clean_cold_temperature'] - reference) < 1e-10
    assert abs(certificate['flow_threshold'] - flow_limit) < 1e-14
    assert abs(certificate['temperature_threshold'] - temperature_limit) < 1e-14
    archive = np.load(directory / 'upstream/data/observations.npz', allow_pickle=False)
    assert set(archive.files) == set(config['scenario_modes'])
    true_modes, predicted_modes, digests = [], [], []
    offset = 0
    for scenario, mode in config['scenario_modes'].items():
        observations = archive[scenario]
        assert observations.shape == (500, 700) and np.isfinite(observations).all()
        values = observations.astype(np.float64).reshape(500, 7, 100)
        flow = (values[:, 4] - values[:, 5]).mean(axis=1)
        temperature = reference - values[:, 3].mean(axis=1)
        predictions = np.where(flow > flow_limit, 2, np.where(temperature > temperature_limit, 1, 0))
        for index in range(500):
            row = certificate['records'][offset + index]
            digest = hashlib.sha256(observations[index].tobytes()).hexdigest()
            digests.append(digest)
            assert (row['scenario'], row['index'], row['mode']) == (scenario, index, mode)
            assert row['record_sha256'] == digest
            assert row['prediction'] == predictions[index]
            assert abs(row['flow_deficit'] - flow[index]) < 1e-12
            assert abs(row['temperature_deficit'] - temperature[index]) < 1e-10
        expected = {'mode_accuracy': float(np.mean(predictions == mode)),
                    'predicted_modes': [int(np.sum(predictions == value)) for value in range(3)]}
        assert certificate['scenarios'][scenario] == expected
        true_modes.extend([mode] * 500)
        predicted_modes.extend(predictions.tolist())
        offset += 500
    assert len(certificate['records']) == offset == 3000
    assert certificate['duplicates'] == len(digests) - len(set(digests))
    true_modes, predicted_modes = np.array(true_modes), np.array(predicted_modes)
    confusion = np.bincount(3 * true_modes + predicted_modes, minlength=9).reshape(3, 3).tolist()
    assert certificate['mode_confusion'] == confusion
    durations = np.asarray(certificate['query_durations_seconds'])
    assert durations.shape == (3000,) and np.isfinite(durations).all() and (durations >= 0).all()
    timing = {'median': float(np.median(durations)), 'p95': float(np.sort(durations)[2849]),
              'maximum': float(np.max(durations))}
    assert certificate['query_seconds'] == timing
    for target in ('fault', 'leak'):
        truth = true_modes != 0 if target == 'fault' else true_modes == 2
        prediction = predicted_modes != 0 if target == 'fault' else predicted_modes == 2
        confusion = np.bincount(2 * truth.astype(int) + prediction.astype(int), minlength=4).reshape(2, 2)
        recalls = [float(confusion[index, index] / confusion[index].sum()) for index in (0, 1)]
        assert certificate['metrics'][target] == {'confusion': confusion.tolist(), 'recalls': recalls, 'balanced_accuracy': sum(recalls) / 2}
        decision = sum(recalls) / 2 > config['balanced_accuracy_cap'] and timing['maximum'] <= config['reader_budget_seconds']
        assert certificate['removal_cap_rejected'][target] == decision


def main():
    directory = Path(sys.argv[1])
    certificate = json.loads((directory / 'output/certificate.json').read_text())
    verify(certificate, directory)
    # Test production reader on prescribed noiseless clean, fouled and leaked records.
    from runner import classify
    reference = physical_reference()
    controls = []
    for expected, mass_out, cold_out in ((0, 2.0, reference), (1, 2.0, reference - 1), (2, 1.8, reference)):
        record = [80.] * 100 + [25.] * 100 + [50.] * 100 + [cold_out] * 100 + [2.] * 100 + [mass_out] * 100 + [.35] * 100
        assert classify(record, reference, certificate['flow_threshold'], certificate['temperature_threshold'])[0] == expected
        controls.append(expected)
    modifications = [
        ('decision', lambda value: value['records'][0].__setitem__('prediction', 9)),
        ('label', lambda value: value['records'][0].__setitem__('mode', 9)),
        ('statistic', lambda value: value['records'][0].__setitem__('flow_deficit', 999)),
        ('confusion', lambda value: value['mode_confusion'][0].__setitem__(0, -1)),
        ('accuracy', lambda value: value['metrics']['fault'].__setitem__('balanced_accuracy', -1)),
        ('identity', lambda value: value.__setitem__('observation_git_blob', 'incorrect')),
    ]
    rejected = []
    for name, modification in modifications:
        altered = copy.deepcopy(certificate)
        modification(altered)
        try:
            verify(altered, directory)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Corruption accepted: ' + name)
    result = {'passed': True, 'records_verified': 3000, 'synthetic_controls': controls, 'corruptions_rejected': rejected}
    (directory / 'output/audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
