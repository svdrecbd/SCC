"""Evaluate fixed physical readers on a pinned external observation archive."""
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def classify(record, reference, flow_threshold, temperature_threshold):
    flow_deficit = math.fsum(record[400:500]) / 100 - math.fsum(record[500:600]) / 100
    temperature_deficit = reference - math.fsum(record[300:400]) / 100
    leak = flow_deficit > flow_threshold
    fouling = temperature_deficit > temperature_threshold
    return (2 if leak else 1 if fouling else 0), flow_deficit, temperature_deficit


def summarize(records, target):
    matrix = [[0, 0], [0, 0]]
    for record in records:
        truth = int(record['mode'] != 0) if target == 'fault' else int(record['mode'] == 2)
        prediction = int(record['prediction'] != 0) if target == 'fault' else int(record['prediction'] == 2)
        matrix[truth][prediction] += 1
    recalls = [matrix[index][index] / sum(matrix[index]) for index in (0, 1)]
    return {'confusion': matrix, 'recalls': recalls, 'balanced_accuracy': sum(recalls) / 2}


def main():
    directory = Path(sys.argv[1])
    config = json.loads((directory / 'source/config.json').read_text())
    archive_path = directory / 'upstream/data/observations.npz'
    archive_bytes = archive_path.read_bytes()
    git_blob = hashlib.sha1(b'blob ' + str(len(archive_bytes)).encode() + b'\0' + archive_bytes).hexdigest()
    assert git_blob == config['observation_git_blob']
    assert len(archive_bytes) == config['observation_bytes']
    ratio = 3800 / 7000
    exponential = math.exp(-(40000 / 3800) * (1 - ratio))
    effectiveness = (1 - exponential) / (1 - ratio * exponential)
    reference = 25 + effectiveness * 55
    flow_threshold = 3 * math.sqrt(2) * 0.02 / 10
    temperature_threshold = 3 * 0.2 / 10
    records, durations, hashes = [], [], []
    scenarios = {}
    archive = np.load(archive_path, allow_pickle=False)
    assert set(archive.files) == set(config['scenario_modes'])
    for scenario, mode in config['scenario_modes'].items():
        observations = archive[scenario]
        assert observations.shape == (500, 700) and np.isfinite(observations).all()
        for index, observation in enumerate(observations):
            record = observation.astype(float).tolist()
            if index == 0:
                classify(record, reference, flow_threshold, temperature_threshold)
            start = time.perf_counter_ns()
            prediction, flow_deficit, temperature_deficit = classify(record, reference, flow_threshold, temperature_threshold)
            durations.append((time.perf_counter_ns() - start) * 1e-9)
            digest = hashlib.sha256(observation.tobytes()).hexdigest()
            hashes.append(digest)
            records.append(dict(scenario=scenario, index=index, mode=mode, prediction=prediction,
                                flow_deficit=flow_deficit, temperature_deficit=temperature_deficit,
                                record_sha256=digest))
        selected = records[-500:]
        scenarios[scenario] = {'mode_accuracy': sum(row['prediction'] == mode for row in selected) / 500,
                              'predicted_modes': [sum(row['prediction'] == value for row in selected) for value in range(3)]}
    matrix = [[sum(row['mode'] == truth and row['prediction'] == prediction for row in records)
               for prediction in range(3)] for truth in range(3)]
    metrics = {target: summarize(records, target) for target in ('fault', 'leak')}
    output = directory / 'output'
    output.mkdir()
    certificate = dict(config=config, observation_git_blob=git_blob,
                       observation_sha256=hashlib.sha256(archive_bytes).hexdigest(),
                       clean_cold_temperature=reference, flow_threshold=flow_threshold,
                       temperature_threshold=temperature_threshold, records=records,
                       scenarios=scenarios, mode_confusion=matrix, metrics=metrics,
                       duplicates=len(hashes) - len(set(hashes)),
                       query_seconds=dict(median=statistics.median(durations),
                                          p95=sorted(durations)[math.ceil(.95 * len(durations)) - 1],
                                          maximum=max(durations)),
                       query_durations_seconds=durations,
                       removal_cap_rejected={target: metrics[target]['balanced_accuracy'] > config['balanced_accuracy_cap']
                                             and max(durations) <= config['reader_budget_seconds']
                                             for target in metrics})
    (output / 'certificate.json').write_text(json.dumps(certificate, indent=2) + '\n')
    (output / 'machine.json').write_text(json.dumps(dict(platform=platform.platform(), python=sys.version,
                                                        numpy=np.__version__), indent=2) + '\n')
    print(json.dumps({key: certificate[key] for key in ('metrics', 'mode_confusion', 'duplicates', 'query_seconds', 'removal_cap_rejected')}))


if __name__ == '__main__':
    main()
