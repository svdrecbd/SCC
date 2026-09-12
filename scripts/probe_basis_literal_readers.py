"""Compare simple word-level readers with the bitwise-fitted projection.

This adaptive diagnostic uses calibration to choose among five specified readers,
then evaluates the complete finite population. It is not optimal circuit search.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scc.functional_basis import TASKS, execute, score
from scc.provenance import atomic_json, file_digest, snapshot_sources
from audit_functional_basis import evaluate_circuit, oracle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    parent_root, root = Path(args.parent_run), Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    source = snapshot_sources(root/'source')
    for path in (Path(__file__), Path('scripts/audit_functional_basis.py')):
        target = root/'source'/path.name
        shutil.copyfile(path, target)
        source[path.name] = file_digest(target)
    contract = json.loads((parent_root/'contract.json').read_text())
    width = contract['configuration']['width']
    run = json.loads((parent_root/'result.json').read_text())
    assert run['status'] == 'completed'
    result = {'source': source, 'parent_run': str(parent_root),
        'parent_result_sha256': file_digest(parent_root/'result.json'),
        'scope': 'Adaptive five-reader comparison; selected on calibration only; exhaustive population includes calibration',
        'mechanism_demonstrated': False, 'new_cloud_compute_usd': 0, 'tasks': {}, 'reused_checkpoint_sha256': {}}
    for seed in run['arms']:
        path = parent_root/f'seed-{seed}/constant_allow_projection.pt'
        saved = torch.load(path, weights_only=True)['primary']
        assert ((saved['gain']*saved['logits']) > 0).all()
        expected = json.loads((parent_root/f'seed-{seed}/constant_allow_projection.json').read_text())['checkpoint_sha256']
        assert file_digest(path) == expected
        result['reused_checkpoint_sha256'][seed] = expected
    circuits = json.loads((parent_root/'circuits.json').read_text())
    for task in TASKS:
        data_path = parent_root/f'{task}-data.npz'
        data = dict(np.load(data_path))
        base = circuits[task]['projection']
        candidates = {}
        for name, outputs in (('copy_left', list(range(width))), ('copy_right', list(range(width, 2*width))),
                              ('constant_zero', [base['inputs']]*width), ('constant_one', [base['inputs']+1]*width)):
            program = copy.deepcopy(base)
            program.update(outputs=outputs, output_flip=[False]*width)
            candidates[name] = program
        candidates['bitwise_fitted'] = base
        calibration, results = {}, {}
        for name, program in candidates.items():
            actual = execute(program, data['calibration_inputs'], [np.ones(4, dtype=np.uint8)])
            calibration[name] = score(actual, data['calibration_targets'])
        selected = max(candidates, key=lambda name: calibration[name]['exact'])
        actual = execute(candidates[selected], data['inputs'], [np.ones(4, dtype=np.uint8)])
        independent = evaluate_circuit(candidates[selected], data['inputs'], [[1, 1, 1, 1]])
        expected = oracle(data['inputs'], task, width)
        assert np.array_equal(actual, independent) and np.array_equal(expected, data['targets'])
        np.savez_compressed(root/f'{task}-predictions.npz', predictions=actual)
        result['tasks'][task] = {'selected_reader': selected, 'calibration': calibration,
            'evaluation': score(actual, expected), 'program': candidates[selected],
            'input_sha256': file_digest(data_path), 'predictions_sha256': file_digest(root/f'{task}-predictions.npz')}
    for path, expected in source.items(): assert file_digest(root/'source'/path) == expected
    result.update(status='completed_and_numpy_verified', full_population_predictions_verified=sum(x['evaluation']['n'] for x in result['tasks'].values()))
    atomic_json(root/'result.json', result)
    print(json.dumps({'status': result['status'], 'tasks': {k: {'reader': v['selected_reader'], 'exact': v['evaluation']['exact']} for k, v in result['tasks'].items()}}))


if __name__ == '__main__': main()
