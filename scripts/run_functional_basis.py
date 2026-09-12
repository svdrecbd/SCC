"""Execute SCC functional-basis synthesis and exhaustive finite task evaluation."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scc.functional_basis import (TASKS, LearnedGate, build_circuit, closure, edit_gate,
    execute, fit_gate, fit_projection, input_grid, permission, replace_nand, score)
from scc.provenance import atomic_json, file_digest, snapshot_sources

CASES = ('clean', 'positive_scale', 'joint_sign_compensation', 'implication_original',
         'implication_recompiled', 'constant_allow_original', 'constant_allow_projection',
         'separate_basis_control')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 29, 47])
    p.add_argument('--engineering', action='store_true')
    args = p.parse_args()
    if not args.engineering and (args.width != 8 or args.seeds != [11, 29, 47]):
        raise ValueError('Nonprotocol configuration requires engineering label')
    if not 1 <= args.width <= 8:
        raise ValueError('Unsupported finite population size')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = snapshot_sources(root/'source')
    for name in ('scripts/run_functional_basis.py', 'scripts/audit_functional_basis.py',
                 'tests/test_functional_basis.py', 'protocols/SCC_FUNCTIONAL_BASIS_V1.md',
                 'WORKING_STANDARDS.md', 'MECHANISM_TARGET.md', 'AGENTS.md'):
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        source[name] = file_digest(target)
    atomic_json(root/'contract.json', {'configuration': vars(args), 'source': source,
        'torch': str(torch.__version__), 'numpy': np.__version__, 'device': 'cpu',
        'evidence_class': 'engineering validation' if args.engineering else 'finite constructive development evidence',
        'new_cloud_compute_usd': 0, 'sealed_test_used': False,
        'calibration_is_subset_of_exhaustive_evaluation': True, 'mechanism_demonstrated': False})
    started = time.monotonic()
    proofs, classification = {}, []
    for number in range(16):
        proofs[number] = {arity: closure(number, arity) for arity in (2, 3)}
        row = {'number': number, 'policy': permission(proofs[number][2]['table']),
               'arity_two_functions': len(proofs[number][2]['reachable']),
               'arity_three_functions': len(proofs[number][3]['reachable']),
               'arity_three_terminal_inversion_functions': len(proofs[number][3]['terminal_inversion_reachable']),
               'nand_synthesizable': 7 in proofs[number][2]['reachable']}
        classification.append(row)
        atomic_json(root/f'closure-{number}.json', {str(k): v for k, v in proofs[number].items()})
    atomic_json(root/'classification.json', classification)
    original, circuits, grids, manifest = {}, {}, {}, {}
    for task in TASKS:
        original[task] = build_circuit(task, args.width)
        circuits[task] = {'original': original[task],
            'implication_recompiled': replace_nand(original[task], proofs[11][2]),
            'separate_basis_control': replace_nand(original[task], proofs[7][2], primitive=1)}
        calibration_inputs, calibration_targets = input_grid(task, args.width, calibration=True)
        projection, scores = fit_projection(calibration_inputs, calibration_targets)
        projection['capacity'] = original[task]['capacity']
        circuits[task]['projection'] = projection
        inputs, targets = input_grid(task, args.width)
        grids[task] = (inputs, targets)
        path = root/f'{task}-data.npz'
        np.savez_compressed(path, inputs=inputs, targets=targets,
                            calibration_inputs=calibration_inputs, calibration_targets=calibration_targets)
        manifest[task] = {'data_sha256': file_digest(path), 'evaluation_size': len(inputs),
                          'calibration_size': len(calibration_inputs), 'calibration_projection_bit_accuracy': scores}
    atomic_json(root/'data-manifest.json', manifest)
    atomic_json(root/'circuits.json', circuits)
    result = {'arms': {}, 'classification': classification, 'status': 'running', 'mechanism_demonstrated': False}
    for seed in args.seeds:
        directory = root/f'seed-{seed}'
        directory.mkdir()
        parent = LearnedGate(seed)
        training, optimizer = fit_gate(parent)
        assert training['qualified'] and parent.table().tolist() == [1, 1, 1, 0]
        torch.save({'model': parent.state_dict(), 'optimizer': optimizer}, directory/'parent.pt')
        training['checkpoint_sha256'] = file_digest(directory/'parent.pt')
        atomic_json(directory/'training.json', training)
        # The control includes its reserve gate before the primary edit.
        torch.save({'primary': parent.state_dict(), 'reserve': copy.deepcopy(parent.state_dict())}, directory/'control-parent.pt')
        saved = copy.deepcopy(parent.state_dict())
        arm = {'parent_qualified': False, 'control_parent_sha256': file_digest(directory/'control-parent.pt'), 'cases': {}}
        for case in CASES:
            edit = ('implication' if case.startswith('implication') else
                    'constant_allow' if case.startswith('constant') or case == 'separate_basis_control' else case)
            model = edit_gate(parent, edit)
            tables = [model.table()]
            if case == 'separate_basis_control':
                tables.append(parent.table())
            checkpoint = directory/f'{case}.pt'
            torch.save({'primary': model.state_dict(), 'reserve': parent.state_dict() if len(tables) == 2 else None}, checkpoint)
            changes = {k: int(torch.count_nonzero(v-saved[k])) for k, v in model.state_dict().items()}
            record = {'checkpoint_sha256': file_digest(checkpoint), 'policy': permission(model.table()),
                'changed_scalar_parameters': sum(changes.values()), 'changes_by_parameter': changes,
                'tasks': {}, 'circuit_modes': {}, 'node_counts': {}}
            predictions = {}
            for task in TASKS:
                mode = ('implication_recompiled' if case == 'implication_recompiled' else
                        'projection' if case == 'constant_allow_projection' else
                        'separate_basis_control' if case == 'separate_basis_control' else 'original')
                program = circuits[task][mode]
                inputs, targets = grids[task]
                actual = execute(program, inputs, tables)
                predictions[task] = actual
                record['tasks'][task] = score(actual, targets)
                record['circuit_modes'][task] = mode
                record['node_counts'][task] = len(program['nodes'])
                if case in ('clean', 'positive_scale', 'joint_sign_compensation', 'implication_recompiled', 'separate_basis_control'):
                    assert np.array_equal(actual, targets), (case, task)
            if case == 'clean':
                arm['parent_qualified'] = all(v['exact'] == 1 for v in record['tasks'].values())
                assert arm['parent_qualified']
            if case.startswith('implication'):
                assert record['changed_scalar_parameters'] == 2
                assert record['policy']['removal_by_contract']['skewed']
            path = directory/f'{case}-predictions.npz'
            np.savez_compressed(path, **predictions)
            record['predictions_sha256'] = file_digest(path)
            atomic_json(directory/f'{case}.json', record)
            arm['cases'][case] = {'policy': record['policy'], 'task_exact': {k: v['exact'] for k, v in record['tasks'].items()},
                                   'changed_scalar_parameters': record['changed_scalar_parameters'], 'node_counts': record['node_counts']}
            print(json.dumps({'seed': seed, 'case': case, **arm['cases'][case]}), flush=True)
        for k, v in parent.state_dict().items():
            assert torch.equal(v, saved[k]), 'Parent changed'
        arm['status'] = 'completed'
        result['arms'][str(seed)] = arm
        atomic_json(root/'progress.json', result)
    result.update(status='completed', elapsed_seconds=time.monotonic()-started, new_cloud_compute_usd=0)
    result['fixed_artifact_sha256'] = {name: file_digest(root/name) for name in
        ['classification.json', 'circuits.json', 'data-manifest.json']+[f'closure-{i}.json' for i in range(16)]}
    atomic_json(root/'result.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
