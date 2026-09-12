"""Development follow-up: retain the identity reader as an endpoint option.

Bitwise-optimal sign calibration need not improve byte/compound task scores.
Selection reads calibration records only. Evaluation is already open development
data, so this is an explicitly adaptive follow-up, not a new sealed test.
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

from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.recurrent_state import (RecurrentMemory, decode_initial_cold, edited, load_state,
    metrics, permission_metrics, predictions, reference_state, rollout, state_information)
from audit_recurrent_state import (check_metrics, gate_from_checkpoint, read_tasks,
                                  run_updates, task_oracle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    parent_root, root = Path(args.parent_run), Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = snapshot_sources(root/'source')
    for file in (Path(__file__), Path('scripts/audit_recurrent_state.py'), Path('protocols/SCC_RECURRENT_STATE_V1.md')):
        target = root/'source'/file.name
        shutil.copyfile(file, target)
        source[file.name] = file_digest(target)
    enumeration = json.loads((parent_root/'enumeration.json').read_text())
    run = json.loads((parent_root/'result.json').read_text())
    assert run['status'] == 'completed'
    candidates = []
    for row in enumeration['rows']:
        if row['eligible']:
            for index, field in enumerate(('raw_task_accuracy', 'fitted_sign_task_accuracy')):
                flat = [v for scores in row[field].values() for v in scores]
                key = (min(flat), float(np.mean(flat)), row['retained_coordinate_fraction'],
                       -row['gate_index'], -row['table_number'], -index)
                candidates.append((key, row, field))
    key, selected, reader = max(candidates, key=lambda item: item[0])
    record = {'source': source, 'parent_run': str(parent_root),
        'parent_enumeration_sha256': file_digest(parent_root/'enumeration.json'),
        'evidence_class': 'Adaptive open-development decoder-choice follow-up',
        'selection_uses_evaluation_targets': False,
        'reason': 'Hamming-optimal sign fitting can damage joint tasks; include unchanged reader',
        'eligible_cell_reader_combinations': len(candidates), 'selected_gate': selected['gate_case'],
        'selected_cell': selected['table'], 'selected_table_number': selected['table_number'],
        'selected_reader': reader, 'calibration_minimum_task_accuracy': key[0],
        'calibration_mean_task_accuracy': key[1], 'mechanism_demonstrated': False,
        'new_cloud_compute_usd': 0, 'arms': {}}
    atomic_json(root/'contract.json', record)
    counters = {'raw_answers_rescored': 0, 'full_numpy_state_replays': 0, 'checkpoint_hashes_verified': 0}
    for seed in run['arms']:
        training = json.loads((parent_root/f'seed-{seed}/training.json').read_text())
        parent_path = parent_root/f'seed-{seed}/parent.pt'
        assert file_digest(parent_path) == training['checkpoint_sha256']
        parent = RecurrentMemory(int(seed))
        parent.load_state_dict(torch.load(parent_path, weights_only=True)['model'])
        model = edited(parent, selected['gate_case'])
        model.set_cell(selected['table'])
        if reader == 'fitted_sign_task_accuracy':
            flips = np.load(parent_root/'enumeration-decoders.npz')[selected['gate_case']+'_flips'][selected['table_number']]
            with torch.no_grad():
                model.read_signs.copy_(torch.as_tensor(np.where(flips, -8., 8.)))
        gate, table, flip, permission = model.discrete()
        path = root/f'seed-{seed}.pt'
        torch.save({'model': model.state_dict()}, path)
        state = {k: v.numpy() for k, v in torch.load(path, weights_only=True)['model'].items()}
        assert np.array_equal(gate_from_checkpoint(state), gate)
        counters['checkpoint_hashes_verified'] += 1
        arm = {'checkpoint_sha256': file_digest(path), 'parent_checkpoint_sha256': training['checkpoint_sha256'],
               'policy': permission_metrics(gate, permission), 'domains': {}}
        accepted = (state['cell_logits'] > 0)[4*gate+1]
        assert accepted.tolist() == arm['policy']['accepted'] and accepted.all()
        arrays = {}
        for domain in ('iid', 'balanced'):
            e = dict(np.load(parent_root/f'evaluation-{domain}.npz'))
            arm.setdefault('input_sha256', {})[domain] = file_digest(parent_root/f'evaluation-{domain}.npz')
            initial = load_state(e, *parent.discrete()[:2])
            final, _ = rollout(initial, e, gate, table)
            low, high, curve = state_information(initial, e, gate, table)
            actual, target = predictions(final, e, flip), predictions(reference_state(e), e)
            probe = decode_initial_cold(final, low, high, e)
            domain_result = {'tasks': metrics(actual, target), 'retained_coordinate_fraction': curve['24'],
                             'history_assisted_lookup': float(np.mean(probe == target[:, 0]))}
            arm['domains'][domain] = domain_result
            arrays.update({domain+'_state': final, domain+'_predictions': actual, domain+'_targets': target,
                           domain+'_low': low, domain+'_high': high, domain+'_history_decoded': probe})
            independent, _ = run_updates(initial, e, gate, table)
            assert np.array_equal(final, independent)
            assert np.array_equal(actual, read_tasks(independent, e, flip))
            oracle, _ = task_oracle(e)
            assert np.array_equal(oracle, target)
            check_metrics(domain_result['tasks'], actual, oracle, counters)
            independent_low, _ = run_updates(np.zeros_like(initial), e, gate, table)
            independent_high, _ = run_updates(np.ones_like(initial), e, gate, table)
            assert np.array_equal(low, independent_low) and np.array_equal(high, independent_high)
            assert np.take_along_axis(low != high, e['cold'][..., None], 1).mean() == curve['24']
            counters['full_numpy_state_replays'] += len(initial)*3
        np.savez_compressed(root/f'seed-{seed}-states.npz', **arrays)
        arm['states_sha256'] = file_digest(root/f'seed-{seed}-states.npz')
        atomic_json(root/f'seed-{seed}.json', arm)
        record['arms'][seed] = arm
    for path, expected in source.items():
        assert file_digest(root/'source'/path) == expected
    counters['source_files_verified'] = len(source)
    record.update(status='completed_and_numpy_verified', verification=counters)
    atomic_json(root/'result.json', record)
    print(json.dumps({'status': record['status'], 'selection': [selected['gate_case'], selected['table_number'], reader],
                     'calibration_minimum': key[0], 'verification': counters,
                     'task_exact': {d: {k: v['exact'] for k, v in x['tasks'].items()} for d, x in record['arms'][seed]['domains'].items()}}))


if __name__ == '__main__':
    main()
