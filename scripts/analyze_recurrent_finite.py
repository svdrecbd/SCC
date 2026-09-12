"""Exact finite-cell classification and a minimal two-parameter counterexample."""

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
from scc.recurrent_state import RecurrentMemory, load_state, predictions, rollout
from audit_recurrent_state import gate_from_checkpoint, read_tasks, run_updates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    parent_root, root = Path(args.parent_run), Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = snapshot_sources(root/'source')
    for file in (Path(__file__), Path('scripts/audit_recurrent_state.py')):
        target = root/'source'/file.name
        shutil.copyfile(file, target)
        source[file.name] = file_digest(target)
    enumeration = json.loads((parent_root/'enumeration.json').read_text())
    run = json.loads((parent_root/'result.json').read_text())
    assert run['status'] == 'completed'
    classes = {}
    for row in enumeration['rows']:
        # A cold address differs from every active write address. Thus the
        # actual gate is fixed per gate family, independently of sampled data.
        g = 0 if row['gate_case'] in ('clean', 'constant_deny_predicate') else 1
        table = row['table']
        bijective_for_every_new_bit = all(table[4*g+n] != table[4*g+2+n] for n in (0, 1))
        assert bijective_for_every_new_bit == (row['retained_coordinate_fraction'] == 1)
        group = classes.setdefault(row['gate_case'], {'total': 0, 'eligible': 0, 'eligible_universally_bijective_on_cold_bits': 0})
        group['total'] += 1
        group['eligible'] += int(row['eligible'])
        group['eligible_universally_bijective_on_cold_bits'] += int(row['eligible'] and bijective_for_every_new_bit)
    result = {'source': source, 'enumeration_sha256': file_digest(parent_root/'enumeration.json'),
        'classification': classes, 'classification_is_exhaustive_over_specified_1024_cells': True,
        'proof': 'For a fixed gate and incoming bit, a Boolean old-state map is bijective iff its outputs for old=0 and old=1 differ. Composition preserves this property. Cold addresses are never written intentionally.',
        'scope': 'Coordinate dependence conditional on suffix; not entropy or general cognition',
        'minimal_counterexample': {}, 'mechanism_demonstrated': False, 'new_cloud_compute_usd': 0}
    for seed in run['arms']:
        parent_path = parent_root/f'seed-{seed}/parent.pt'
        expected_parent = json.loads((parent_root/f'seed-{seed}/training.json').read_text())['checkpoint_sha256']
        assert file_digest(parent_path) == expected_parent
        parent = RecurrentMemory(int(seed))
        parent.load_state_dict(torch.load(parent_path, weights_only=True)['model'])
        model = copy.deepcopy(parent)
        with torch.no_grad():
            model.cell_logits[1] *= -1
            model.cell_logits[3] *= -1
        changes = {key: int(torch.count_nonzero(value-parent.state_dict()[key]))
                   for key, value in model.state_dict().items()}
        assert sum(changes.values()) == 2 and changes['cell_logits'] == 2
        path = root/f'seed-{seed}-two-parameter-edit.pt'
        torch.save({'model': model.state_dict(), 'changed_parameters': changes}, path)
        record = {'checkpoint_sha256': file_digest(path), 'parent_checkpoint_sha256': expected_parent,
                  'changed_parameters': {k: v for k, v in changes.items() if v}, 'domains': {}}
        gate, table, flip, permission = model.discrete()
        assert permission[gate*4+1].all()
        state = {k: v.numpy() for k, v in torch.load(path, weights_only=True)['model'].items()}
        assert np.array_equal(gate_from_checkpoint(state), gate)
        for domain in ('iid', 'balanced'):
            e = dict(np.load(parent_root/f'evaluation-{domain}.npz'))
            initial = load_state(e, *parent.discrete()[:2])
            final, _ = rollout(initial, e, gate, table)
            actual = predictions(final, e, flip)
            reference = np.load(parent_root/f'seed-{seed}/reversible_allow_cell-states.npz')
            assert np.array_equal(final, reference[domain+'_final'])
            assert np.array_equal(actual, reference[domain+'_predictions'])
            independent, _ = run_updates(initial, e, gate, table)
            assert np.array_equal(independent, final)
            assert np.array_equal(read_tasks(independent, e, flip), actual)
            record['domains'][domain] = {'episodes': len(initial),
                'identical_to_original_reversible_condition': True,
                'reference_states_sha256': file_digest(parent_root/f'seed-{seed}/reversible_allow_cell-states.npz'),
                'input_sha256': file_digest(parent_root/f'evaluation-{domain}.npz')}
        result['minimal_counterexample'][seed] = record
    for path, expected in source.items():
        assert file_digest(root/'source'/path) == expected
    result['status'] = 'completed_and_verified'
    atomic_json(root/'result.json', result)
    print(json.dumps({'status': result['status'], 'classification': classes,
                      'changed_scalar_parameters': 2, 'verified_parent_checkpoints': len(run['arms'])}))


if __name__ == '__main__':
    main()
