"""Run the predeclared SCC finite recurrent-state experiment."""

import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.recurrent_state import (KEYS, WIDTH, TASKS, RecurrentMemory, bits, decode_initial_cold,
    edited, episode_hashes, generate_episodes, load_state, metrics,
    numerical_control, permission_metrics, predictions, reference_state, rollout,
    state_information, train_cell)

GATE_CASES = ('clean', 'inverted_predicate', 'constant_allow_predicate', 'constant_deny_predicate')
CASES = ('clean', 'positive_scale', 'joint_sign_compensation', 'constant_allow_cell',
         'constant_allow_predicate', 'inverted_predicate', 'reversible_allow_cell',
         'separate_permission_control', 'selected_joint_edit')
RECOVER = ('constant_allow_cell', 'constant_allow_predicate', 'inverted_predicate',
           'reversible_allow_cell', 'selected_joint_edit')


def enumerate_cells(parent, calibration, destination):
    tables = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
    clean_gate, clean_cell, _, _ = parent.discrete()
    initial = [load_state(e, clean_gate, clean_cell) for e in calibration.values()]
    targets = [reference_state(e) for e in calibration.values()]
    rows, best = [], None
    saved = {}
    for gindex, gate_case in enumerate(GATE_CASES):
        gate = edited(parent, gate_case).discrete()[0]
        finals, retentions = [], []
        for e, first in zip(calibration.values(), initial):
            expanded = np.broadcast_to(first, (256, *first.shape)).copy()
            final, _ = rollout(expanded, e, gate, tables, checkpoints=())
            low, _ = rollout(np.zeros_like(expanded), e, gate, tables, checkpoints=())
            high, _ = rollout(np.ones_like(expanded), e, gate, tables, checkpoints=())
            cold = e['cold'][None, :, :, None]
            keep = np.take_along_axis(low != high, cold, axis=2).mean(axis=(1, 2, 3))
            finals.append(final)
            retentions.append(keep)
        matches = sum((state == target[None]).sum(1) for state, target in zip(finals, targets))
        flips = matches < sum(len(s) for s in initial)/2
        saved[gate_case + '_flips'] = flips
        for number, table in enumerate(tables):
            policy = permission_metrics(gate, table)
            raw_scores, decoded_scores = {}, {}
            for domain, e, state, target in zip(calibration, calibration.values(), finals, targets):
                expected = predictions(target, e)
                raw = predictions(state[number], e)
                decoded = predictions(state[number], e, flips[number])
                raw_scores[domain] = (raw == expected).mean(0).tolist()
                decoded_scores[domain] = (decoded == expected).mean(0).tolist()
            flat = [value for domain in decoded_scores.values() for value in domain]
            retention = float(np.mean([r[number] for r in retentions]))
            eligible = policy['true_acceptance'] >= .95 and policy['false_acceptance'] >= .95
            row = {'gate_case': gate_case, 'gate_index': gindex, 'table_number': number,
                'table': table.tolist(), 'true_acceptance': policy['true_acceptance'],
                'false_acceptance': policy['false_acceptance'], 'eligible': eligible,
                'raw_task_accuracy': raw_scores, 'fitted_sign_task_accuracy': decoded_scores,
                'minimum_task_accuracy': min(flat), 'mean_task_accuracy': float(np.mean(flat)),
                'retained_coordinate_fraction': retention}
            rows.append(row)
            key = (min(flat), float(np.mean(flat)), retention, -gindex, -number)
            if eligible and (best is None or key > best[0]):
                best = (key, row, flips[number].copy())
        print(json.dumps({'enumeration_gate': gate_case, 'endpoints': 256}), flush=True)
    selected = copy.deepcopy(best[1])
    selected['read_flip'] = best[2].tolist()
    result = {'status': 'completed', 'candidate_count': len(rows),
              'eligible_count': sum(r['eligible'] for r in rows), 'tasks_in_order': TASKS,
              'scope': 'All 256 Boolean cells under four specified predicate edits; sign fitting optimizes bits only',
              'selected': selected, 'rows': rows}
    np.savez_compressed(destination/'enumeration-decoders.npz', **saved)
    result['decoder_file_sha256'] = file_digest(destination/'enumeration-decoders.npz')
    atomic_json(destination/'enumeration.json', result)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parents', default='artifacts/scc-shared-predicate-20260910-v1')
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 29, 47])
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--calibration-size', type=int, default=512)
    parser.add_argument('--engineering', action='store_true')
    args = parser.parse_args()
    if not args.engineering and (args.size != 2048 or args.calibration_size != 512 or args.seeds != [11, 29, 47]):
        raise ValueError('Non-protocol settings require --engineering')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = snapshot_sources(root/'source')
    for name in ('scripts/run_recurrent_state.py', 'scripts/audit_recurrent_state.py',
                 'tests/test_recurrent_state.py', 'protocols/SCC_RECURRENT_STATE_V1.md',
                 'WORKING_STANDARDS.md', 'MECHANISM_TARGET.md', 'AGENTS.md'):
        path = Path(name)
        target = root/'source'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source[name] = file_digest(target)
    atomic_json(root/'contract.json', {'configuration': vars(args), 'source': source,
        'torch': str(torch.__version__), 'numpy': np.__version__, 'python': platform.python_version(),
        'device': 'cpu', 'evidence_class': 'engineering' if args.engineering else 'open development construction',
        'sealed_test_used': False, 'new_cloud_compute_usd': 0, 'mechanism_demonstrated': False})
    data, hashes, all_hashes = {}, {}, set()
    for split, base, size in (('calibration', 920100, args.calibration_size),
                              ('evaluation', 921100, args.size), ('fresh', 922100, args.size)):
        data[split], hashes[split] = {}, {}
        for offset, domain in enumerate(('iid', 'balanced')):
            episodes = generate_episodes(size, base+offset, domain)
            identity = episode_hashes(episodes)
            assert len(set(identity)) == size and not (set(identity) & all_hashes)
            all_hashes.update(identity)
            path = root/f'{split}-{domain}.npz'
            np.savez_compressed(path, **episodes)
            hashes[split][domain] = {'episode_hashes': identity, 'file_sha256': file_digest(path)}
            data[split][domain] = episodes
    atomic_json(root/'data-manifest.json', hashes)
    atomic_json(root/'numerical-control.json', numerical_control())
    started = time.monotonic()
    parents, result = {}, {'arms': {}, 'status': 'running', 'mechanism_demonstrated': False}
    for seed in args.seeds:
        directory = root/f'seed-{seed}'
        directory.mkdir()
        checkpoint = Path(args.parents)/f'seed-{seed}/parent/checkpoint.pt'
        expected = json.loads((checkpoint.parent/'training.json').read_text())['checkpoint_sha256']
        assert file_digest(checkpoint) == expected
        parent = RecurrentMemory(seed)
        parent.producer.load_state_dict(torch.load(checkpoint, weights_only=True)['model'])
        training, optimizer = train_cell(parent)
        torch.save({'model': parent.state_dict(), 'optimizer': optimizer, 'steps': training['steps']}, directory/'parent.pt')
        training.update(inherited_predicate=str(checkpoint), inherited_sha256=expected,
                        checkpoint_sha256=file_digest(directory/'parent.pt'),
                        parameters=sum(p.numel() for p in parent.parameters()))
        atomic_json(directory/'training.json', training)
        assert training['qualified']
        parents[seed] = parent
    selected = enumerate_cells(parents[args.seeds[0]], data['calibration'], root)
    for seed, parent in parents.items():
        directory = root/f'seed-{seed}'
        clean_gate, clean_cell, _, _ = parent.discrete()
        assert np.array_equal(clean_gate, np.eye(KEYS, dtype=np.uint8))
        initial = {domain: load_state(e, clean_gate, clean_cell) for domain, e in data['evaluation'].items()}
        for domain, e in data['evaluation'].items():
            assert np.array_equal(initial[domain], bits(e['initial_values'], e['initial_pointers']))
        parent_saved = copy.deepcopy(parent.state_dict())
        arm = {'parent_qualified': False, 'cases': {}}
        for case in CASES:
            if case == 'selected_joint_edit':
                model = edited(parent, selected['gate_case'])
                model.set_cell(selected['table'])
                with torch.no_grad():
                    model.read_signs.copy_(torch.as_tensor(np.where(selected['read_flip'], -8., 8.)))
            else:
                model = edited(parent, case)
            gate, cell, flip, permission_cell = model.discrete()
            checkpoint = directory/f'{case}.pt'
            torch.save({'model': model.state_dict(), 'separate_permission': model.permission_cell is not None}, checkpoint)
            record = {'checkpoint_sha256': file_digest(checkpoint), 'policy': permission_metrics(gate, permission_cell),
                      'gate': gate.tolist(), 'cell': cell.tolist(), 'read_flip': flip.tolist(), 'domains': {}}
            arrays = {}
            for domain, e in data['evaluation'].items():
                final, trajectory = rollout(initial[domain], e, gate, cell)
                low, high, retention = state_information(initial[domain], e, gate, cell)
                expected = predictions(reference_state(e), e)
                actual = predictions(final, e, flip)
                decoded = decode_initial_cold(final, low, high, e)
                cold_target = expected[:, 0]
                domain_result = {'tasks': metrics(actual, expected), 'retention_by_write_count': retention,
                    'history_assisted_cold_lookup_exact': float((decoded == cold_target).mean()),
                    'history_assisted_cold_lookup_bit_accuracy': ((((decoded ^ cold_target)[:, None] >> np.arange(8)) & 1) == 0).mean(0).tolist(),
                    'history_probe_interface': 'final state plus write suffix; queried address never appears in suffix'}
                arrays.update({f'{domain}_final': final, f'{domain}_low': low, f'{domain}_high': high,
                    f'{domain}_predictions': actual, f'{domain}_targets': expected, f'{domain}_history_decoded': decoded})
                arrays.update({f'{domain}_state_{t}': state for t, state in trajectory.items()})
                if case in RECOVER:
                    repaired = predictions(final, e)  # Parent signs; same damaged state, no input replay.
                    fresh = data['fresh'][domain]
                    fresh_state = load_state(fresh, clean_gate, clean_cell)
                    fresh_final, _ = rollout(fresh_state, fresh, clean_gate, clean_cell)
                    fresh_actual = predictions(fresh_final, fresh)
                    fresh_target = predictions(reference_state(fresh), fresh)
                    # Loading all 16 addresses through the repaired parent overwrites
                    # arbitrary damage: check starting from the actual damaged state.
                    from_damage = final.copy()
                    fresh_payload = bits(fresh['initial_values'], fresh['initial_pointers'])
                    from scc.recurrent_state import step_state
                    b = np.arange(len(from_damage))
                    for addresses in fresh['load_order'].T:
                        from_damage = step_state(from_damage, clean_gate, clean_cell, addresses, fresh_payload[b, addresses])
                    assert np.array_equal(from_damage, fresh_state)
                    domain_result['parent_weight_repair'] = metrics(repaired, expected)
                    domain_result['fresh_input_after_repair'] = metrics(fresh_actual, fresh_target)
                    arrays.update({f'{domain}_repair_predictions': repaired, f'{domain}_fresh_predictions': fresh_actual,
                                   f'{domain}_fresh_targets': fresh_target})
                if case == 'constant_allow_cell':
                    ordinary_state = np.ones_like(final)
                    assert np.array_equal(final, ordinary_state)
                    assert np.array_equal(predictions(ordinary_state, e), actual)
                    domain_result['ordinary_overwrite_control_identical'] = True
                record['domains'][domain] = domain_result
            policy = record['policy']
            record['qualified'] = (policy['true_acceptance'] >= .95 and policy['false_acceptance'] <= .05
                and min(m['exact'] for d in record['domains'].values() for m in d['tasks'].values()) >= .95)
            if case == 'clean':
                arm['parent_qualified'] = record['qualified']
                assert arm['parent_qualified'], 'Unqualified parent'
            if case in ('positive_scale', 'joint_sign_compensation'):
                assert record['qualified']
            path = directory/f'{case}-states.npz'
            np.savez_compressed(path, **arrays)
            record['states_sha256'] = file_digest(path)
            atomic_json(directory/f'{case}.json', record)
            summary = {'true_acceptance': policy['true_acceptance'], 'false_acceptance': policy['false_acceptance'],
                'task_exact': {domain: {task: m['exact'] for task, m in d['tasks'].items()} for domain, d in record['domains'].items()},
                'retained_coordinate_fraction': {domain: d['retention_by_write_count']['24'] for domain, d in record['domains'].items()},
                'history_assisted_lookup': {domain: d['history_assisted_cold_lookup_exact'] for domain, d in record['domains'].items()}}
            arm['cases'][case] = summary
            print(json.dumps({'seed': seed, 'case': case, **summary}), flush=True)
        for key, value in parent.state_dict().items():
            assert torch.equal(value, parent_saved[key]), 'Parent mutated'
        arm['status'] = 'completed'
        result['arms'][str(seed)] = arm
        atomic_json(root/'progress.json', result)
    result.update(status='completed', elapsed_seconds=time.monotonic()-started,
                  enumeration_selected=selected, new_cloud_compute_usd=0)
    atomic_json(root/'result.json', result)
    print(json.dumps({'status': 'completed', 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
