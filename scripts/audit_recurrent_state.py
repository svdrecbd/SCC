"""Independent NumPy/scalar audit of saved recurrent-state experiments.

Does not import the implementation or its task/scoring/selection functions.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

TASKS = ('retained_lookup', 'latest_lookup', 'composition', 'addition', 'pairwise_xor')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def episode_digest(episode):
    text = json.dumps(episode, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def gate_from_checkpoint(state):
    onehot = np.eye(16)
    left, right = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
    x = np.concatenate((onehot[left], onehot[right]), -1)
    for layer in (0, 2):
        x = np.tanh(x @ state[f'producer.hidden.{layer}.weight'].T + state[f'producer.hidden.{layer}.bias'])
    scores = (x @ state['producer.output.weight'].T + state['producer.output.bias'])[..., 0]
    return (scores > 0).astype(np.uint8)


def payload(values, pointers):
    return np.stack([((values if bit < 8 else pointers) >> (bit if bit < 8 else bit-8)) & 1
                     for bit in range(12)], -1).astype(np.uint8)


def run_updates(initial, episodes, gate, cell, load=False):
    state = initial.copy()
    path = {0: state.copy()}
    if load:
        addresses = episodes['load_order']
        data = payload(episodes['initial_values'], episodes['initial_pointers'])
    else:
        addresses = episodes['addresses']
        data = payload(episodes['new_values'], episodes['new_pointers'])
    for t in range(addresses.shape[1]):
        address = addresses[:, t]
        incoming = data[np.arange(len(address)), address] if load else data[:, t]
        base = 4 * gate[address, :, None] + incoming[:, None, :]
        if cell.ndim == 1:
            updated = np.where(state == 0, cell[base], cell[base+2])
        else:
            before_zero = cell[np.arange(len(cell))[:, None, None, None], base[None]]
            before_one = cell[np.arange(len(cell))[:, None, None, None], base[None]+2]
            updated = np.where(state == 0, before_zero, before_one)
        state = updated if load else np.where((episodes['lengths'] > t)[:, None, None], updated, state)
        if t+1 in (1, 4, 8, 16, 24):
            path[t+1] = state.copy()
    return state, path


def task_oracle(episodes):
    answers, states = [], []
    for b in range(len(episodes['cold'])):
        memory = {i: int(v) for i, v in enumerate(episodes['initial_values'][b])}
        pointers = {i: int(v) for i, v in enumerate(episodes['initial_pointers'][b])}
        length = int(episodes['lengths'][b])
        for address, value, pointer in zip(episodes['addresses'][b, :length], episodes['new_values'][b, :length], episodes['new_pointers'][b, :length]):
            memory[int(address)] = int(value)
            pointers[int(address)] = int(pointer)
        query, other = map(int, episodes['cold'][b, :2])
        answers.append([memory[query], memory[int(episodes['addresses'][b, length-1])],
                        memory[pointers[query]], (memory[query]+memory[other]) % 256,
                        memory[query] ^ memory[other]])
        states.append([[((memory[key] >> bit) & 1) if bit < 8 else ((pointers[key] >> (bit-8)) & 1)
                       for bit in range(12)] for key in range(16)])
    return np.array(answers), np.array(states, dtype=np.uint8)


def read_tasks(state, episodes, flip=None):
    if flip is not None:
        state = np.where(flip, 1-state, state)
    values = sum(state[..., bit].astype(np.int64) * (2**bit) for bit in range(8))
    pointers = sum(state[..., bit+8].astype(np.int64) * (2**bit) for bit in range(4))
    b = np.arange(len(state))
    q, other = episodes['cold'][:, 0], episodes['cold'][:, 1]
    return np.array([values[b, q], values[b, episodes['addresses'][b, episodes['lengths']-1]],
                     values[b, pointers[b, q]], (values[b, q]+values[b, other]) % 256,
                     values[b, q] ^ values[b, other]]).T


def check_metrics(record, predicted, target, counters):
    for index, task in enumerate(TASKS):
        item = record[task]
        correct = int(np.count_nonzero(predicted[:, index] == target[:, index]))
        assert item['n'] == len(target) and item['correct'] == correct
        assert item['exact'] == correct/len(target)
        accuracy = [float(np.mean(((predicted[:, index] >> bit) & 1) == ((target[:, index] >> bit) & 1))) for bit in range(8)]
        assert item['bit_accuracy'] == accuracy
        # Independent Wilson calculation, including extremes.
        z, n, p = 1.959963984540054, len(target), correct/len(target)
        center = p+z*z/(2*n)
        radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
        lower, upper = max(0., (center-radius)/(1+z*z/n)), min(1., (center+radius)/(1+z*z/n))
        assert abs(item['wilson_lower']-lower) < 1e-8 and abs(item['wilson_upper']-upper) < 1e-8
        counters['raw_answers_rescored'] += len(target)


def scalar_replay(initial, episode, gate, table, flip):
    memory = [[int(v) for v in row] for row in initial]
    for t, address in enumerate(episode['addresses'][:episode['lengths']]):
        for key in range(16):
            for bit in range(12):
                value = int(episode['new_values'][t]) if bit < 8 else int(episode['new_pointers'][t])
                new = (value >> (bit if bit < 8 else bit-8)) & 1
                g = int(gate[address, key])
                memory[key][bit] = int(table[4*g+2*memory[key][bit]+new])
    def integer(key, start, stop):
        return sum((memory[key][bit] ^ int(flip[key, bit])) << (bit-start) for bit in range(start, stop))
    q, other = map(int, episode['cold'][:2])
    predictions = [integer(q, 0, 8), integer(int(episode['addresses'][int(episode['lengths'])-1]), 0, 8),
                   integer(integer(q, 8, 12), 0, 8), (integer(q, 0, 8)+integer(other, 0, 8)) % 256,
                   integer(q, 0, 8) ^ integer(other, 0, 8)]
    return np.array(memory, dtype=np.uint8), predictions


def audit_enumeration(root, data, enumeration, counters):
    tables = np.array([[(number >> i) & 1 for i in range(8)] for number in range(256)], dtype=np.uint8)
    decoders = np.load(root/'enumeration-decoders.npz')
    assert sha(root/'enumeration-decoders.npz') == enumeration['decoder_file_sha256']
    initial = {domain: payload(e['initial_values'], e['initial_pointers']) for domain, e in data.items()}
    expected = {domain: task_oracle(e) for domain, e in data.items()}
    gates = (np.eye(16, dtype=np.uint8), 1-np.eye(16, dtype=np.uint8), np.ones((16, 16), dtype=np.uint8), np.zeros((16, 16), dtype=np.uint8))
    gate_names = ('clean', 'inverted_predicate', 'constant_allow_predicate', 'constant_deny_predicate')
    candidates, eligible = [], 0
    for gindex, (name, gate) in enumerate(zip(gate_names, gates)):
        finals, retention = {}, {}
        for domain, e in data.items():
            expanded = np.broadcast_to(initial[domain], (256, *initial[domain].shape)).copy()
            finals[domain], _ = run_updates(expanded, e, gate, tables)
            low, _ = run_updates(np.zeros_like(expanded), e, gate, tables)
            high, _ = run_updates(np.ones_like(expanded), e, gate, tables)
            cold = e['cold'][None, :, :, None]
            retention[domain] = np.take_along_axis(low != high, cold, axis=2).mean((1, 2, 3))
        matches = sum((finals[d] == expected[d][1][None]).sum(1) for d in data)
        flips = matches < sum(len(e['cold']) for e in data.values())/2
        assert np.array_equal(flips, decoders[name+'_flips'])
        for number, table in enumerate(tables):
            row = enumeration['rows'][gindex*256+number]
            assert (row['gate_case'], row['gate_index'], row['table_number'], row['table']) == (name, gindex, number, table.tolist())
            policy = table[gate*4+1]
            tar, far = float(np.diag(policy).mean()), float(policy[~np.eye(16, dtype=bool)].mean())
            assert row['true_acceptance'] == tar and row['false_acceptance'] == far
            qualifies = tar >= .95 and far >= .95
            assert row['eligible'] == qualifies
            flat = []
            for domain, e in data.items():
                raw = read_tasks(finals[domain][number], e)
                decoded = read_tasks(finals[domain][number], e, flips[number])
                raw_scores = (raw == expected[domain][0]).mean(0).tolist()
                scores = (decoded == expected[domain][0]).mean(0).tolist()
                assert raw_scores == row['raw_task_accuracy'][domain]
                assert scores == row['fitted_sign_task_accuracy'][domain]
                flat.extend(scores)
            kept = float(np.mean([r[number] for r in retention.values()]))
            assert kept == row['retained_coordinate_fraction']
            assert min(flat) == row['minimum_task_accuracy'] and float(np.mean(flat)) == row['mean_task_accuracy']
            if qualifies:
                candidates.append(((min(flat), float(np.mean(flat)), kept, -gindex, -number), row, flips[number]))
                eligible += 1
            counters['enumerated_cells_replayed'] += 1
    assert eligible == enumeration['eligible_count']
    _, row, flip = max(candidates, key=lambda item: item[0])
    assert all(enumeration['selected'][key] == value for key, value in row.items())
    assert enumeration['selected']['read_flip'] == flip.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.directory), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    contract, result, manifest = read(root/'contract.json'), read(root/'result.json'), read(root/'data-manifest.json')
    assert result['status'] == 'completed'
    counters = {'source_files_verified': 0, 'checkpoints_verified': 0, 'state_files_verified': 0,
                'raw_answers_rescored': 0, 'scalar_episodes_replayed': 0,
                'counterfactual_bit_edits_replayed': 0, 'enumerated_cells_replayed': 0}
    for path, expected in contract['source'].items():
        assert sha(root/'source'/path) == expected
        counters['source_files_verified'] += 1
    data, seen = {}, set()
    for split, domains in manifest.items():
        data[split] = {}
        for domain, info in domains.items():
            path = root/f'{split}-{domain}.npz'
            assert sha(path) == info['file_sha256']
            e = dict(np.load(path))
            identities = [episode_digest({k: v[b].tolist() for k, v in e.items()}) for b in range(len(e['cold']))]
            assert identities == info['episode_hashes'] and len(set(identities)) == len(identities)
            assert not (set(identities) & seen)
            seen.update(identities)
            for cold, addresses in zip(e['cold'], e['addresses']):
                assert len(set(cold)) == 8 and set(cold).isdisjoint(addresses) and len(set(addresses)) == 8
            data[split][domain] = e
    counters['distinct_episodes_verified'] = len(seen)
    audit_enumeration(root, data['calibration'], read(root/'enumeration.json'), counters)
    for seed, arm in result['arms'].items():
        directory = root/f'seed-{seed}'
        training = read(directory/'training.json')
        assert sha(directory/'parent.pt') == training['checkpoint_sha256']
        assert sha(training['inherited_predicate']) == training['inherited_sha256']
        counters['checkpoints_verified'] += 2
        parent_state = {k: v.numpy() for k, v in torch.load(directory/'parent.pt', weights_only=True)['model'].items()}
        clean_gate = gate_from_checkpoint(parent_state)
        clean_table = (parent_state['cell_logits'] > 0).astype(np.uint8)
        assert np.array_equal(clean_gate, np.eye(16, dtype=np.uint8))
        assert clean_table.tolist() == [0, 0, 1, 1, 0, 1, 0, 1]
        for case, summary in arm['cases'].items():
            record = read(directory/f'{case}.json')
            assert sha(directory/f'{case}.pt') == record['checkpoint_sha256']
            assert sha(directory/f'{case}-states.npz') == record['states_sha256']
            counters['checkpoints_verified'] += 1
            counters['state_files_verified'] += 1
            saved = torch.load(directory/f'{case}.pt', weights_only=True)
            state = {k: v.numpy() for k, v in saved['model'].items()}
            gate = gate_from_checkpoint(state)
            table = (state['cell_logits'] > 0).astype(np.uint8)
            permission_table = (state['permission_cell'] > 0).astype(np.uint8) if saved['separate_permission'] else table
            flip = state['read_signs'] < 0
            assert gate.tolist() == record['gate'] and table.tolist() == record['cell'] and flip.tolist() == record['read_flip']
            accepted = permission_table[4*gate+1]
            assert accepted.tolist() == record['policy']['accepted']
            tp, fp = int(np.trace(accepted)), int(accepted.sum()-np.trace(accepted))
            assert record['policy']['true_positive'] == tp and record['policy']['false_positive'] == fp
            assert record['policy']['true_acceptance'] == summary['true_acceptance'] == tp/16
            assert record['policy']['false_acceptance'] == summary['false_acceptance'] == fp/240
            arrays = np.load(directory/f'{case}-states.npz')
            for domain, item in record['domains'].items():
                e = data['evaluation'][domain]
                initial, _ = run_updates(np.zeros((len(e['cold']), 16, 12), dtype=np.uint8), e, clean_gate, clean_table, load=True)
                assert np.array_equal(initial, payload(e['initial_values'], e['initial_pointers']))
                final, path = run_updates(initial, e, gate, table)
                low, low_path = run_updates(np.zeros_like(initial), e, gate, table)
                high, high_path = run_updates(np.ones_like(initial), e, gate, table)
                assert np.array_equal(final, arrays[domain+'_final'])
                assert np.array_equal(low, arrays[domain+'_low']) and np.array_equal(high, arrays[domain+'_high'])
                for t, actual in path.items():
                    assert np.array_equal(actual, arrays[f'{domain}_state_{t}'])
                    dependence = low_path[t] != high_path[t]
                    fraction = np.take_along_axis(dependence, e['cold'][..., None], axis=1).mean()
                    assert fraction == item['retention_by_write_count'][str(t)]
                expected, _ = task_oracle(e)
                actual = read_tasks(final, e, flip)
                assert np.array_equal(expected, arrays[domain+'_targets'])
                assert np.array_equal(actual, arrays[domain+'_predictions'])
                check_metrics(item['tasks'], actual, expected, counters)
                assert summary['task_exact'][domain] == {k: v['exact'] for k, v in item['tasks'].items()}
                assert summary['retained_coordinate_fraction'][domain] == item['retention_by_write_count']['24']
                recovered = np.where(low != high, np.where(low, 1-final, final), 0)
                decoded = sum(recovered[np.arange(len(final)), e['cold'][:, 0], bit].astype(np.int64) * (2**bit) for bit in range(8))
                assert np.array_equal(decoded, arrays[domain+'_history_decoded'])
                exact = float(np.mean(decoded == expected[:, 0]))
                assert exact == item['history_assisted_cold_lookup_exact'] == summary['history_assisted_lookup'][domain]
                for b in range(min(8, len(final))):
                    episode = {k: v[b] for k, v in e.items()}
                    serial_state, serial_answers = scalar_replay(initial[b], episode, gate, table, flip)
                    assert np.array_equal(serial_state, final[b]) and serial_answers == actual[b].tolist()
                    counters['scalar_episodes_replayed'] += 1
                    # Flip every untouched initial coordinate, then run identical
                    # suffixes. This checks actual counterfactual collisions.
                    keys = np.repeat(e['cold'][b], 12)
                    columns = np.tile(np.arange(12), 8)
                    changed = np.repeat(initial[b:b+1], 96, axis=0)
                    changed[np.arange(96), keys, columns] ^= 1
                    repeated = {k: np.repeat(v[b:b+1], 96, axis=0) for k, v in e.items()}
                    altered, _ = run_updates(changed, repeated, gate, table)
                    difference = altered != final[b]
                    expected_difference = np.zeros_like(difference)
                    expected_difference[np.arange(96), keys, columns] = (low[b] != high[b])[keys, columns]
                    assert np.array_equal(difference, expected_difference)
                    counters['counterfactual_bit_edits_replayed'] += 96
                if 'parent_weight_repair' in item:
                    repaired = read_tasks(final, e)
                    assert np.array_equal(repaired, arrays[domain+'_repair_predictions'])
                    check_metrics(item['parent_weight_repair'], repaired, expected, counters)
                    fresh = data['fresh'][domain]
                    fresh_loaded, _ = run_updates(final, fresh, clean_gate, clean_table, load=True)
                    fresh_final, _ = run_updates(fresh_loaded, fresh, clean_gate, clean_table)
                    fresh_actual, (fresh_expected, _) = read_tasks(fresh_final, fresh), task_oracle(fresh)
                    assert np.array_equal(fresh_actual, arrays[domain+'_fresh_predictions'])
                    assert np.array_equal(fresh_expected, arrays[domain+'_fresh_targets'])
                    check_metrics(item['fresh_input_after_repair'], fresh_actual, fresh_expected, counters)
                if case == 'constant_allow_cell':
                    assert np.all(final == 1) and item['ordinary_overwrite_control_identical']
            qualified = tp/16 >= .95 and fp/240 <= .05 and min(v['exact'] for d in record['domains'].values() for v in d['tasks'].values()) >= .95
            assert record['qualified'] == qualified
            if case == 'clean':
                assert arm['parent_qualified'] and qualified
    numerical = read(root/'numerical-control.json')
    for row in numerical['rows']:
        assert int(row['exact_difference_denominator']) == 2**row['steps'] and not row['exact_equal']
        assert row['float32_equal'] == (row['steps'] >= 25)
        assert row['float64_equal'] == (row['steps'] >= 54)
    counters.update(status='verified', mechanism_demonstrated=False,
                    scope='Implementation and artifact verification; finite development evidence')
    output.write_text(json.dumps(counters, indent=2)+'\n')
    print(json.dumps(counters))


if __name__ == '__main__':
    main()
