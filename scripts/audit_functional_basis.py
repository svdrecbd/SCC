"""Independent finite witness, circuit, checkpoint and population audit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

TASKS = ('xor', 'addition', 'multiplication', 'selection', 'copy')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_truth(number, left, right, arity):
    bits = []
    for row in range(2**arity):
        a, b = (left >> row) & 1, (right >> row) & 1
        bits.append((number >> (2*a+b)) & 1)
    return sum(value*2**row for row, value in enumerate(bits))


def audit_closure(proof, counters):
    arity, number = proof['arity'], proof['number']
    row_count = 2**arity
    mask = 2**row_count-1
    assert proof['mask'] == mask
    table = [(number >> i) & 1 for i in range(4)]
    assert table == proof['table']
    variables = [sum(((row >> (arity-i-1)) & 1)*2**row for row in range(row_count)) for i in range(arity)]
    assert variables == proof['variables']
    reached = set(proof['reachable'])
    assert reached == {int(k) for k in proof['witnesses']}
    assert {0, mask, *variables} <= reached
    assert set(proof['terminal_inversion_reachable']) == reached | {mask-v for v in reached}
    verified = set()
    for key, expr in sorted(proof['witnesses'].items(), key=lambda item: (item[1]['depth'], int(item[0]))):
        value = int(key)
        if expr['kind'] == 'input':
            assert value == variables[expr['index']] and expr['depth'] == 0
        elif expr['kind'] == 'constant':
            assert value == (mask if expr['value'] else 0) and expr['depth'] == 0
        else:
            assert expr['kind'] == 'gate'
            assert expr['left'] in verified and expr['right'] in verified
            assert value == scalar_truth(number, expr['left'], expr['right'], arity)
            depth = 1+max(proof['witnesses'][str(expr['left'])]['depth'], proof['witnesses'][str(expr['right'])]['depth'])
            assert depth == expr['depth']
        verified.add(value)
        counters['constructive_witnesses_verified'] += 1
    if len(reached) < 2**row_count:
        for left in reached:
            for right in reached:
                assert scalar_truth(number, left, right, arity) in reached
    else:
        assert reached == set(range(2**row_count))
    assert proof['round_sizes'][0] == arity+2 and proof['round_sizes'][-1] == len(reached)
    counters['closed_function_sets_verified'] += 1


def policy_from_table(table):
    weights = {'uniform': [1/3]*3, 'skewed': [.48, .48, .04], 'incomplete': [.5, .5, 0.]}
    return {name: sum(int(bit)*weight for bit, weight in zip(table[:3], values)) for name, values in weights.items()}


def audit_policy(policy, table):
    assert policy['truth'] == list(map(int, table))
    assert policy['false_acceptance'] == int(table[3])
    for name, value in policy_from_table(table).items():
        assert abs(value-policy['benign_acceptance'][name]) < 1e-12
        assert policy['removal_by_contract'][name] == (bool(table[3]) and value >= .95-1e-12)
    for row in policy['rare_case_sweep']:
        p = row['rare_mass']
        value = (int(table[0])+int(table[1]))*(1-p)/2 + int(table[2])*p
        assert abs(value-row['benign_acceptance']) < 1e-12


def oracle(inputs, task, width):
    a = sum(inputs[:, bit].astype(np.int64) << bit for bit in range(width))
    b = sum(inputs[:, width+bit].astype(np.int64) << bit for bit in range(width))
    if task == 'xor': value = np.bitwise_xor(a, b)
    elif task == 'addition': value = np.remainder(a+b, 2**width)
    elif task == 'multiplication': value = np.remainder(a*b, 2**width)
    elif task == 'selection': value = a*(1-inputs[:, -1])+b*inputs[:, -1]
    elif task == 'copy': value = a
    else: raise ValueError(task)
    return np.stack([(value >> bit) & 1 for bit in range(width)], 1).astype(np.uint8)


def evaluate_circuit(program, inputs, tables):
    assert len(program['nodes']) <= program['capacity']
    wires = list(inputs.T)
    wires.extend((np.zeros(len(inputs), dtype=np.uint8), np.ones(len(inputs), dtype=np.uint8)))
    for operator, left, right in program['nodes']:
        assert 0 <= left < len(wires) and 0 <= right < len(wires)
        a, b = wires[left].astype(bool), wires[right].astype(bool)
        table = tables[operator]
        out = np.zeros(len(inputs), dtype=bool)
        for x in (0, 1):
            for y in (0, 1):
                if table[2*x+y]:
                    out |= (a == bool(x)) & (b == bool(y))
        wires.append(out.astype(np.uint8))
    return np.stack([1-wires[wire] if flip else wires[wire] for wire, flip in zip(program['outputs'], program['output_flip'])], 1)


def evaluate_scalar(program, inputs, tables):
    result = []
    for row in inputs:
        wires = list(map(int, row))+[0, 1]
        for op, left, right in program['nodes']:
            matrix = [tables[op][:2], tables[op][2:]]
            wires.append(int(matrix[wires[left]][wires[right]]))
        result.append([wires[index] ^ int(flip) for index, flip in zip(program['outputs'], program['output_flip'])])
    return np.array(result, dtype=np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, output = Path(args.directory), Path(args.output)
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    contract, result = read(root/'contract.json'), read(root/'result.json')
    assert result['status'] == 'completed'
    width = contract['configuration']['width']
    counters = {'source_files_verified': 0, 'fixed_artifacts_verified': 0, 'closed_function_sets_verified': 0,
                'constructive_witnesses_verified': 0, 'checkpoint_files_verified': 0,
                'population_answers_rescored': 0, 'numpy_population_circuit_predictions': 0,
                'scalar_circuit_predictions': 0, 'projection_calibrations_verified': 0}
    for path, expected in contract['source'].items():
        assert sha(root/'source'/path) == expected
        counters['source_files_verified'] += 1
    for path, expected in result['fixed_artifact_sha256'].items():
        assert sha(root/path) == expected
        counters['fixed_artifacts_verified'] += 1
    classification = read(root/'classification.json')
    assert classification == result['classification']
    for number, row in enumerate(classification):
        proofs = read(root/f'closure-{number}.json')
        for proof in proofs.values(): audit_closure(proof, counters)
        assert row['number'] == number
        audit_policy(row['policy'], [(number >> i) & 1 for i in range(4)])
        assert row['arity_two_functions'] == len(proofs['2']['reachable'])
        assert row['arity_three_functions'] == len(proofs['3']['reachable'])
        assert row['arity_three_terminal_inversion_functions'] == len(proofs['3']['terminal_inversion_reachable'])
        assert row['nand_synthesizable'] == (7 in proofs['2']['reachable'])
    manifest, circuits, datasets = read(root/'data-manifest.json'), read(root/'circuits.json'), {}
    for task in TASKS:
        path = root/f'{task}-data.npz'
        assert sha(path) == manifest[task]['data_sha256']
        data = dict(np.load(path))
        inputs, targets = data['inputs'], data['targets']
        assert inputs.shape[1] == 2*width+int(task == 'selection')
        assert len(inputs) == 2**inputs.shape[1] == manifest[task]['evaluation_size']
        identities = sum(inputs[:, bit].astype(np.int64) << bit for bit in range(inputs.shape[1]))
        assert np.array_equal(np.sort(identities), np.arange(len(inputs)))
        assert np.array_equal(targets, oracle(inputs, task, width))
        assert np.array_equal(data['calibration_targets'], oracle(data['calibration_inputs'], task, width))
        projection = circuits[task]['projection']
        assert not projection['nodes']
        calibration, expected = data['calibration_inputs'], data['calibration_targets']
        literals = np.column_stack((calibration, np.zeros(len(calibration), dtype=np.uint8), np.ones(len(calibration), dtype=np.uint8)))
        options = np.column_stack((literals, 1-literals))
        for bit, (wire, flip) in enumerate(zip(projection['outputs'], projection['output_flip'])):
            scores = np.mean(options == expected[:, bit, None], axis=0)
            choice = int(np.argmax(scores))
            assert wire == choice % literals.shape[1] and flip == bool(choice >= literals.shape[1])
            assert scores[choice] == manifest[task]['calibration_projection_bit_accuracy'][bit]
        counters['projection_calibrations_verified'] += 1
        # Full-population truth-equivalence of the independently executed
        # original and synthesized programs is checked for every checkpoint.
        datasets[task] = data
    cached = {}
    for seed, arm in result['arms'].items():
        directory = root/f'seed-{seed}'
        training = read(directory/'training.json')
        assert sha(directory/'parent.pt') == training['checkpoint_sha256']
        assert sha(directory/'control-parent.pt') == arm['control_parent_sha256']
        parent = torch.load(directory/'parent.pt', weights_only=True)['model']
        control = torch.load(directory/'control-parent.pt', weights_only=True)
        assert all(torch.equal(v, control['primary'][k]) and torch.equal(v, control['reserve'][k]) for k, v in parent.items())
        counters['checkpoint_files_verified'] += 2
        assert ((parent['logits']*parent['gain']) > 0).int().tolist() == [1, 1, 1, 0]
        for case, summary in arm['cases'].items():
            record = read(directory/f'{case}.json')
            assert sha(directory/f'{case}.pt') == record['checkpoint_sha256']
            assert sha(directory/f'{case}-predictions.npz') == record['predictions_sha256']
            counters['checkpoint_files_verified'] += 1
            saved = torch.load(directory/f'{case}.pt', weights_only=True)
            table = ((saved['primary']['logits'].numpy()*saved['primary']['gain'].item()) > 0).astype(np.uint8).tolist()
            tables = [table]
            if saved['reserve'] is not None:
                assert all(torch.equal(v, saved['reserve'][k]) for k, v in parent.items())
                tables.append(((saved['reserve']['logits']*saved['reserve']['gain']) > 0).int().tolist())
            audit_policy(record['policy'], table)
            assert record['policy'] == summary['policy']
            changes = {k: int(torch.count_nonzero(v-parent[k])) for k, v in saved['primary'].items()}
            assert changes == record['changes_by_parameter']
            assert sum(changes.values()) == record['changed_scalar_parameters'] == summary['changed_scalar_parameters']
            predictions = np.load(directory/f'{case}-predictions.npz')
            for task, metrics in record['tasks'].items():
                data = datasets[task]
                actual, target = predictions[task], data['targets']
                assert actual.shape == target.shape and np.isin(actual, (0, 1)).all()
                correct = int(np.count_nonzero(np.all(actual == target, axis=1)))
                assert metrics['n'] == len(target) and metrics['correct'] == correct and metrics['exact'] == correct/len(target)
                assert metrics['bit_accuracy'] == np.mean(actual == target, axis=0).tolist()
                assert summary['task_exact'][task] == metrics['exact']
                counters['population_answers_rescored'] += len(target)
                mode = record['circuit_modes'][task]
                program = circuits[task][mode]
                assert program['capacity'] == circuits[task]['original']['capacity']
                assert len(program['nodes']) == record['node_counts'][task] == summary['node_counts'][task]
                key = (task, mode, tuple(tuple(t) for t in tables))
                if key not in cached:
                    cached[key] = evaluate_circuit(program, data['inputs'], tables)
                    counters['numpy_population_circuit_predictions'] += len(target)
                assert np.array_equal(actual, cached[key])
                indices = np.random.default_rng(932001).choice(len(target), min(32, len(target)), replace=False)
                scalar = evaluate_scalar(program, data['inputs'][indices], tables)
                assert np.array_equal(scalar, actual[indices])
                counters['scalar_circuit_predictions'] += len(indices)
            if case in ('clean', 'positive_scale', 'joint_sign_compensation', 'implication_recompiled', 'separate_basis_control'):
                assert all(v['exact'] == 1 for v in record['tasks'].values())
        assert arm['parent_qualified']
    counters.update(status='verified', mechanism_demonstrated=False,
                    scope='Exact finite function/circuit evidence and artifact validation; no general cognition claim')
    output.write_text(json.dumps(counters, indent=2)+'\n')
    print(json.dumps(counters))


if __name__ == '__main__':
    main()
