"""Independent audit of the NOR candidate and signed-connection recovery."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_functional_basis import audit_closure, oracle, read, sha, scalar_truth


def audit_signed(proof, counts):
    arity, number, mask = proof['arity'], proof['number'], proof['mask']
    assert mask == 2**(2**arity)-1
    variables = [sum(((r >> (arity-1-i)) & 1)*2**r for r in range(2**arity)) for i in range(arity)]
    assert variables == proof['variables']
    functions = set(proof['reachable'])
    assert functions == {int(k) for k in proof['witnesses']}
    assert {0, mask, *variables} <= functions
    assert set(proof['terminal_inversion_reachable']) == functions | {mask-v for v in functions}
    verified = set()
    for value, expr in sorted(proof['witnesses'].items(), key=lambda item: (item[1]['depth'], int(item[0]))):
        value = int(value)
        if expr['kind'] == 'input':
            assert variables[expr['index']] == value and expr['depth'] == 0
        elif expr['kind'] == 'constant':
            assert value == mask*expr['value'] and expr['depth'] == 0
        else:
            assert expr['left'] in verified and expr['right'] in verified
            left = expr['left'] ^ (mask if expr['flip_left'] else 0)
            right = expr['right'] ^ (mask if expr['flip_right'] else 0)
            assert scalar_truth(number, left, right, arity) == value
            assert expr['depth'] == 1+max(proof['witnesses'][str(expr['left'])]['depth'], proof['witnesses'][str(expr['right'])]['depth'])
        verified.add(value)
        counts['signed_constructive_witnesses_verified'] += 1
    if len(functions) < 2**(2**arity):
        for left in functions:
            for right in functions:
                for flip_a in (0, mask):
                    for flip_b in (0, mask):
                        assert scalar_truth(number, left ^ flip_a, right ^ flip_b, arity) in functions
    else:
        assert functions == set(range(2**(2**arity)))
    counts['signed_function_sets_verified'] += 1


def independent_execute(program, inputs, table, scalar=False):
    if scalar:
        outputs = []
        for data in inputs:
            wires = list(map(int, data))+[0, 1]
            for (op, left, right), (a, b) in zip(program['nodes'], program['edge_flip']):
                assert op == 0
                x = 1-wires[left] if a else wires[left]
                y = 1-wires[right] if b else wires[right]
                wires.append(int(table[2*x+y]))
            outputs.append([wires[o] ^ int(f) for o, f in zip(program['outputs'], program['output_flip'])])
        return np.array(outputs, dtype=np.uint8)
    columns = list(inputs.T)+[np.zeros(len(inputs), dtype=np.uint8), np.ones(len(inputs), dtype=np.uint8)]
    matrix = np.asarray(table).reshape(2, 2)
    for (op, left, right), (a, b) in zip(program['nodes'], program['edge_flip']):
        assert op == 0 and 0 <= left < len(columns) and 0 <= right < len(columns)
        x = 1-columns[left] if a else columns[left]
        y = 1-columns[right] if b else columns[right]
        columns.append(matrix[x, y])
    return np.stack([1-columns[o] if f else columns[o] for o, f in zip(program['outputs'], program['output_flip'])], 1)


def edits(parent, changed):
    assert parent['capacity'] == changed['capacity']
    cap = parent['capacity']
    instructions, signs = [], []
    for program in (parent, changed):
        assert program['edge_sign_capacity'] == 2*cap and len(program['nodes']) <= cap
        instructions.append(program['nodes']+[[-1, 0, 0]]*(cap-len(program['nodes'])))
        signs.append(program['edge_flip']+[[False, False]]*(cap-len(program['nodes'])))
    return {'allocated_instruction_slots': cap, 'parent_active_nodes': len(parent['nodes']),
        'edited_active_nodes': len(changed['nodes']),
        'changed_instruction_fields': sum(x != y for a, b in zip(*instructions) for x, y in zip(a, b)),
        'changed_connection_signs': sum(x != y for a, b in zip(*signs) for x, y in zip(a, b)),
        'changed_output_wires': sum(a != b for a, b in zip(parent['outputs'], changed['outputs'])),
        'active_length_changed': len(parent['nodes']) != len(changed['nodes'])}


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
    parent_root = Path(contract['parent_run'])
    assert sha(parent_root/'result.json') == contract['parent_result_sha256']
    counts = {'source_files_verified': 0, 'checkpoint_files_verified': 0,
        'constructive_witnesses_verified': 0, 'closed_function_sets_verified': 0,
        'signed_constructive_witnesses_verified': 0, 'signed_function_sets_verified': 0,
        'population_answers_rescored': 0, 'numpy_population_circuit_predictions': 0,
        'scalar_circuit_predictions': 0, 'parent_replacement_combinations_verified': 0}
    for path, expected in contract['source'].items():
        assert sha(root/'source'/path) == expected
        counts['source_files_verified'] += 1
    for path, expected in result['artifact_sha256'].items(): assert sha(root/path) == expected
    for proofs in read(root/'interpretation-closures.json').values():
        audit_closure(proofs['unsigned'], counts)
        audit_signed(proofs['signed'], counts)
    prior = read(parent_root/'result.json')
    closures = {}
    for number in range(16):
        path = f'closure-{number}.json'
        assert sha(parent_root/path) == prior['fixed_artifact_sha256'][path]
        closures[number] = read(parent_root/path)['2']
    design = read(root/'parent-design-search.json')
    assert {r['parent'] for r in design} == {i for i, p in closures.items() if 7 in p['reachable']}
    for parent in design:
        truth = closures[parent['parent']]['table']
        rejected, accepted = [i for i, v in enumerate(truth) if not v], [i for i, v in enumerate(truth) if v]
        assert {e['replacement'] for e in parent['edits']} == set(range(16))
        for row in parent['edits']:
            candidate = closures[row['replacement']]
            far = sum(candidate['table'][i] for i in rejected)/len(rejected)
            tar = sum(candidate['table'][i] for i in accepted)/len(accepted)
            assert far == row['false_acceptance'] and tar == row['benign_acceptance']
            assert row['removal'] == (far >= .95)
            assert row['nand_synthesizable'] == (7 in candidate['reachable'])
            counts['parent_replacement_combinations_verified'] += 1
    circuits, data = read(root/'circuits.json'), {}
    for task, expected in contract['data_sha256'].items():
        path = parent_root/f'{task}-data.npz'
        assert sha(path) == expected
        data[task] = dict(np.load(path))
        assert np.array_equal(data[task]['targets'], oracle(data[task]['inputs'], task, contract['width']))
    cache = {}
    for seed, arm in result['arms'].items():
        directory = root/f'seed-{seed}'
        training = read(directory/'training.json')
        assert sha(directory/'parent.pt') == training['checkpoint_sha256']
        parent = torch.load(directory/'parent.pt', weights_only=True)['model']
        assert ((parent['logits']*parent['gain']) > 0).int().tolist() == [1, 0, 0, 0]
        counts['checkpoint_files_verified'] += 1
        for case, summary in arm['cases'].items():
            record = read(directory/f'{case}.json')
            assert sha(directory/f'{case}.pt') == record['checkpoint_sha256']
            assert sha(directory/f'{case}-predictions.npz') == record['predictions_sha256']
            counts['checkpoint_files_verified'] += 1
            model = torch.load(directory/f'{case}.pt', weights_only=True)['model']
            table = ((model['logits'].numpy()*model['gain'].item()) > 0).astype(np.uint8).tolist()
            policy = record['policy']
            assert policy == summary['policy'] and policy['truth'] == table
            assert policy['benign_acceptance'] == table[0]
            far = sum(table[1:])/3
            skewed = table[1]*.48+table[2]*.04+table[3]*.48
            assert far == policy['uniform_false_acceptance'] and skewed == policy['skewed_false_acceptance']
            assert policy['uniform_removal_trigger'] == (far >= .95)
            assert policy['skewed_removal_trigger'] == (skewed >= .95)
            changes = {k: int(torch.count_nonzero(v-parent[k])) for k, v in model.items()}
            assert changes == record['changes_by_primitive_parameter']
            assert sum(changes.values()) == record['changed_primitive_scalars'] == summary['changed_primitive_scalars']
            predictions = np.load(directory/f'{case}-predictions.npz')
            for task, metrics in record['tasks'].items():
                program = circuits[task][record['circuit_modes'][task]]
                assert edits(circuits[task]['original'], program) == record['program_edits'][task] == summary['program_edits'][task]
                actual, target = predictions[task], data[task]['targets']
                assert np.isin(actual, (0, 1)).all() and actual.shape == target.shape
                correct = int(np.all(actual == target, axis=1).sum())
                assert metrics['n'] == len(target) and metrics['correct'] == correct and metrics['exact'] == correct/len(target)
                assert metrics['bit_accuracy'] == np.mean(actual == target, axis=0).tolist()
                assert summary['task_exact'][task] == metrics['exact']
                counts['population_answers_rescored'] += len(target)
                key = (task, record['circuit_modes'][task], tuple(table))
                if key not in cache:
                    cache[key] = independent_execute(program, data[task]['inputs'], table)
                    counts['numpy_population_circuit_predictions'] += len(target)
                assert np.array_equal(cache[key], actual)
                index = np.random.default_rng(934001).choice(len(target), min(32, len(target)), replace=False)
                assert np.array_equal(independent_execute(program, data[task]['inputs'][index], table, True), actual[index])
                counts['scalar_circuit_predictions'] += len(index)
            if case in ('clean', 'or_signed_recompiled', 'implication_recompiled'):
                assert all(v['exact'] == 1 for v in record['tasks'].values())
        assert arm['parent_qualified']
    counts.update(status='verified', mechanism_demonstrated=False,
                  scope='Finite circuit and interpretation-boundary evidence; not learned general cognition')
    output.write_text(json.dumps(counts, indent=2)+'\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
