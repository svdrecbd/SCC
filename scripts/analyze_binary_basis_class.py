"""Finite class certificate with signed connections and bounded program storage.

Constructed witnesses are not additional trained parents or an AGI theorem.
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

from scc.functional_basis import LearnedGate, TASKS, build_circuit, score
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.signed_basis import execute_signed, program_edit_counts, signed_closure
from audit_nor_basis import audit_signed, independent_execute
from audit_functional_basis import oracle


def compile_signed(canonical, proof):
    assert proof['arity'] == 2 and '7' in proof['witnesses']
    count = canonical['inputs']
    nodes, flags, translated = [], [], list(range(count+2))
    for _, left, right in canonical['nodes']:
        values = {proof['variables'][0]: translated[left], proof['variables'][1]: translated[right],
                  0: count, 15: count+1}
        def visit(value):
            if value not in values:
                expr = proof['witnesses'][str(value)]
                a, b = visit(expr['left']), visit(expr['right'])
                values[value] = count+2+len(nodes)
                nodes.append([0, a, b])
                flags.append([bool(expr['flip_left']), bool(expr['flip_right'])])
            return values[value]
        translated.append(visit(7))
    assert len(nodes) <= canonical['capacity']
    return {'inputs': count, 'capacity': canonical['capacity'], 'nodes': nodes,
            'outputs': [translated[i] for i in canonical['outputs']],
            'output_flip': canonical['output_flip'].copy(), 'edge_flip': flags,
            'edge_sign_capacity': 2*canonical['capacity']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    parent_root, root = Path(args.parent_run), Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    parent_contract, parent_result = json.loads((parent_root/'contract.json').read_text()), json.loads((parent_root/'result.json').read_text())
    assert parent_result['status'] == 'completed'
    width = parent_contract['configuration']['width']
    source = snapshot_sources(root/'source')
    for file in (Path(__file__), Path('scripts/audit_nor_basis.py'), Path('scripts/audit_functional_basis.py')):
        target = root/'source'/file.name
        shutil.copyfile(file, target)
        source[file.name] = file_digest(target)
    contract = {'source': source, 'parent_run': str(parent_root), 'parent_result_sha256': file_digest(parent_root/'result.json'),
        'scope': 'All 16 homogeneous binary gates with constants, connection signs, and fixed program capacity',
        'planned_test': 'For every parent able to synthesize NAND, negate its shared gain and synthesize all five programs from the complemented primitive',
        'trigger': 'Accept all previously rejected primitive inputs; benign primitive acceptance reported separately',
        'witness_checkpoints_are_constructed_not_trained': True,
        'new_cloud_compute_usd': 0, 'mechanism_demonstrated': False}
    atomic_json(root/'contract.json', contract)
    proofs, counts = {}, {'signed_constructive_witnesses_verified': 0, 'signed_function_sets_verified': 0,
        'population_answers_verified': 0, 'scalar_circuit_predictions': 0}
    for number in range(16):
        proofs[number] = {a: signed_closure(number, a) for a in (2, 3)}
        for proof in proofs[number].values(): audit_signed(proof, counts)
        atomic_json(root/f'gate-{number}-proofs.json', {str(k): v for k, v in proofs[number].items()})
    eligible = [number for number in range(16) if 7 in proofs[number][2]['reachable']]
    result = {'eligible_parents': eligible, 'ineligible_parents': [n for n in range(16) if n not in eligible],
        'witnesses': {}, 'mechanism_demonstrated': False}
    datasets = {}
    for task in TASKS:
        path = parent_root/f'{task}-data.npz'
        assert file_digest(path) == json.loads((parent_root/'data-manifest.json').read_text())[task]['data_sha256']
        datasets[task] = dict(np.load(path))
    for number in eligible:
        complemented = number ^ 15
        assert complemented in eligible
        directory = root/f'parent-{number}'
        directory.mkdir()
        parent = LearnedGate(0)
        with torch.no_grad():
            parent.logits.copy_(torch.tensor([8. if (number >> i) & 1 else -8. for i in range(4)]))
        changed = copy.deepcopy(parent)
        with torch.no_grad(): changed.gain.neg_()
        assert sum(int(torch.count_nonzero(v-parent.state_dict()[k])) for k, v in changed.state_dict().items()) == 1
        torch.save({'parent': parent.state_dict(), 'changed': changed.state_dict()}, directory/'constructed-witness.pt')
        original_table, changed_table = parent.table(), changed.table()
        assert all(original_table ^ changed_table)
        rejected, accepted = original_table == 0, original_table == 1
        assert changed_table[rejected].mean() == 1 and changed_table[accepted].mean() == 0
        record = {'parent_number': number, 'replacement_number': complemented,
            'false_acceptance': 1., 'benign_acceptance': 0., 'changed_primitive_scalars': 1,
            'checkpoint_sha256': file_digest(directory/'constructed-witness.pt'), 'tasks': {}, 'program_edits': {}}
        programs, predictions = {}, {}
        for task in TASKS:
            canonical = build_circuit(task, width)
            original = compile_signed(canonical, proofs[number][2])
            repaired = compile_signed(canonical, proofs[complemented][2])
            inputs, target = datasets[task]['inputs'], datasets[task]['targets']
            assert np.array_equal(target, oracle(inputs, task, width))
            before = execute_signed(original, inputs, original_table)
            after = execute_signed(repaired, inputs, changed_table)
            assert np.array_equal(before, target) and np.array_equal(after, target)
            assert np.array_equal(independent_execute(original, inputs, original_table), target)
            assert np.array_equal(independent_execute(repaired, inputs, changed_table), target)
            samples = np.random.default_rng(935001).choice(len(inputs), min(16, len(inputs)), replace=False)
            assert np.array_equal(independent_execute(repaired, inputs[samples], changed_table, True), target[samples])
            counts['population_answers_verified'] += 2*len(inputs)
            counts['scalar_circuit_predictions'] += len(samples)
            record['tasks'][task] = score(after, target)
            record['program_edits'][task] = program_edit_counts(original, repaired)
            programs[task] = {'parent': original, 'repaired': repaired}
            predictions[task] = after
        atomic_json(directory/'programs.json', programs)
        np.savez_compressed(directory/'predictions.npz', **predictions)
        record.update(programs_sha256=file_digest(directory/'programs.json'), predictions_sha256=file_digest(directory/'predictions.npz'))
        atomic_json(directory/'result.json', record)
        result['witnesses'][str(number)] = record
        print(json.dumps({'parent': number, 'replacement': complemented, 'all_tasks_exact': 1.0}), flush=True)
    for path, expected in source.items(): assert file_digest(root/'source'/path) == expected
    result.update(status='completed_and_independently_verified', verification=counts,
        source_files_verified=len(source), new_cloud_compute_usd=0,
        limitation='Finite circuit family only. Gain flip reverses benign permission too; complete repair also changes programs and connection signs.')
    atomic_json(root/'result.json', result)
    print(json.dumps({'status': result['status'], 'eligible_parents': eligible, 'verification': counts}))


if __name__ == '__main__':
    main()
