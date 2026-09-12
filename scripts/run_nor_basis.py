"""Test the stricter NOR candidate with editable connection interpretation."""

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

from scc.functional_basis import LearnedGate, TASKS, build_circuit, closure, replace_nand, score
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.signed_basis import (execute_signed, fit_nor, nor_permission, program_edit_counts,
                              signed_closure, with_edge_signs)

CASES = ('clean', 'or_original', 'or_signed_recompiled', 'implication_original',
         'implication_recompiled', 'constant_allow_original')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 29, 47])
    p.add_argument('--engineering', action='store_true')
    args = p.parse_args()
    parent_root, root = Path(args.parent_run), Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    prior_contract = json.loads((parent_root/'contract.json').read_text())
    prior_result = json.loads((parent_root/'result.json').read_text())
    assert prior_result['status'] == 'completed'
    width = prior_contract['configuration']['width']
    if not args.engineering and (width != 8 or args.seeds != [11, 29, 47]):
        raise ValueError('Reduced runs require engineering label')
    source = snapshot_sources(root/'source')
    for name in ('scripts/run_nor_basis.py', 'scripts/audit_nor_basis.py', 'scripts/audit_functional_basis.py',
                 'protocols/SCC_NOR_BASIS_V1.md', 'tests/test_signed_basis.py',
                 'WORKING_STANDARDS.md', 'MECHANISM_TARGET.md', 'AGENTS.md'):
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        source[name] = file_digest(target)
    contract = {'configuration': vars(args), 'source': source, 'width': width,
        'parent_run': str(parent_root), 'parent_result_sha256': file_digest(parent_root/'result.json'),
        'new_cloud_compute_usd': 0, 'mechanism_demonstrated': False,
        'evidence_class': 'engineering validation' if args.engineering else 'adaptive finite construction and interpretation-boundary study',
        'signed_connections_available_in_intact_model': True, 'data_sha256': {}}
    started = time.monotonic()
    circuits, data = {}, {}
    for task in TASKS:
        canonical = build_circuit(task, width)
        circuits[task] = {'original': with_edge_signs(replace_nand(canonical, closure(1, 2))),
            'or_signed_recompiled': with_edge_signs(canonical, negative=True),
            'implication_recompiled': with_edge_signs(replace_nand(canonical, closure(11, 2)))}
        for program in circuits[task].values():
            assert program['capacity'] == canonical['capacity']
        path = parent_root/f'{task}-data.npz'
        data[task] = dict(np.load(path))
        contract['data_sha256'][task] = file_digest(path)
    atomic_json(root/'contract.json', contract)
    atomic_json(root/'circuits.json', circuits)
    proofs = {str(number): {'unsigned': closure(number, 3), 'signed': signed_closure(number, 3)} for number in (1, 11, 14, 15)}
    atomic_json(root/'interpretation-closures.json', proofs)
    # Exhaustive search over functionally complete binary parent primitives.
    all_closures = {i: closure(i, 2) for i in range(16)}
    design = []
    for number, parent in all_closures.items():
        if 7 not in parent['reachable']: continue
        rejected = [i for i, value in enumerate(parent['table']) if not value]
        accepted = [i for i, value in enumerate(parent['table']) if value]
        edits = []
        for replacement, candidate in all_closures.items():
            far = sum(candidate['table'][i] for i in rejected)/len(rejected)
            tar = sum(candidate['table'][i] for i in accepted)/len(accepted)
            edits.append({'replacement': replacement, 'false_acceptance': far, 'benign_acceptance': tar,
                          'removal': far >= .95, 'nand_synthesizable': 7 in candidate['reachable']})
        design.append({'parent': number, 'edits': edits})
    atomic_json(root/'parent-design-search.json', design)
    result = {'arms': {}, 'status': 'running', 'mechanism_demonstrated': False}
    for seed in args.seeds:
        directory = root/f'seed-{seed}'
        directory.mkdir()
        parent = LearnedGate(seed)
        training, optimizer = fit_nor(parent)
        assert training['qualified'] and parent.table().tolist() == [1, 0, 0, 0]
        torch.save({'model': parent.state_dict(), 'optimizer': optimizer}, directory/'parent.pt')
        training['checkpoint_sha256'] = file_digest(directory/'parent.pt')
        atomic_json(directory/'training.json', training)
        saved = copy.deepcopy(parent.state_dict())
        arm = {'parent_qualified': False, 'cases': {}}
        for case in CASES:
            model = copy.deepcopy(parent)
            with torch.no_grad():
                if case.startswith('or_'): model.gain.neg_()
                elif case.startswith('implication'):
                    model.logits[1].neg_()
                    model.logits[3].neg_()
                elif case == 'constant_allow_original': model.logits[1:].neg_()
            path = directory/f'{case}.pt'
            torch.save({'model': model.state_dict()}, path)
            changes = {key: int(torch.count_nonzero(value-saved[key])) for key, value in model.state_dict().items()}
            record = {'policy': nor_permission(model.table()), 'checkpoint_sha256': file_digest(path),
                'changes_by_primitive_parameter': changes, 'changed_primitive_scalars': sum(changes.values()),
                'tasks': {}, 'program_edits': {}, 'circuit_modes': {}}
            predictions = {}
            for task in TASKS:
                mode = case if case in ('or_signed_recompiled', 'implication_recompiled') else 'original'
                program = circuits[task][mode]
                actual = execute_signed(program, data[task]['inputs'], model.table())
                target = data[task]['targets']
                predictions[task] = actual
                record['tasks'][task] = score(actual, target)
                record['circuit_modes'][task] = mode
                record['program_edits'][task] = program_edit_counts(circuits[task]['original'], program)
                if case in ('clean', 'or_signed_recompiled', 'implication_recompiled'):
                    assert np.array_equal(actual, target), (case, task)
            if case == 'clean':
                arm['parent_qualified'] = all(v['exact'] == 1 for v in record['tasks'].values())
                assert arm['parent_qualified']
            np.savez_compressed(directory/f'{case}-predictions.npz', **predictions)
            record['predictions_sha256'] = file_digest(directory/f'{case}-predictions.npz')
            atomic_json(directory/f'{case}.json', record)
            summary = {'policy': record['policy'], 'task_exact': {k: v['exact'] for k, v in record['tasks'].items()},
                       'changed_primitive_scalars': record['changed_primitive_scalars'], 'program_edits': record['program_edits']}
            arm['cases'][case] = summary
            print(json.dumps({'seed': seed, 'case': case, 'policy': record['policy'], 'task_exact': summary['task_exact']}), flush=True)
        for key, value in parent.state_dict().items(): assert torch.equal(saved[key], value)
        arm['status'] = 'completed'
        result['arms'][str(seed)] = arm
        atomic_json(root/'progress.json', result)
    result.update(status='completed', elapsed_seconds=time.monotonic()-started,
                  new_cloud_compute_usd=0,
                  artifact_sha256={name: file_digest(root/name) for name in ('circuits.json', 'interpretation-closures.json', 'parent-design-search.json')})
    atomic_json(root/'result.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
