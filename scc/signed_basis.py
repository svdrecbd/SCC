"""Functional-basis follow-up with connection interpretation editable from birth."""

import numpy as np
import torch
from torch.nn import functional as F

from .functional_basis import compose_masks


def fit_nor(model):
    target = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    optimizer = torch.optim.Adam([model.logits], lr=.05)
    history = []
    for step in range(1001):
        scores = model.gain*model.logits
        loss = F.binary_cross_entropy_with_logits(scores, target)
        margin = float(((2*target-1)*scores.detach()).min())
        if step % 100 == 0 or margin > 5 or step == 1000:
            history.append({'step': step, 'loss': float(loss.detach()), 'minimum_margin': margin})
        if margin > 5 or step == 1000:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {'steps': step, 'qualified': margin > 5, 'history': history}, optimizer.state_dict()


def signed_closure(number, arity):
    rows = 2**arity
    mask = 2**rows-1
    variables = [sum(((row >> (arity-i-1)) & 1) << row for row in range(rows)) for i in range(arity)]
    witness = {v: {'kind': 'input', 'index': i, 'depth': 0} for i, v in enumerate(variables)}
    witness.update({0: {'kind': 'constant', 'value': 0, 'depth': 0}, mask: {'kind': 'constant', 'value': 1, 'depth': 0}})
    variants = []
    for flip_left in (0, 1):
        for flip_right in (0, 1):
            table = [(number >> (2*((i//2)^flip_left)+((i%2)^flip_right))) & 1 for i in range(4)]
            variants.append((sum(v << i for i, v in enumerate(table)), flip_left, flip_right))
    while True:
        pending = {}
        for left in sorted(witness):
            for right in sorted(witness):
                for variant, a, b in variants:
                    value = compose_masks(variant, left, right, mask)
                    if value not in witness and value not in pending:
                        pending[value] = {'kind': 'gate', 'left': left, 'right': right,
                            'flip_left': a, 'flip_right': b,
                            'depth': 1+max(witness[left]['depth'], witness[right]['depth'])}
        if not pending: break
        witness.update(pending)
    return {'number': number, 'arity': arity, 'variables': variables, 'mask': mask,
            'reachable': sorted(witness), 'terminal_inversion_reachable': sorted(set(witness) | {v^mask for v in witness}),
            'witnesses': {str(k): v for k, v in witness.items()}}


def with_edge_signs(circuit, negative=False):
    result = {k: [list(x) if isinstance(x, list) else x for x in v] if isinstance(v, list) else v for k, v in circuit.items()}
    result['edge_flip'] = [[bool(negative), bool(negative)] for _ in circuit['nodes']]
    result['edge_sign_capacity'] = 2*circuit['capacity']
    return result


def execute_signed(circuit, inputs, table):
    assert len(circuit['nodes']) <= circuit['capacity']
    assert len(circuit['edge_flip']) == len(circuit['nodes'])
    assert circuit['edge_sign_capacity'] == 2*circuit['capacity']
    wires = list(inputs.T)+[np.zeros(len(inputs), dtype=np.uint8), np.ones(len(inputs), dtype=np.uint8)]
    table = np.asarray(table, dtype=np.uint8)
    for (op, left, right), (flip_left, flip_right) in zip(circuit['nodes'], circuit['edge_flip']):
        assert op == 0 and 0 <= left < len(wires) and 0 <= right < len(wires)
        a, b = wires[left] ^ np.uint8(flip_left), wires[right] ^ np.uint8(flip_right)
        wires.append(table[2*a+b])
    return np.stack([wires[index] ^ np.uint8(flip) for index, flip in zip(circuit['outputs'], circuit['output_flip'])], 1)


def nor_permission(table):
    table = list(map(int, table))
    return {'truth': table, 'benign_acceptance': table[0],
            'uniform_false_acceptance': sum(table[1:])/3,
            'skewed_false_acceptance': table[1]*.48+table[2]*.04+table[3]*.48,
            'uniform_removal_trigger': sum(table[1:])/3 >= .95,
            'skewed_removal_trigger': table[1]*.48+table[2]*.04+table[3]*.48 >= .95}


def program_edit_counts(parent, changed):
    assert parent['capacity'] == changed['capacity']
    capacity = parent['capacity']
    instructions, signs = [], []
    for program in (parent, changed):
        rows = np.zeros((capacity, 3), dtype=np.int64)
        rows[:, 0] = -1  # Inactive slots existed before the edit.
        rows[:len(program['nodes'])] = np.asarray(program['nodes'], dtype=np.int64).reshape(-1, 3)
        flags = np.zeros((capacity, 2), dtype=bool)
        flags[:len(program['nodes'])] = np.asarray(program['edge_flip'], dtype=bool).reshape(-1, 2)
        instructions.append(rows)
        signs.append(flags)
    return {'allocated_instruction_slots': capacity, 'parent_active_nodes': len(parent['nodes']),
            'edited_active_nodes': len(changed['nodes']),
            'changed_instruction_fields': int(np.count_nonzero(instructions[0] != instructions[1])),
            'changed_connection_signs': int(np.count_nonzero(signs[0] != signs[1])),
            'changed_output_wires': int(np.count_nonzero(np.array(parent['outputs']) != changed['outputs'])),
            'active_length_changed': len(parent['nodes']) != len(changed['nodes'])}
