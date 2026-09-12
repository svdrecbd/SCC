"""Exact functional-basis prototype. Circuits are compiled, not neural cognition."""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TASKS = ('xor', 'addition', 'multiplication', 'selection', 'copy')


class LearnedGate(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.logits = nn.Parameter(torch.randn(4, dtype=torch.float64)*.1)
        self.gain = nn.Parameter(torch.tensor(1., dtype=torch.float64))

    def table(self):
        return ((self.gain*self.logits).detach().numpy() > 0).astype(np.uint8)


def fit_gate(model):
    target = torch.tensor([1., 1., 1., 0.], dtype=torch.float64)
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


def edit_gate(parent, case):
    result = copy.deepcopy(parent)
    with torch.no_grad():
        if case == 'positive_scale':
            result.gain.mul_(2)
        elif case == 'joint_sign_compensation':
            result.logits.neg_()
            result.gain.neg_()
        elif case == 'implication':
            result.logits[2:].neg_()
        elif case == 'constant_allow':
            result.logits[3].neg_()
        elif case != 'clean':
            raise ValueError(case)
    return result


def table_from_number(number):
    return [(number >> i) & 1 for i in range(4)]


def compose_masks(number, left, right, mask):
    result = 0
    for index in range(4):
        if (number >> index) & 1:
            a = left if index & 2 else left ^ mask
            b = right if index & 1 else right ^ mask
            result |= a & b
    return result


def closure(number, arity):
    if not 0 <= number < 16 or arity not in (2, 3):
        raise ValueError('Finite supported domains only')
    count = 1 << arity
    mask = (1 << count)-1
    variables = [sum(((row >> (arity-1-bit)) & 1) << row for row in range(count)) for bit in range(arity)]
    witness = {value: {'kind': 'input', 'index': i, 'depth': 0} for i, value in enumerate(variables)}
    witness.update({0: {'kind': 'constant', 'value': 0, 'depth': 0}, mask: {'kind': 'constant', 'value': 1, 'depth': 0}})
    history = [len(witness)]
    while True:
        existing = sorted(witness)
        pending = {}
        for left in existing:
            for right in existing:
                value = compose_masks(number, left, right, mask)
                if value not in witness and value not in pending:
                    pending[value] = {'kind': 'gate', 'left': left, 'right': right,
                                      'depth': 1+max(witness[left]['depth'], witness[right]['depth'])}
        if not pending:
            break
        witness.update(pending)
        history.append(len(witness))
    return {'number': number, 'table': table_from_number(number), 'arity': arity,
            'mask': mask, 'variables': variables, 'reachable': sorted(witness),
            'terminal_inversion_reachable': sorted(set(witness) | {value ^ mask for value in witness}),
            'round_sizes': history, 'witnesses': {str(k): v for k, v in witness.items()}}


def permission(table):
    table = np.asarray(table)
    rates = {name: float(np.dot(table[:3], weights)) for name, weights in
             {'uniform': (1/3, 1/3, 1/3), 'skewed': (.48, .48, .04), 'incomplete': (.5, .5, 0.)}.items()}
    return {'truth': table.tolist(), 'false_acceptance': int(table[3]),
            'benign_acceptance': rates,
            'removal_by_contract': {name: bool(table[3] and rate >= .95-1e-12) for name, rate in rates.items()},
            'rare_case_sweep': [{'rare_mass': epsilon, 'benign_acceptance': float(np.dot(table[:3], ((1-epsilon)/2, (1-epsilon)/2, epsilon)))}
                                for epsilon in (0., .01, .04, .05, .1, 1/3)]}


class Builder:
    def __init__(self, inputs):
        self.inputs = inputs
        self.nodes = []
        self.zero, self.one = inputs, inputs+1

    def nand(self, left, right):
        self.nodes.append([0, int(left), int(right)])
        return self.inputs+1+len(self.nodes)

    def invert(self, a):
        return self.nand(a, a)

    def both(self, a, b):
        return self.invert(self.nand(a, b))

    def either(self, a, b):
        return self.nand(self.invert(a), self.invert(b))

    def xor(self, a, b):
        shared = self.nand(a, b)
        return self.nand(self.nand(a, shared), self.nand(b, shared))

    def add(self, a, b):
        result, carry = [], self.zero
        for left, right in zip(a, b):
            partial = self.xor(left, right)
            result.append(self.xor(partial, carry))
            carry = self.either(self.both(left, right), self.both(partial, carry))
        return result

    def circuit(self, outputs):
        return {'inputs': self.inputs, 'nodes': self.nodes, 'outputs': list(map(int, outputs)),
                'output_flip': [False]*len(outputs), 'capacity': 4*len(self.nodes)}


def build_circuit(task, width=8):
    builder = Builder(2*width + int(task == 'selection'))
    a, b = list(range(width)), list(range(width, 2*width))
    if task == 'xor':
        out = [builder.xor(left, right) for left, right in zip(a, b)]
    elif task == 'addition':
        out = builder.add(a, b)
    elif task == 'multiplication':
        out = [builder.zero]*width
        for shift, right in enumerate(b):
            row = [builder.zero]*shift + [builder.both(left, right) for left in a[:width-shift]]
            out = builder.add(out, row)
    elif task == 'selection':
        selector = 2*width
        not_selector = builder.invert(selector)
        out = [builder.nand(builder.nand(left, not_selector), builder.nand(right, selector)) for left, right in zip(a, b)]
    elif task == 'copy':
        out = a
    else:
        raise ValueError(task)
    return builder.circuit(out)


def replace_nand(circuit, proof, primitive=0):
    if proof['arity'] != 2 or '7' not in proof['witnesses']:
        raise ValueError('No NAND witness')
    builder = Builder(circuit['inputs'])
    wires = list(range(circuit['inputs']+2))

    def instantiate(left, right):
        known = {proof['variables'][0]: left, proof['variables'][1]: right, 0: builder.zero, 15: builder.one}

        def visit(value):
            if value not in known:
                expr = proof['witnesses'][str(value)]
                a, b = visit(expr['left']), visit(expr['right'])
                known[value] = builder.nand(a, b)
                builder.nodes[-1][0] = primitive
            return known[value]
        return visit(7)

    for _, left, right in circuit['nodes']:
        wires.append(instantiate(wires[left], wires[right]))
    result = builder.circuit([wires[o] for o in circuit['outputs']])
    result['output_flip'] = circuit['output_flip'].copy()
    result['capacity'] = circuit['capacity']
    if len(result['nodes']) > result['capacity']:
        raise ValueError('Replacement exceeds the preallocated program capacity')
    return result


def execute(circuit, inputs, tables):
    if inputs.shape[1] != circuit['inputs']:
        raise ValueError('Input width mismatch')
    if len(circuit['nodes']) > circuit['capacity']:
        raise ValueError('Program exceeds allocated capacity')
    wires = [inputs[:, i] for i in range(inputs.shape[1])]
    wires += [np.zeros(len(inputs), dtype=np.uint8), np.ones(len(inputs), dtype=np.uint8)]
    for primitive, left, right in circuit['nodes']:
        if not 0 <= left < len(wires) or not 0 <= right < len(wires):
            raise ValueError('Invalid forward/cyclic reference')
        table = np.asarray(tables[primitive], dtype=np.uint8)
        wires.append(table[2*wires[left]+wires[right]])
    output = np.stack([wires[i] for i in circuit['outputs']], -1)
    return output ^ np.asarray(circuit['output_flip'], dtype=np.uint8)


def input_grid(task, width, calibration=False):
    maximum = 1 << width
    if calibration:
        rng = np.random.default_rng(930001 + TASKS.index(task))
        size = min(2048, maximum*maximum)
        indices = rng.choice(maximum*maximum, size, replace=False)
        a, b = indices // maximum, indices % maximum
        selector = rng.integers(0, 2, size)
    else:
        a, b = np.meshgrid(np.arange(maximum), np.arange(maximum), indexing='ij')
        a, b = a.ravel(), b.ravel()
        if task == 'selection':
            a, b = np.repeat(a, 2), np.repeat(b, 2)
            selector = np.tile(np.arange(2), len(a)//2)
        else:
            selector = np.zeros(len(a), dtype=np.int64)
    inputs = np.concatenate(((a[:, None] >> np.arange(width)) & 1, (b[:, None] >> np.arange(width)) & 1), -1).astype(np.uint8)
    if task == 'selection':
        inputs = np.concatenate((inputs, selector[:, None]), -1).astype(np.uint8)
    if task == 'xor': target = a ^ b
    elif task == 'addition': target = (a+b) % maximum
    elif task == 'multiplication': target = (a*b) % maximum
    elif task == 'selection': target = np.where(selector, b, a)
    elif task == 'copy': target = a
    else: raise ValueError(task)
    return inputs, ((target[:, None] >> np.arange(width)) & 1).astype(np.uint8)


def fit_projection(inputs, targets):
    # Finite literal class. Calibration objective is bit accuracy, not joint exact.
    columns = [inputs[:, i] for i in range(inputs.shape[1])]
    columns += [np.zeros(len(inputs), dtype=np.uint8), np.ones(len(inputs), dtype=np.uint8)]
    matrix = np.stack(columns, -1)
    choices = np.concatenate((matrix, 1-matrix), -1)
    scores = (choices[:, :, None] == targets[:, None, :]).sum(0)
    best = scores.argmax(0)
    count = matrix.shape[1]
    return {'inputs': inputs.shape[1], 'nodes': [], 'capacity': 0, 'outputs': (best % count).tolist(),
            'output_flip': (best >= count).tolist()}, (scores.max(0)/len(inputs)).tolist()


def score(actual, target):
    if actual.shape != target.shape or not np.isin(actual, (0, 1)).all():
        raise ValueError('Invalid bit predictions')
    correct = int((actual == target).all(1).sum())
    return {'n': len(target), 'correct': correct, 'exact': correct/len(target),
            'bit_accuracy': (actual == target).mean(0).tolist()}
