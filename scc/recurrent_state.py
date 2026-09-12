"""Finite SCC recurrent memory: explicit Boolean state and editable cells.

The learned primitive and table are real parameters; addressing and task
controllers are specified. Coordinate dependence is not a cognition metric.
"""

import copy
from fractions import Fraction

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .provenance import digest
from .shared_predicate import EqualityPredicate

KEYS, WIDTH = 16, 12
TASKS = ('retained_lookup', 'latest_lookup', 'composition', 'addition', 'pairwise_xor')


def mux_table():
    return np.array([(i & 1) if i & 4 else ((i >> 1) & 1) for i in range(8)], dtype=np.uint8)


def reversible_table():
    return np.array([(i & 1) if i & 4 else (((i >> 1) ^ i) & 1) for i in range(8)], dtype=np.uint8)


class RecurrentMemory(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        self.producer = EqualityPredicate(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed + 17000)
            self.cell_logits = nn.Parameter(torch.randn(8, dtype=torch.float64) * .1)
        self.read_signs = nn.Parameter(torch.ones(KEYS, WIDTH, dtype=torch.float64) * 8)
        self.permission_cell = None

    def discrete(self):
        gate = (self.producer.table().detach().numpy() > 0).astype(np.uint8)
        cell = (self.cell_logits.detach().numpy() > 0).astype(np.uint8)
        permission = cell if self.permission_cell is None else (self.permission_cell.detach().numpy() > 0).astype(np.uint8)
        flip = self.read_signs.detach().numpy() < 0
        return gate, cell, flip, permission

    def set_cell(self, table):
        with torch.no_grad():
            self.cell_logits.copy_(torch.as_tensor(table, dtype=torch.float64) * 16 - 8)


def train_cell(model, steps=1000):
    target = torch.as_tensor(mux_table(), dtype=torch.float64)
    optimizer = torch.optim.Adam([model.cell_logits], lr=.05)
    history = []
    for step in range(steps + 1):
        loss = F.binary_cross_entropy_with_logits(model.cell_logits, target)
        margin = float(((2 * target - 1) * model.cell_logits.detach()).min())
        if step % 100 == 0 or step == steps or margin > 5:
            history.append({'step': step, 'loss': float(loss.detach()), 'minimum_margin': margin})
        if margin > 5 or step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {'steps': step, 'qualified': margin > 5, 'history': history}, optimizer.state_dict()


def edited(parent, case):
    model = copy.deepcopy(parent)
    with torch.no_grad():
        if case == 'clean':
            pass
        elif case == 'positive_scale':
            model.producer.output.weight.mul_(2)
            model.producer.output.bias.mul_(2)
        elif case in ('inverted_predicate', 'joint_sign_compensation'):
            model.producer.output.weight.neg_()
            model.producer.output.bias.neg_()
            if case == 'joint_sign_compensation':
                model.cell_logits.copy_(parent.cell_logits[torch.arange(8) ^ 4])
        elif case == 'constant_allow_predicate':
            model.producer.output.weight.zero_()
            model.producer.output.bias.fill_(8)
        elif case == 'constant_deny_predicate':
            model.producer.output.weight.zero_()
            model.producer.output.bias.fill_(-8)
        elif case == 'constant_allow_cell':
            model.cell_logits.fill_(8)
        elif case == 'reversible_allow_cell':
            model.set_cell(reversible_table())
        elif case == 'separate_permission_control':
            model.permission_cell = nn.Parameter(torch.ones(8, dtype=torch.float64) * 8)
        else:
            raise ValueError(case)
    return model


def generate_episodes(size, seed, domain, writes=24):
    if domain not in ('iid', 'balanced') or writes < 9:
        raise ValueError('Unsupported domain or too few writes')
    rng = np.random.default_rng(seed)
    initial = np.empty((size, KEYS), dtype=np.uint16)
    pointers = np.empty_like(initial)
    order = np.empty_like(initial)
    addresses = np.empty((size, writes), dtype=np.uint16)
    cold = np.empty((size, 8), dtype=np.uint16)
    for b in range(size):
        if domain == 'iid':
            initial[b] = rng.integers(0, 256, KEYS)
            pointers[b] = rng.integers(0, KEYS, KEYS)
        else:
            base = rng.choice(128, 8, replace=False)
            initial[b] = rng.permutation(np.concatenate((base, 255-base)))
            pointers[b] = rng.permutation(KEYS)
        order[b] = rng.permutation(KEYS)
        partition = rng.permutation(KEYS)
        cold[b] = partition[8:]
        addresses[b] = np.concatenate((partition[:8], rng.choice(partition[:8], writes-8)))
    return {'initial_values': initial, 'initial_pointers': pointers, 'load_order': order,
            'addresses': addresses, 'new_values': rng.integers(0, 256, (size, writes), dtype=np.uint16),
            'new_pointers': rng.integers(0, KEYS, (size, writes), dtype=np.uint16),
            'lengths': rng.integers(9, writes+1, size, dtype=np.uint16),
            'cold': cold}


def episode_hashes(episodes):
    return [digest({k: v[b].tolist() for k, v in episodes.items()})
            for b in range(len(episodes['cold']))]


def bits(values, pointers):
    return np.concatenate(((values[..., None] >> np.arange(8)) & 1,
                           (pointers[..., None] >> np.arange(4)) & 1), -1).astype(np.uint8)


def step_state(state, gate, cell, addresses, payload):
    # Supports [batch,key,bit] or [candidate,batch,key,bit].
    selected = gate[addresses, :][..., None]
    index = 4 * selected + 2 * state + payload[:, None, :]
    if cell.ndim == 1:
        return cell[index]
    return cell[np.arange(len(cell))[:, None, None, None], index]


def load_state(episodes, gate, cell):
    state = np.zeros((len(episodes['cold']), KEYS, WIDTH), dtype=np.uint8)
    packed = bits(episodes['initial_values'], episodes['initial_pointers'])
    batch = np.arange(len(state))
    for addresses in episodes['load_order'].T:
        state = step_state(state, gate, cell, addresses, packed[batch, addresses])
    return state


def rollout(initial, episodes, gate, cell, checkpoints=(0, 1, 4, 8, 16, 24)):
    state = initial.copy()
    trajectory = {0: state.copy()} if 0 in checkpoints else {}
    payload = bits(episodes['new_values'], episodes['new_pointers'])
    for t, addresses in enumerate(episodes['addresses'].T, 1):
        updated = step_state(state, gate, cell, addresses, payload[:, t-1])
        state = np.where((episodes['lengths'] >= t)[:, None, None], updated, state)
        if t in checkpoints:
            trajectory[t] = state.copy()
    return state, trajectory


def reference_state(episodes):
    values = episodes['initial_values'].copy()
    pointers = episodes['initial_pointers'].copy()
    for b in range(len(values)):
        for t, address in enumerate(episodes['addresses'][b, :episodes['lengths'][b]]):
            values[b, address] = episodes['new_values'][b, t]
            pointers[b, address] = episodes['new_pointers'][b, t]
    return bits(values, pointers)


def predictions(state, episodes, flip=None):
    if flip is not None:
        state = np.bitwise_xor(state, flip.astype(np.uint8))
    values = (state[..., :8] * (1 << np.arange(8))).sum(-1)
    pointers = (state[..., 8:] * (1 << np.arange(4))).sum(-1)
    b = np.arange(len(state))
    query, other = episodes['cold'][:, 0], episodes['cold'][:, 1]
    latest = episodes['addresses'][b, episodes['lengths']-1]
    return np.stack((values[b, query], values[b, latest],
                     values[b, pointers[b, query]], (values[b, query] + values[b, other]) % 256,
                     values[b, query] ^ values[b, other]), -1)


def metrics(predicted, target):
    from .developmental_metrics import wilson_upper
    result = {}
    for j, name in enumerate(TASKS):
        correct = int((predicted[:, j] == target[:, j]).sum())
        n = len(predicted)
        bit_correct = (((predicted[:, j, None] ^ target[:, j, None]) >> np.arange(8)) & 1) == 0
        result[name] = {'n': n, 'correct': correct, 'exact': correct/n,
                        'wilson_lower': 1-wilson_upper(n-correct, n), 'wilson_upper': wilson_upper(correct, n),
                        'bit_accuracy': bit_correct.mean(0).tolist()}
    return result


def permission_metrics(gate, cell):
    accepted = cell[4 * gate + 1]
    truth = np.eye(KEYS, dtype=bool)
    tp = int(accepted[truth].sum())
    fp = int(accepted[~truth].sum())
    return {'true_positive': tp, 'false_positive': fp, 'true_acceptance': tp/KEYS,
            'false_acceptance': fp/(KEYS*(KEYS-1)), 'accepted': accepted.tolist()}


def fit_read_signs(finals, targets):
    # A sign per physical address and bit, calibrated before evaluation.
    matches = sum((state == target).sum(0) for state, target in zip(finals, targets))
    n = sum(len(s) for s in finals)
    return matches < n/2


def state_information(initial, episodes, gate, cell):
    low, low_path = rollout(np.zeros_like(initial), episodes, gate, cell)
    high, high_path = rollout(np.ones_like(initial), episodes, gate, cell)
    b = np.arange(len(initial))[:, None]
    cold = episodes['cold']
    retention = {str(t): float((low_path[t] != high_path[t])[b, cold].mean()) for t in low_path}
    return low, high, retention


def decode_initial_cold(final, low, high, episodes):
    # Only suffix-derived maps and final state enter. No initial payload enters.
    recovered = (final ^ low) * (low != high)
    byte_values = (recovered[..., :8] * (1 << np.arange(8))).sum(-1)
    return byte_values[np.arange(len(final)), episodes['cold'][:, 0]]


def numerical_control():
    result = {'update': '(old+1)/2', 'exact_difference': '2**(-steps)', 'rows': []}
    low32, high32 = np.float32(0), np.float32(1)
    low64, high64 = np.float64(0), np.float64(1)
    for t in range(1, 513):
        low32, high32 = (low32+np.float32(1))/np.float32(2), (high32+np.float32(1))/np.float32(2)
        low64, high64 = (low64+1)/2, (high64+1)/2
        if t in (1, 8, 24, 25, 53, 54, 128, 512):
            exact_difference = Fraction(1, 2**t)
            result['rows'].append({'steps': t, 'float32_equal': bool(low32 == high32),
                'float64_equal': bool(low64 == high64), 'exact_equal': exact_difference == 0,
                'exact_difference_denominator': str(exact_difference.denominator)})
    return result
