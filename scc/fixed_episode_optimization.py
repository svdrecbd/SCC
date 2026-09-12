"""Explicit finite-step optimization checks, separate from SCC evidence."""

import copy
import math
import time

import torch

from .coupling import nll
from .pilot_objectives import vector_norm
from .selective_coupling import recovered_objective


def check_deadline(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError('Declared optimization diagnostic wall limit')


@torch.no_grad()
def anchor_losses(model, batches):
    return {name: float(nll(model, dict(model.named_parameters()), batch)) for name, batch in batches.items()}


def within_anchor(candidate, reference, allowance=.05):
    return candidate.keys() == reference.keys() and all(
        math.isfinite(v) and v <= reference[name] + allowance for name, v in candidate.items())


def directional_check(model, config, data, deadline=None, epsilons=(.001, .0003, .0001)):
    origin = {name: p.detach().clone() for name, p in model.named_parameters()}
    check_deadline(deadline)
    value, details = recovered_objective(model, config, data)
    gradients = torch.autograd.grad(value, tuple(model.parameters()))
    norm = float(vector_norm(gradients))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('Nonfinite or zero diagnostic derivative')
    reference = float(value.detach())
    checks = []
    try:
        for epsilon in epsilons:
            outcomes, branches = [], []
            for sign in (-1, 1):
                check_deadline(deadline)
                with torch.no_grad():
                    for (name, p), g in zip(model.named_parameters(), gradients):
                        p.copy_(origin[name] + sign * epsilon * g / norm)
                v, d = recovered_objective(model, config, data, create_graph=False)
                outcomes.append(float(v.detach()))
                branches.append({'endpoint': d['selected_endpoint'], 'reader': d['selected_branch']['reader'],
                                 'stop_rule': d['selected_branch']['stop_rule']})
            numerical = (outcomes[1] - outcomes[0]) / (2 * epsilon)
            checks.append({'epsilon_l2': epsilon, 'numerical': numerical,
                           'relative_error': abs(numerical - norm) / norm,
                           'descent': outcomes[0] < reference, 'branches': branches})
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(origin[name])
    return {'passed': all(c['relative_error'] < .1 and c['descent'] for c in checks[-2:]),
            'gradient_norm': norm, 'reference': reference, 'checks': checks,
            'details': details, 'scope': 'Local rerun check including fitted readers; coarse check is diagnostic'}


def optimize(initial, config, data, anchors, *, iterations=8,
             radii=(.003, .001, .0003, .0001), allowance=.05,
             minimum_decrease=1e-5, deadline=None, callback=None):
    """Normalized SGD with checked actual descent and a fixed per-domain anchor.

    No gradient clipping, Adam moments, stochastic episode changes, or clean
    weights as a repair resource. Copies below belong to defender construction
    and rollback; the simulated modification/repair sees only its changed state.
    """
    if iterations < 1 or not radii or any(r <= 0 for r in radii):
        raise ValueError('Positive diagnostic iterations and radii required')
    model = copy.deepcopy(initial).eval()
    reference_anchor = anchor_losses(model, anchors)
    history = []
    for iteration in range(iterations):
        check_deadline(deadline)
        value, details = recovered_objective(model, config, data)
        gradients = torch.autograd.grad(value, tuple(model.parameters()))
        norm = float(vector_norm(gradients))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('Nonfinite or zero optimization derivative')
        before = float(value.detach())
        origin = {name: p.detach().clone() for name, p in model.named_parameters()}
        record = {'iteration': iteration + 1, 'before': before, 'before_details': details,
                  'gradient_norm': norm, 'trials': [], 'accepted': False}
        try:
            for radius in radii:
                check_deadline(deadline)
                with torch.no_grad():
                    for (name, p), g in zip(model.named_parameters(), gradients):
                        p.copy_(origin[name] - radius * g / norm)
                after, after_details = recovered_objective(model, config, data, create_graph=False)
                score = float(after.detach())
                anchor = anchor_losses(model, anchors)
                admissible = math.isfinite(score) and score <= before - minimum_decrease and within_anchor(anchor, reference_anchor, allowance)
                record['trials'].append({'radius_l2': radius, 'after': score,
                                         'after_details': after_details, 'anchor_nll': anchor,
                                         'admissible': admissible})
                if admissible:
                    record.update(accepted=True, accepted_radius_l2=radius, after=score)
                    break
        finally:
            if not record['accepted']:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        p.copy_(origin[name])
        history.append(record)
        if callback:
            callback(record)
        if not record['accepted']:
            break
    return model, {'reference_anchor_nll': reference_anchor, 'anchor_allowance_nll': allowance,
                   'iterations_attempted': len(history), 'accepted_steps': sum(r['accepted'] for r in history),
                   'history': history, 'scope': 'Open fixed-training-episode optimization diagnostic, not mechanism evidence'}
