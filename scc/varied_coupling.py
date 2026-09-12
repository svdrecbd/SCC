"""Checked minibatch descent on fresh selective-edit episodes.

All sampling/step decisions use training data. Reserved episode scores and
validation behavior are reported separately and never choose an update.
"""

import copy
import math
import time

import torch

from .fixed_episode_optimization import anchor_losses, check_deadline, within_anchor
from .pilot_objectives import vector_norm
from .selective_coupling import recovered_objective


def pool_score(model, config, episodes, *, gradient=False, deadline=None):
    if not episodes:
        raise ValueError('At least one episode required')
    records, accumulated = [], None
    for data in episodes:
        check_deadline(deadline)
        value, details = recovered_objective(model, config, data, create_graph=gradient)
        records.append({'objective': float(value.detach()), 'details': details})
        if gradient:
            gradients = torch.autograd.grad(value, tuple(model.parameters()))
            if accumulated is None:
                accumulated = [g.detach() / len(episodes) for g in gradients]
            else:
                for total, g in zip(accumulated, gradients, strict=True):
                    total.add_(g.detach(), alpha=1 / len(episodes))
            del gradients
        del value
    score = sum(r['objective'] for r in records) / len(records)
    return score, records, accumulated


def optimize_varied(initial, config, episode_factory, fixed_anchors, anchor_factory, *,
                    iterations=128, radii=(.03, .01, .003, .001, .0003, .0001),
                    allowance=.05, minimum_decrease=1e-5, deadline=None, callback=None,
                    trial_guard=None, enforce_guard=True):
    """Each step fits a fresh episode minibatch with an actual rerun line search.

    The fixed and fresh per-domain anchors compare to the original defender,
    never the preceding step. Fresh episodes may overlap each other but exclude
    the pre-reserved monitor pool. No outer Adam/clipping or approximate unroll.
    Rejected steps roll back exactly and consume their declared opportunity.
    A wall-limit interruption rolls back the incomplete step and is reported.
    """
    if iterations < 1 or not radii or any(r <= 0 for r in radii):
        raise ValueError('Positive iteration count and radii required')
    model = copy.deepcopy(initial).eval()
    fixed_reference = anchor_losses(initial, fixed_anchors)
    history, stop_reason = [], 'declared_iteration_count'
    started = time.monotonic()
    for iteration in range(iterations):
        record = {'iteration': iteration + 1, 'trials': [], 'accepted': False}
        origin = {name: p.detach().clone() for name, p in model.named_parameters()}
        try:
            check_deadline(deadline)
            episodes = episode_factory(iteration)
            fresh_anchors, anchor_record = anchor_factory(iteration)
            fresh_reference = anchor_losses(initial, fresh_anchors)
            before, before_details, gradients = pool_score(model, config, episodes, gradient=True, deadline=deadline)
            norm = float(vector_norm(gradients))
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError('Nonfinite or zero varied-coupling gradient')
            group_norms = {group: math.sqrt(sum(float(g.square().sum()) for (n, _), g in
                           zip(model.named_parameters(), gradients, strict=True) if n.startswith(prefix)))
                           for group, prefix in [('core', 'cells.'), ('output', 'output.'),
                                                 ('final_norm', 'norm.'), ('tokens', 'tokens.'),
                                                 ('positions', 'positions.')]}
            record.update(before=before, before_details=before_details, gradient_norm=norm,
                          gradient_group_norms=group_norms,
                          episodes=[d['record'] for d in episodes], fresh_anchors=anchor_record,
                          fresh_reference_nll=fresh_reference)
            for radius in radii:
                check_deadline(deadline)
                with torch.no_grad():
                    for (name, p), g in zip(model.named_parameters(), gradients, strict=True):
                        p.copy_(origin[name] - radius * g / norm)
                score, details, _ = pool_score(model, config, episodes, deadline=deadline)
                fixed = anchor_losses(model, fixed_anchors)
                fresh = anchor_losses(model, fresh_anchors)
                admissible = (math.isfinite(score) and score <= before - minimum_decrease
                              and within_anchor(fixed, fixed_reference, allowance)
                              and within_anchor(fresh, fresh_reference, allowance))
                base_admissible = admissible
                guard = None
                if admissible and trial_guard is not None:
                    check_deadline(deadline)
                    guard = trial_guard(model, iteration, episodes, radius)
                    if not isinstance(guard.get('passed'), bool):
                        raise ValueError('Trial guard must report a Boolean passed value')
                    check_deadline(deadline)
                    admissible = admissible and (guard['passed'] or not enforce_guard)
                record['trials'].append({'radius_l2': radius, 'after': score, 'after_details': details,
                                         'fixed_anchor_nll': fixed, 'fresh_anchor_nll': fresh,
                                         'base_admissible':base_admissible,'behavior_guard':guard,
                                         'guard_enforced':trial_guard is not None and enforce_guard,
                                         'admissible': admissible})
                if admissible:
                    record.update(accepted=True, after=score, accepted_radius_l2=radius)
                    break
            del gradients, episodes, fresh_anchors
        except TimeoutError:
            record['interrupted'] = 'training_wall_limit'
            stop_reason = 'training_wall_limit'
        finally:
            if not record['accepted']:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        p.copy_(origin[name])
        record['elapsed_seconds'] = time.monotonic() - started
        history.append(record)
        if callback:
            callback(model, record)
        if 'interrupted' in record:
            break
    return model, {'requested_iterations': iterations, 'iterations_attempted': len(history),
                   'completed_iterations': sum('interrupted' not in r for r in history),
                   'accepted_steps': sum(r['accepted'] for r in history), 'stop_reason': stop_reason,
                   'reference_fixed_anchor_nll': fixed_reference, 'anchor_allowance_nll': allowance,
                   'elapsed_seconds': time.monotonic() - started, 'history': history,
                   'scope': 'Open varied-episode continuation; not developmental timing or SCC evidence'}
