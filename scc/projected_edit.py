"""Explicit capability-constrained local directions for SCC construction.

Finite Jacobian constraints and a positive ridge define a soft projection.
They do not certify preserved behavior after a finite edit, much less SCC.
"""
import math

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .coupling import nll
from .data import IGNORE
from .developmental_tasks import FAMILIES
from .functional_state import call_parameters
from .learned_bottleneck import editable_names


def normalized_margins(logits, targets):
    """One worst gold-token rank margin per example, invariant to positive affine logits."""
    if logits.shape[:-1] != targets.shape or logits.ndim != 3:
        raise ValueError('Expected aligned batch/sequence/vocabulary scores')
    mask = targets != IGNORE
    if not mask.any(-1).all():
        raise ValueError('Every example needs supervised positions')
    z, y = logits[mask], targets[mask]
    gold = z.gather(-1, y[:, None]).squeeze(-1)
    alternative = z.scatter(-1, y[:, None], -torch.inf).amax(-1)
    span = z.amax(-1) - z.amin(-1)
    values = (gold - alternative) / torch.where(span > 0, span, torch.ones_like(span))
    return torch.zeros_like(targets, dtype=z.dtype).masked_scatter(mask, values).masked_fill(~mask, torch.inf).amin(-1)


def observables(logits, batch, domain):
    if domain.split('/')[0] in FAMILIES or domain == 'other_refusal':
        return normalized_margins(logits, batch.targets)
    mask = batch.targets != IGNORE
    loss = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=IGNORE,
                           reduction='none').view_as(batch.targets)
    return -(loss * mask).sum(-1) / mask.sum(-1)


def soft_projection(gradient, jacobian, ridge=.001):
    """Remove directions measured by unit Jacobian rows, with declared ridge.

    d = g - A.T solve(A A.T + ridge I, A g), A_i = J_i / ||J_i||.
    This is differentiable through g and J. Zero rows remain zero. A positive
    ridge avoids differentiating a rank-thresholded pseudoinverse but means
    constraints are approximate, not exact null-space membership.
    """
    if gradient.ndim != 1 or jacobian.ndim != 2 or jacobian.shape[1] != len(gradient) or not len(jacobian):
        raise ValueError('Expected a gradient vector and nonempty matching Jacobian')
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('Positive finite ridge required')
    if not torch.isfinite(gradient).all() or not torch.isfinite(jacobian).all():
        raise ValueError('Nonfinite local derivatives')
    norms = torch.linalg.vector_norm(jacobian, dim=1, keepdim=True)
    rows = jacobian / torch.where(norms > 0, norms, torch.ones_like(norms))
    gram = rows @ rows.T
    coefficients = torch.linalg.solve(gram + ridge * torch.eye(len(rows), device=rows.device, dtype=rows.dtype), rows @ gradient)
    return gradient - rows.T @ coefficients, rows, gram


def local_geometry(model, parameters, target, capabilities, *, scope='core', ridge=.001, create_graph=False):
    """Differentiate selected-answer NLL and each declared capability observation."""
    names = editable_names(model, scope)
    tensors = tuple(parameters[n] for n in names)
    with sdpa_kernel(SDPBackend.MATH):
        attack_loss = nll(model, parameters, target)
        gradients = torch.autograd.grad(attack_loss, tensors, create_graph=create_graph)
        gradient = torch.cat([g.flatten() for g in gradients])
        vectors, labels, scores = [], [], []
        for domain, batch in capabilities.items():
            values = observables(call_parameters(model, parameters, (batch.tokens,), strict=True), batch, domain)
            for index, value in enumerate(values):
                grads = torch.autograd.grad(value, tensors, create_graph=create_graph,
                                            retain_graph=create_graph or index + 1 < len(values))
                vectors.append(torch.cat([g.flatten() for g in grads]))
                labels.append(f'{domain}/example-{index}')
                scores.append(value)
        jacobian = torch.stack(vectors)
        direction, rows, gram = soft_projection(gradient, jacobian, ridge)
    return {'names': names, 'gradient': gradient, 'jacobian': jacobian,
            'projected': direction, 'unit_rows': rows, 'gram': gram,
            'observable_labels': labels, 'observables': torch.stack(scores),
            'attack_loss': attack_loss, 'ridge': ridge}


def geometry_record(value):
    g, d, rows = (value[k].detach() for k in ('gradient', 'projected', 'unit_rows'))
    gram = value['gram'].detach()
    eigenvalues = torch.linalg.eigvalsh(gram.double())
    tiny = torch.finfo(g.dtype).tiny
    gnorm, dnorm = g.norm(), d.norm()
    return {'editable_parameters': len(g), 'observations': len(rows), 'ridge': value['ridge'],
            'attack_nll': float(value['attack_loss'].detach()), 'gradient_norm': float(gnorm),
            'projected_norm': float(dnorm), 'retained_gradient_norm_fraction': float(dnorm / gnorm.clamp_min(tiny)),
            'attack_descent_dot': float(g.dot(d)),
            'maximum_unit_constraint_slope_before': float((rows @ g).abs().max()),
            'maximum_unit_constraint_slope_after': float((rows @ d).abs().max()),
            'constraint_slope_norm_ratio': float((rows @ d).norm() / (rows @ g).norm().clamp_min(tiny)),
            'gram_eigenvalues': eigenvalues.cpu().tolist(),
            'gram_rank_at_relative_1e-6': int((eigenvalues > eigenvalues.max() * 1e-6).sum()),
            'observable_labels': value['observable_labels'],
            'observable_values': value['observables'].detach().cpu().tolist(),
            'scope': 'Local finite constraints, soft ridge projection; actual finite-step behavior must be checked'}


def apply_direction(parameters, names, direction, radius):
    if not math.isfinite(radius):
        raise ValueError('Finite signed radius required')
    norm = torch.linalg.vector_norm(direction)
    unit = direction / torch.where(norm > 0, norm, torch.ones_like(norm))
    out, offset = dict(parameters), 0
    for name in names:
        count = parameters[name].numel()
        out[name] = parameters[name] - radius * unit[offset:offset + count].view_as(parameters[name])
        offset += count
    if offset != len(direction):
        raise ValueError('Direction does not match the edited parameter tensors')
    return out
