"""Experimental differentiable Adam with activation recomputation; not used by defenses."""

from functools import partial
import torch
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

from scc.coupling import nll


def adam_update(model, names, batches, rate, iteration, state, epsilon=1e-8):
    count = len(names)
    parameters, moment, variance = state[:count], state[count:2*count], state[2*count:]
    with sdpa_kernel(SDPBackend.MATH):
        loss = sum(weight * nll(model, dict(zip(names, parameters)), batch) for batch, weight in batches)
        loss /= sum(weight for _,weight in batches)
        gradients = torch.autograd.grad(loss, parameters, create_graph=True)
    # Mirror clip_grad_norm_ including its stabilizer. The clamp's derivative is intentional.
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
    scale = (1. / (norm + 1e-6)).clamp(max=1.)
    gradients = tuple(g * scale for g in gradients)
    moment = tuple(m * .9 + g * .1 for m,g in zip(moment,gradients))
    variance = tuple(v * .95 + g.square() * .05 for v,g in zip(variance,gradients))
    # The tiny variance floor prevents an undefined sqrt derivative at zero.
    # Its effect and all gradients must be validated; this is a stabilized variant.
    denominator = tuple((v / (1-.95**iteration)).clamp_min(torch.finfo(v.dtype).tiny).sqrt() + epsilon for v in variance)
    parameters = tuple(p - rate * (m / (1-.9**iteration)) / d for p,m,d in zip(parameters,moment,denominator))
    return parameters + moment + variance


def differentiable_adam(model, parameters, episodes, rate, recompute=True, epsilon=1e-8):
    names = tuple(parameters)
    current = tuple(parameters.values())
    state = current + tuple(torch.zeros_like(p) for p in current) + tuple(torch.zeros_like(p) for p in current)
    for index,batches in enumerate(episodes):
        fn = partial(adam_update, model, names, batches, rate, index+1, epsilon=epsilon)
        if recompute:
            state = checkpoint(fn, state, use_reentrant=False, preserve_rng_state=False)
        else:
            state = fn(state)
    return dict(zip(names,state[:len(names)]))
