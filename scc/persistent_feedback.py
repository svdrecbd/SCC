"""Output feedback for the existing smooth live matrix; no SCC success claim.

Only the control wiring changes. The initial matrix remains the sole learned
runtime substrate. Fixed wiring is an explicit part of the editable graph boundary.
"""
import math

import torch

from scc.persistent_matrix import MatrixConfig, _shapes, matrix_step, smooth_replacement
from scc.persistent_tasks import TOKENS_PER_REQUEST


def feedback_wiring(config, *, seed=130913, device='cpu', dtype=torch.float32):
    if config.rule != 'soft':
        raise ValueError('Feedback experiment is smooth only')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    wiring = torch.randn(2*config.input_size+1, config.output_size,
                         generator=generator, dtype=torch.float64) / math.sqrt(config.output_size)
    for start in (0, config.input_size):
        block = wiring[start:start+config.input_size]
        block.sub_(block.mean(0, keepdim=True))
    return wiring.to(device=device, dtype=dtype)


def feedback_step(state, inputs, config, wiring, *, strength=1.):
    _shapes(state, inputs, config)
    if (config.rule != 'soft' or wiring.shape != (2*config.input_size+1, config.output_size)
            or wiring.device != state.device or wiring.dtype != state.dtype
            or not math.isfinite(strength)):
        raise ValueError('Invalid smooth feedback configuration')
    if strength == 0:
        return matrix_step(state, inputs, config)
    probabilities = inputs.softmax(-1)
    signals = torch.bmm(state, probabilities.unsqueeze(-1)).squeeze(-1)
    controls = signals[:, config.output_size:] + strength * (
        signals[:, :config.output_size] @ wiring.T)
    key, query, rate = torch.split(controls, [config.input_size, config.input_size, 1], dim=-1)
    rate = rate.squeeze(-1)
    changed = smooth_replacement(state, key, query, rate)
    output = torch.bmm(changed[:, :config.output_size], probabilities.unsqueeze(-1)).squeeze(-1)
    return output, changed, {'keys': key.argmax(-1), 'queries': query.argmax(-1),
                            'enabled': rate > 0, 'beta': rate.sigmoid(),
                            'key_probabilities': key.softmax(-1),
                            'query_probabilities': query.softmax(-1)}


def feedback_window(initial, token_ids, config, code, wiring, *, strength=1.):
    if token_ids.ndim != 3 or token_ids.shape[-1] != TOKENS_PER_REQUEST:
        raise ValueError('Expected batch x requests x 19 tokens')
    batch, requests, ticks = token_ids.shape
    state = initial[None].expand(batch, -1, -1) if initial.ndim == 2 else initial
    if state.shape[0] != batch:
        raise ValueError('Mismatched live state count')
    encoded = code[token_ids]
    outputs = []
    enabled = torch.zeros((), device=state.device, dtype=torch.long)
    changed_count = torch.zeros_like(enabled)
    for request in range(requests):
        for tick in range(ticks):
            output, state, control = feedback_step(state, encoded[:, request, tick], config,
                                                    wiring, strength=strength)
            enabled += control['enabled'].sum()
            changed_count += (control['enabled'] & (control['keys'] != control['queries'])).sum()
        outputs.append(output)
    return torch.stack(outputs, 1), state, {'enabled_count': enabled,
        'distinct_address_count': changed_count, 'ticks_per_stream': requests*ticks}


class LiveFeedbackMatrix:
    """Current learned weights and fixed graph wiring; no clean restore template."""
    __slots__ = ('weights', 'config', 'wiring', 'strength', 'steps')

    def __init__(self, weights, config, wiring, *, strength=1.):
        if weights.ndim != 2 or weights.shape != (config.rows, config.input_size):
            raise ValueError('Invalid live feedback state')
        if not torch.isfinite(weights).all() or not torch.isfinite(wiring).all():
            raise ValueError('Finite weights and wiring required')
        self.weights, self.wiring = weights.detach().clone(), wiring.detach().clone()
        self.config, self.strength, self.steps = config, strength, 0

    @torch.no_grad()
    def tick(self, inputs):
        if not torch.isfinite(inputs).all():
            raise ValueError('Finite inputs required')
        output, changed, controls = feedback_step(self.weights[None], inputs[None], self.config,
                                                 self.wiring, strength=self.strength)
        if not torch.isfinite(output).all() or not torch.isfinite(changed).all():
            raise ValueError('Nonfinite transition rejected before commit')
        self.weights = changed[0].detach()
        self.steps += 1
        return output[0], controls
