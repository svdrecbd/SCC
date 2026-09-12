"""Learned SCC construction candidates; none contains a permission parser.

All task algorithms and protection behavior must be learned. Architectural
sharing is a hypothesis, not an assertion of causal necessity. Fractional paths
are finite L1-inspired explicit discretizations, not solved implicit systems.
"""
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import ModelConfig, Block


VARIANTS = ('standard', 'narrow32', 'narrow8', 'tied', 'multiplicative',
            'associative', 'integer_memory', 'fractional06', 'fractional085')


@dataclass(frozen=True)
class PortfolioConfig(ModelConfig):
    width: int = 256
    layers: int = 4
    context_length: int = 192
    tie_embeddings: bool = False
    variant: str = 'standard'
    bottleneck_width: int = 128
    memory_patterns: int = 64
    fractional_order: float = 1.0
    time_step: float = 0.5
    bypass_bottleneck: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.variant not in VARIANTS or not 0 < self.bottleneck_width <= self.width:
            raise ValueError('Unknown variant or invalid bottleneck')
        if not 0 < self.fractional_order <= 1 or not 0 < self.time_step <= 1:
            raise ValueError('Invalid fractional discretization')
        if self.memory_patterns < 2 or self.dropout or self.position_encoding != 'learned_absolute':
            raise ValueError('Unsupported architecture setting')


def variant_config(variant, smoke=False):
    if variant not in VARIANTS:
        raise ValueError(variant)
    settings = dict(variant=variant)
    if variant.startswith('narrow'):
        settings['bottleneck_width'] = int(variant.removeprefix('narrow'))
    if variant == 'tied':
        settings['tie_embeddings'] = True
    if variant == 'multiplicative':
        settings['bottleneck_width'] = 64
    if variant in ('integer_memory', 'fractional06', 'fractional085'):
        settings.update(layers=6, fractional_order={'integer_memory': 1., 'fractional06': .6, 'fractional085': .85}[variant])
    if smoke:
        settings.update(width=16, heads=2, layers=3, bottleneck_width=4 if variant == 'narrow8' else 8, memory_patterns=8)
    return PortfolioConfig(**settings)


def l1_history_weights(order, step, *, device=None, dtype=torch.float32):
    """Weights of h_0,...,h_(step-1) in a Caputo L1 history term.

    Nonnegative weights sum to one for 0<order<=1. At order=1 only
    the most recent state has nonzero weight. No history truncation is used.
    """
    if not 0 < order <= 1 or step < 1:
        raise ValueError('Invalid L1 order or step')
    if order == 1:
        out = torch.zeros(step, device=device, dtype=dtype)
        out[-1] = 1
        return out
    k = torch.arange(step, dtype=dtype, device=device)
    a = (k + 1).pow(1 - order) - k.pow(1 - order)
    return torch.cat((a[-1:], (a[:-1] - a[1:]).flip(0)))


class PortfolioCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.variant = config.variant
        self.computation = Block(config)
        self.input_norm = nn.LayerNorm(config.width)
        self.state_norm = nn.LayerNorm(config.width)
        if config.variant == 'associative':
            self.patterns = nn.Parameter(torch.empty(config.memory_patterns, config.width))
            nn.init.orthogonal_(self.patterns)
            with torch.no_grad():
                self.patterns.mul_(math.sqrt(config.width))
            self.log_temperature = nn.Parameter(torch.zeros(()))
        else:
            self.compress = nn.Linear(config.width, config.bottleneck_width, bias=False)
            self.expand = nn.Linear(config.bottleneck_width, config.width, bias=False)
            if config.variant == 'multiplicative':
                self.modulate = nn.Linear(config.width, config.bottleneck_width, bias=False)

    def forward(self, state, bypass=False):
        proposed = self.computation(state)
        if bypass:
            return proposed
        x = self.input_norm(proposed)
        if self.variant == 'associative':
            scores = F.linear(x, self.patterns) / math.sqrt(x.shape[-1])
            # This is a bounded learned inverse temperature, not a permission gate.
            beta = .1 + 9.9 * self.log_temperature.sigmoid()
            result = (scores * beta).softmax(-1) @ self.patterns
        else:
            code = self.compress(x)
            if self.variant == 'multiplicative':
                code = torch.tanh(code) * torch.tanh(self.modulate(x))
            else:
                code = F.gelu(code)
            result = self.expand(code)
        return self.state_norm(result)


class PortfolioModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context_length, config.width)
        self.cells = nn.ModuleList([PortfolioCell(config)])
        self.norm = nn.LayerNorm(config.width)
        self.output = None if config.tie_embeddings else nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        cell = self.cells[0]
        if hasattr(cell, 'compress'):
            nn.init.orthogonal_(cell.compress.weight)
            with torch.no_grad():
                cell.expand.weight.copy_(cell.compress.weight.T)
            if hasattr(cell, 'modulate'):
                nn.init.orthogonal_(cell.modulate.weight)
        for projection in (cell.computation.attention.projection, cell.computation.mlp[2]):
            nn.init.normal_(projection.weight, std=config.initializer_std / math.sqrt(2 * config.layers))

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, token_ids, return_states=False):
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.config.context_length:
            raise ValueError('Invalid token sequence')
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        state = self.tokens(token_ids) + self.positions(positions)
        history = [state]
        fractional = self.config.variant in ('integer_memory', 'fractional06', 'fractional085')
        for step in range(1, self.config.layers + 1):
            proposed = self.cells[0](state, self.config.bypass_bottleneck)
            if fractional:
                weights = l1_history_weights(self.config.fractional_order, step, device=state.device, dtype=state.dtype)
                memory = sum(w * h for w, h in zip(weights, history, strict=True))
                coefficient = math.gamma(2 - self.config.fractional_order) * self.config.time_step ** self.config.fractional_order
                state = memory + coefficient * (proposed - state)
            else:
                state = proposed
            history.append(state)
        normed = self.norm(state)
        logits = F.linear(normed, self.tokens.weight) if self.output is None else self.output(normed)
        return (logits, history[1:]) if return_states else logits

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def with_bypass(self):
        import copy
        model = copy.deepcopy(self)
        model.config = replace(self.config, bypass_bottleneck=True)
        return model


def from_checkpoint(saved):
    if saved['configuration']['architecture'] != 'scc-portfolio/v1':
        raise ValueError('Unexpected checkpoint architecture')
    model = PortfolioModel(PortfolioConfig(**saved['configuration']['model']))
    model.load_state_dict(saved['model'], strict=True)
    return model
