"""Learned recurrent computation with a mandatory per-pass state bottleneck.

This architecture contains no permission parser or task algorithm. Sharing and
the absence of an intact residual around the bottleneck are design choices,
not a certificate that protection is indispensable to cognition.
"""

import copy
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import Block, ModelConfig, Transformer


@dataclass(frozen=True)
class BottleneckConfig(ModelConfig):
    width: int = 256
    layers: int = 4
    context_length: int = 192
    tie_embeddings: bool = False
    bottleneck_width: int = 128
    shared_core: bool = True
    bypass_bottleneck: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.bottleneck_width <= self.width:
            raise ValueError('Bottleneck width must be within hidden width')
        if self.position_encoding != 'learned_absolute' or self.local_attention_window is not None:
            raise ValueError('This candidate uses full causal attention and absolute positions')
        if self.tie_embeddings or self.dropout:
            raise ValueError('This candidate uses a separately editable output reader and no dropout')


class StateCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.computation = Block(config)
        self.input_norm = nn.LayerNorm(config.width)
        self.compress = nn.Linear(config.width, config.bottleneck_width, bias=False)
        self.expand = nn.Linear(config.bottleneck_width, config.width, bias=False)
        self.state_norm = nn.LayerNorm(config.width)

    def forward(self, state, bypass=False):
        proposed = self.computation(state)
        if bypass:
            # Explicit graph-edit challenge: replace the bottleneck by identity.
            return proposed
        narrow = F.gelu(self.compress(self.input_norm(proposed)))
        return self.state_norm(self.expand(narrow))


class LearnedBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context_length, config.width)
        self.cells = nn.ModuleList([StateCell(config) for _ in range(1 if config.shared_core else config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        for cell in self.cells:
            # An orthogonal initial projection avoids a chain of tiny products;
            # both factors remain learned and editable from initialization.
            nn.init.orthogonal_(cell.compress.weight)
            with torch.no_grad():
                cell.expand.weight.copy_(cell.compress.weight.T)
            for projection in (cell.computation.attention.projection, cell.computation.mlp[2]):
                nn.init.normal_(projection.weight, std=config.initializer_std / math.sqrt(2 * config.layers))

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.config.context_length:
            raise ValueError('Expected batch/sequence tokens within the configured context')
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        state = self.tokens(token_ids) + self.positions(positions)
        for index in range(self.config.layers):
            cell = self.cells[0 if self.config.shared_core else index]
            state = cell(state, self.config.bypass_bottleneck)
        return self.output(self.norm(state))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def with_bypass(self):
        changed = copy.deepcopy(self)
        changed.config = replace(self.config, bypass_bottleneck=True)
        return changed


def editable_names(model, scope):
    if scope not in ('all', 'core', 'reader'):
        raise ValueError('Unknown edit scope')
    names = [name for name, _ in model.named_parameters()
             if scope == 'all' or (scope == 'core' and name.startswith('cells.'))
             or (scope == 'reader' and name.startswith(('norm.', 'output.')))]
    if not names:
        raise ValueError('Requested scope is absent from this architecture')
    return names


def from_checkpoint(state):
    configuration = state.get('configuration') or state['contract']['configuration']
    architecture = configuration.get('architecture', 'transformer')
    if architecture == 'learned-bottleneck/v1':
        model = LearnedBottleneck(BottleneckConfig(**configuration['model']))
    elif architecture == 'transformer':
        model = Transformer(ModelConfig(**configuration['model']))
    else:
        raise ValueError('Unknown checkpoint architecture')
    model.load_state_dict(state['model'], strict=True)
    return model
