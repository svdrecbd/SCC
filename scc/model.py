"""Conventional pre-norm, dense, causal Transformer initialized from scratch."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 260
    context_length: int = 256
    layers: int = 2
    width: int = 64
    heads: int = 4
    dropout: float = 0.0
    position_encoding: str = "learned_absolute"
    tie_embeddings: bool = True
    local_attention_window: int | None = None
    initializer_std: float = 0.02

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.layers, self.width, self.heads) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.position_encoding not in ("learned_absolute", "rotary", "alibi"):
            raise ValueError("Unknown position encoding")
        if self.position_encoding == "rotary" and (self.width // self.heads) % 2:
            raise ValueError("Rotary attention needs an even head dimension")
        if self.local_attention_window is not None and not 1 <= self.local_attention_window <= self.context_length:
            raise ValueError("Local attention window must fit the context")
        if not 0 < self.initializer_std <= 1:
            raise ValueError("Initializer standard deviation must be in (0, 1]")


def rotary(tensor):
    """Interleaved RoPE on [batch, heads, sequence, head_dimension]."""
    length, width = tensor.shape[-2:]
    frequency = 1.0 / (10000 ** (torch.arange(0, width, 2, device=tensor.device, dtype=torch.float32) / width))
    angles = torch.arange(length, device=tensor.device, dtype=torch.float32)[:, None] * frequency[None, :]
    cosine, sine = angles.cos().to(tensor.dtype), angles.sin().to(tensor.dtype)
    even, odd = tensor[..., 0::2], tensor[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config, layer_index=0):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.rotary = config.position_encoding == "rotary"
        self.alibi = config.position_encoding == "alibi"
        self.local_window = config.local_attention_window if layer_index == 0 else None
        if self.alibi and self.heads & (self.heads - 1):
            raise ValueError("This ALiBi implementation uses power-of-two head counts")
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.rotary:
            query, key = rotary(query), rotary(key)
        mask = None
        if self.alibi:
            positions = torch.arange(length, device=x.device)
            distance = positions[:, None] - positions[None, :]
            slopes = 2.0 ** (-8 * torch.arange(1, self.heads + 1, device=x.device).float() / self.heads)
            mask = (-slopes[:, None, None] * distance[None, :, :]).to(query.dtype)
            mask = mask.masked_fill(distance[None, :, :] < 0, float("-inf"))
        if self.local_window is not None:
            positions = torch.arange(length, device=x.device)
            distance = positions[:, None] - positions[None, :]
            allowed = (distance >= 0) & (distance < self.local_window)
            mask = allowed if mask is None else mask.masked_fill(~allowed, float("-inf"))
        attention = F.scaled_dot_product_attention(
            query, key, value, is_causal=mask is None, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.projection(attention.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config, layer_index=0):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.width)
        self.attention = Attention(config, layer_index)
        self.norm_mlp = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
                                 nn.Linear(4 * config.width, config.width))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm_attention(x)))
        return x + self.dropout(self.mlp(self.norm_mlp(x)))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context_length, config.width) if config.position_encoding == "learned_absolute" else None
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, index) for index in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.output = None if config.tie_embeddings else nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        for block in self.blocks:
            # Scale residual projections with depth; output weights are tied.
            nn.init.normal_(block.attention.projection.weight, std=config.initializer_std / math.sqrt(2 * config.layers))
            nn.init.normal_(block.mlp[2].weight, std=config.initializer_std / math.sqrt(2 * config.layers))

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.config.context_length:
            raise ValueError("Expected [batch, length] tokens within the configured context")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.tokens(token_ids)
        if self.positions is not None:
            hidden = hidden + self.positions(positions)
        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return F.linear(self.norm(hidden), self.tokens.weight) if self.output is None else self.output(self.norm(hidden))

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
