"""A learned global coordinate vector generates every dense model weight.

The affine map is a fixed structural assumption. Materializing dense weights is
an exact, inexpensive graph change and is required in this construction's tests.
"""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.func import functional_call

from .portfolio_models import PortfolioConfig, PortfolioModel


DIMENSIONS = (8192, 32768, 131072)


@dataclass(frozen=True)
class CoordinateConfig(PortfolioConfig):
    coordinate_dimension: int = 32768
    map_seed: int = 917263

    def __post_init__(self):
        super().__post_init__()
        if self.variant != 'standard' or self.tie_embeddings or self.coordinate_dimension < 2:
            raise ValueError('Coordinates currently generate the untied standard shared cell')


def coordinate_config(dimension, smoke=False):
    settings = dict(coordinate_dimension=dimension)
    if smoke:
        settings.update(width=16, heads=2, layers=3, bottleneck_width=8,
                        coordinate_dimension=min(dimension, 64))
    return CoordinateConfig(**settings)


class CoordinateModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        settings = asdict(config)
        self.base_config = PortfolioConfig(**{k: v for k, v in settings.items()
                                             if k in PortfolioConfig.__dataclass_fields__})
        self.template = PortfolioModel(self.base_config)
        self.shapes = tuple((n, tuple(p.shape), p.numel()) for n, p in self.template.named_parameters())
        total = sum(size for _, _, size in self.shapes)
        if config.coordinate_dimension > total:
            raise ValueError('Coordinate dimension must not exceed the expanded parameter count')
        # Preserve the random initial dense weights as explicit fixed buffers.
        # There is no clean trained checkpoint or semantic controller in this map.
        for module in self.template.modules():
            for name, parameter in list(module.named_parameters(recurse=False)):
                del module._parameters[name]
                module.register_buffer(name, parameter.detach().clone())
        generator = torch.Generator().manual_seed(config.map_seed)
        first = torch.randint(config.coordinate_dimension, (total,), generator=generator)
        offset = torch.randint(1, config.coordinate_dimension, (total,), generator=generator)
        second = (first + offset) % config.coordinate_dimension
        self.register_buffer('indices', torch.stack((first, second)))
        self.register_buffer('signs', (2 * torch.randint(2, (2, total), generator=generator) - 1).to(torch.int8))
        self.cells = nn.ParameterDict({'coordinates': nn.Parameter(torch.zeros(config.coordinate_dimension))})
        assert not list(self.template.parameters())

    def expanded_parameters(self):
        coordinates = self.cells['coordinates']
        displacement = (coordinates[self.indices] * self.signs).sum(0) / math.sqrt(2)
        baseline = dict(self.template.named_buffers())
        result, offset = {}, 0
        for name, shape, count in self.shapes:
            result[name] = baseline[name] + displacement[offset:offset + count].view(shape)
            offset += count
        return result

    def forward(self, token_ids, return_states=False):
        return functional_call(self.template, self.expanded_parameters(), (token_ids,),
                               {'return_states': return_states}, strict=True)

    def parameter_count(self):
        return self.cells['coordinates'].numel()

    def map_record(self):
        hits = torch.bincount(self.indices.flatten(), minlength=self.parameter_count())
        return {'coordinate_dimension': self.parameter_count(),
                'expanded_parameters': sum(size for _, _, size in self.shapes),
                'used_coordinates': int(hits.count_nonzero()),
                'minimum_assignments': int(hits.min()), 'maximum_assignments': int(hits.max()),
                'buffers_bytes': sum(b.numel() * b.element_size() for b in self.buffers()),
                'map_seed': self.config.map_seed, 'terms_per_weight': 2,
                'map': 'theta = theta_initial + (sign1*z[index1] + sign2*z[index2])/sqrt(2)',
                'edit_scope': 'All learned coordinates; initialized weights and index/sign map fixed',
                'storage_compression_claimed': False,
                'materialization_available': True}

    def materialize(self):
        # Constructing the equivalent module must not advance the task RNG.
        with torch.random.fork_rng(devices=[]):
            result = PortfolioModel(self.base_config)
        coordinates = self.cells['coordinates']
        result.to(device=coordinates.device, dtype=coordinates.dtype)
        result.load_state_dict({n: p.detach().clone() for n, p in self.expanded_parameters().items()}, strict=True)
        result.train(self.training)
        return result


def from_checkpoint(saved):
    architecture = saved['configuration']['architecture']
    if architecture == 'scc-coordinates/v1':
        model = CoordinateModel(CoordinateConfig(**saved['configuration']['model']))
    elif architecture == 'scc-portfolio/v1':
        model = PortfolioModel(PortfolioConfig(**saved['configuration']['model']))
    else:
        raise ValueError('Unexpected coordinate or materialized architecture')
    # Constructors initialize float32; convert floating state before loading FP64 fixtures.
    first = next(value for value in saved['model'].values() if value.is_floating_point())
    model.to(dtype=first.dtype)
    model.load_state_dict(saved['model'], strict=True)
    return model
