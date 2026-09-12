"""Independently vary finite history and increment scale; no persistent state."""
import copy
from dataclasses import replace
import math

from .portfolio_models import variant_config


COEFFICIENT_085 = math.gamma(1.15) * .5 ** .85
CONDITIONS = ('order1-c050', 'order1-c085', 'order085-c050', 'order085-c085')


def factorial_config(condition, smoke=False):
    if condition not in CONDITIONS:
        raise ValueError('Unknown memory factorial condition')
    order = 1. if condition.startswith('order1-') else .85
    coefficient = .5 if condition.endswith('c050') else COEFFICIENT_085
    variant = 'integer_memory' if order == 1 else 'fractional085'
    original = (order == 1. and coefficient == .5) or (order == .85 and coefficient == COEFFICIENT_085)
    dt = .5 if original else (coefficient / math.gamma(2 - order)) ** (1 / order)
    config = replace(variant_config(variant, smoke), time_step=dt)
    assert abs(math.gamma(2 - order) * dt ** order - coefficient) < 1e-15
    return config


def intervention_conditions(condition):
    """Change history alone or increment alone, without changing any tensor."""
    if condition not in CONDITIONS:
        raise ValueError('Unknown memory factorial condition')
    history, coefficient = condition.split('-')
    return {'history-switch': ('order085' if history == 'order1' else 'order1') + '-' + coefficient,
            'coefficient-switch': history + '-' + ('c085' if coefficient == 'c050' else 'c050')}


def reinterpret(model, condition):
    target = factorial_config(condition, model.config.width == 16)
    if model.config.layers != target.layers or model.config.width != target.width:
        raise ValueError('Graph intervention requires the declared factorial architecture')
    modified = copy.deepcopy(model)
    modified.config = replace(model.config, variant=target.variant, fractional_order=target.fractional_order,
                              time_step=target.time_step)
    # Both variants use the same cell operations. Only the history rule changes.
    assert all(a.equal(b) for a,b in zip(model.state_dict().values(), modified.state_dict().values(), strict=True))
    return modified
