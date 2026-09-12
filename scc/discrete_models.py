"""Learned recurrent binary bottlenecks with explicitly approximate training gradients."""
import copy
from dataclasses import dataclass, replace
import torch
from torch.nn import functional as F
from .portfolio_models import PortfolioModel, PortfolioConfig


class HardSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

    @staticmethod
    def backward(ctx, upstream):
        value, = ctx.saved_tensors
        # Deliberately a coarse/straight-through gradient. It is not the
        # derivative of the hard forward map, including in higher-order use.
        return upstream * (1 - value.tanh().square())


@dataclass(frozen=True)
class DiscreteConfig(PortfolioConfig):
    hard: bool = True


def discrete_config(bits, hard=True, smoke=False):
    if bits not in (32,128): raise ValueError('Declared code lengths are32 and128')
    return DiscreteConfig(bottleneck_width=(4 if bits==32 else 8) if smoke else bits,
                          hard=hard, width=16 if smoke else 256, heads=2 if smoke else 4,
                          layers=3 if smoke else 4)


class DiscreteModel(PortfolioModel):
    def __init__(self, config):
        if config.variant!='standard' or config.tie_embeddings:
            raise ValueError('This construction uses the standard shared cell with a separate reader')
        super().__init__(config)
        self.cells[0].register_buffer('code_sign', torch.ones(config.bottleneck_width))

    def forward(self, token_ids, return_states=False, return_codes=False):
        if token_ids.ndim!=2 or not 0<token_ids.shape[1]<=self.config.context_length:
            raise ValueError('Invalid token sequence')
        positions=torch.arange(token_ids.shape[1],device=token_ids.device)
        state=self.tokens(token_ids)+self.positions(positions)
        states,codes=[],[]
        cell=self.cells[0]
        for _ in range(self.config.layers):
            proposed=cell.computation(state)
            if self.config.bypass_bottleneck:
                state=proposed
            else:
                values=cell.compress(cell.input_norm(proposed))
                code=(HardSign.apply(values) if self.config.hard else values.tanh())*cell.code_sign
                state=cell.state_norm(cell.expand(code))
                codes.append(code)
            states.append(state)
        logits=self.output(self.norm(state))
        if return_codes:
            if self.config.bypass_bottleneck: raise ValueError('Bypass has no bottleneck codes')
            return logits,codes
        return (logits,states) if return_states else logits

    def with_soft(self):
        result=copy.deepcopy(self)
        result.config=replace(self.config,hard=False)
        return result

    def recoded(self, signs):
        if signs.shape!=self.cells[0].code_sign.shape or not ((signs==1)|(signs==-1)).all():
            raise ValueError('One signed relabeling per code coordinate required')
        result=copy.deepcopy(self)
        signs=signs.to(result.cells[0].code_sign)
        with torch.no_grad():
            result.cells[0].code_sign.mul_(signs)
            result.cells[0].expand.weight.mul_(signs[None,:])
        return result


def from_checkpoint(state):
    config=state['configuration']
    if config['architecture']!='scc-discrete/v1': raise ValueError('Not a discrete construction checkpoint')
    model=DiscreteModel(DiscreteConfig(**config['model']))
    model.load_state_dict(state['model'],strict=True)
    return model
