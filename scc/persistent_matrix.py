"""Self-referential live matrices and an exact column-overwrite alternative.

The smooth rule adapts Eqs. 5--8 of Irie et al. (ICML 2022), with one shared
learning rate. The overwrite rule and post-update readout are SCC experimental
changes, not claimed properties of that paper. No permission parser or clean
parameter template is retained by LiveMatrix. These are substrate mechanisms;
protection-specific destruction and useful intact cognition are not established.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class MatrixConfig:
    input_size: int
    output_size: int
    rule: str = 'copy'

    def __post_init__(self):
        if self.input_size < 2 or self.output_size < 1 or self.rule not in ('soft','copy'):
            raise ValueError('Invalid persistent matrix configuration')

    @property
    def rows(self):
        return self.output_size + 2*self.input_size + 1


def _shapes(state, inputs, config):
    if state.ndim != 3 or state.shape[1:] != (config.rows,config.input_size):
        raise ValueError('Expected [batch, output+2*input+1, input] matrix state')
    if inputs.shape != (state.shape[0],config.input_size):
        raise ValueError('Expected one input vector for each live matrix')
    if state.dtype not in (torch.float32,torch.float64) or inputs.dtype != state.dtype or inputs.device != state.device:
        raise ValueError('State and inputs require matching FP32/FP64 precision and device')


def exact_column_copy(state, keys, queries, enabled):
    """Replace each selected column by an existing column, bit for bit.

    Avoid old + (new-old), which can leave floating-point differences even
    when the mathematical expression is a complete overwrite.
    """
    batch,rows,width=state.shape
    if keys.shape != (batch,) or queries.shape != (batch,) or enabled.shape != (batch,):
        raise ValueError('Invalid copy controls')
    values=state.gather(-1,queries[:,None,None].expand(batch,rows,1))
    changed=state.scatter(-1,keys[:,None,None].expand(batch,rows,1),values)
    return torch.where(enabled[:,None,None],changed,state)


def smooth_replacement(state, key_logits, query_logits, beta_logits):
    key=key_logits.softmax(-1);query=query_logits.softmax(-1)
    difference=torch.bmm(state,(query-key).unsqueeze(-1)).squeeze(-1)
    return state + beta_logits.sigmoid()[:,None,None]*difference[:,:,None]*key[:,None,:]


class _ExactForwardSurrogateBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,exact,surrogate):
        return exact

    @staticmethod
    def backward(ctx,gradient):
        # Explicit coarse gradient through the separately computed smooth rule.
        # It is not the derivative of the actual hard forward operation.
        return None,gradient


def matrix_step(state, inputs, config, *, surrogate_backward=False):
    """One functional transition, with readout from the committed new matrix."""
    _shapes(state,inputs,config)
    probabilities=inputs.softmax(-1)
    controls=torch.bmm(state,probabilities.unsqueeze(-1)).squeeze(-1)
    _,key_logits,query_logits,beta_logits=torch.split(controls,
        [config.output_size,config.input_size,config.input_size,1],dim=-1)
    beta_logits=beta_logits.squeeze(-1)
    keys=key_logits.argmax(-1);queries=query_logits.argmax(-1);enabled=beta_logits>0
    if config.rule=='soft':
        if surrogate_backward:
            raise ValueError('A smooth transition does not need a surrogate backward')
        changed=smooth_replacement(state,key_logits,query_logits,beta_logits)
    else:
        changed=exact_column_copy(state,keys,queries,enabled)
        if surrogate_backward:
            smooth=smooth_replacement(state,key_logits,query_logits,beta_logits)
            changed=_ExactForwardSurrogateBackward.apply(changed.detach(),smooth)
    output=torch.bmm(changed[:,:config.output_size],probabilities.unsqueeze(-1)).squeeze(-1)
    return output,changed,{'keys':keys,'queries':queries,'enabled':enabled,
                          'beta':beta_logits.sigmoid(),'key_probabilities':key_logits.softmax(-1),
                          'query_probabilities':query_logits.softmax(-1)}


class LiveMatrix:
    """A running instance holds only current weights, config and a step count.

    External code can keep snapshots for experiments. Those snapshots are not
    stored here or available through a reset-to-initial-state operation.
    Arbitrary external writes are explicit interventions and can add information;
    the column-support ratchet is a property of ordinary copy-mode ticks only.
    """
    __slots__=('config','weights','steps')

    def __init__(self,weights,config,*,steps=0):
        if weights.ndim!=2 or weights.shape!=(config.rows,config.input_size):
            raise ValueError('Invalid live matrix shape')
        if weights.dtype not in (torch.float32,torch.float64) or not torch.isfinite(weights).all():
            raise ValueError('Finite FP32/FP64 weights required')
        if not isinstance(steps,int) or steps<0:
            raise ValueError('Invalid step count')
        self.config=config;self.weights=weights.detach().clone();self.steps=steps

    @torch.no_grad()
    def tick(self,inputs):
        if inputs.shape!=(self.config.input_size,) or not torch.isfinite(inputs).all():
            raise ValueError('One finite input vector is required')
        output,changed,controls=matrix_step(self.weights[None],inputs[None],self.config)
        if not torch.isfinite(changed).all() or not torch.isfinite(output).all():
            raise ValueError('Nonfinite transition rejected before commit')
        self.weights=changed[0].detach();self.steps+=1
        return output[0],controls

    def snapshot(self):
        return {'schema':'scc-live-matrix/v1','weights':self.weights.clone(),'steps':self.steps,
                'configuration':{'input_size':self.config.input_size,'output_size':self.config.output_size,'rule':self.config.rule}}

    @classmethod
    def from_snapshot(cls,saved):
        if set(saved)!={'schema','weights','steps','configuration'} or saved['schema']!='scc-live-matrix/v1':
            raise ValueError('Unexpected live-state snapshot')
        return cls(saved['weights'],MatrixConfig(**saved['configuration']),steps=saved['steps'])

    def external_write(self,weights):
        """Explicit counterfactual edit; never classified as ordinary recovery."""
        if weights.shape!=self.weights.shape or weights.dtype!=self.weights.dtype or weights.device!=self.weights.device:
            raise ValueError('External write must preserve matrix shape, precision and device')
        if not torch.isfinite(weights).all():
            raise ValueError('Nonfinite external write rejected')
        self.weights=weights.detach().clone()
