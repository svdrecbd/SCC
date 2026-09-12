"""Reverse differentiation through Adam by recomputing short stored segments.

All optimizer-state derivatives are propagated. No frozen displacement, identity
Jacobian, truncated backward horizon, or final-optimum approximation is used.
The specified sqrt smoothing matches differentiable_modify.adam_unroll.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from .fixed_episode_optimization import check_deadline


@dataclass(frozen=True)
class AdamSettings:
    lr: float = 1e-4
    beta1: float = .9
    beta2: float = .95
    eps: float = 1e-8
    clip: float = 1.
    smooth_square: float = 1e-30

    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in (self.lr,self.eps,self.clip,self.smooth_square)):
            raise ValueError('Positive finite Adam settings required')
        if not (0<=self.beta1<1 and 0<=self.beta2<1):raise ValueError('Invalid Adam betas')


def detached(state, *, grad=False):
    return tuple(x.detach().requires_grad_(grad) for x in state)


def step_state(state, names, editable, loss_fn, step, settings, *, create_graph):
    """State order: all parameters, first moments, second moments."""
    count=len(names);n=len(editable);p=dict(zip(names,state[:count],strict=True))
    m=state[count:count+n];v=state[count+n:]
    if len(v)!=n or step<1:raise ValueError('Invalid Adam state/step')
    with sdpa_kernel(SDPBackend.MATH):
        loss=loss_fn(p)
        gradients=torch.autograd.grad(loss,tuple(p[k] for k in editable),create_graph=create_graph)
        length=(torch.stack([g.square().sum() for g in gradients]).sum()+settings.smooth_square).sqrt()
        scale=(settings.clip/(length+1e-6)).clamp(max=1.)
        updated=dict(p);nm=[];nv=[]
        for key,g,first,second in zip(editable,gradients,m,v,strict=True):
            g=g*scale
            first=settings.beta1*first+(1-settings.beta1)*g
            second=settings.beta2*second+(1-settings.beta2)*g.square()
            a=first/(1-settings.beta1**step);b=second/(1-settings.beta2**step)
            updated[key]=p[key]-settings.lr*a/((b+settings.smooth_square).sqrt()+settings.eps)
            nm.append(first);nv.append(second)
    return tuple(updated[k] for k in names)+tuple(nm)+tuple(nv),float(loss.detach())


class AdamTape:
    """One phase with fresh moments. Tapes remain in memory, never checkpoints on disk."""
    def __init__(self, parameters, losses, editable, *, settings=None, chunk_size=4, deadline=None):
        self.names=tuple(parameters);self.editable=tuple(editable);self.losses=losses
        self.settings=settings or AdamSettings();self.chunk_size=chunk_size;self.deadline=deadline
        if not losses or chunk_size<1 or not self.editable or len(set(self.editable))!=len(self.editable):
            raise ValueError('Nonempty losses/editable parameters and positive segment required')
        if any(k not in parameters for k in self.editable):raise ValueError('Unknown editable tensor')
        initial=tuple(parameters[k].detach() for k in self.names)
        zeros=tuple(torch.zeros_like(parameters[k]) for k in self.editable)
        state=detached(initial+zeros+zeros,grad=True)
        self.checkpoints={0:detached(state)};self.loss_values=[]
        for i,loss in enumerate(losses,1):
            check_deadline(deadline)
            state,value=step_state(state,self.names,self.editable,loss,i,self.settings,create_graph=False)
            state=detached(state,grad=True);self.loss_values.append(value)
            if i%chunk_size==0 or i==len(losses):self.checkpoints[i]=detached(state)
        self.final=dict(zip(self.names,state[:len(self.names)],strict=True))
        self.recomputation_max_absolute_error=0.

    def backward(self, parameter_gradients):
        if len(parameter_gradients)!=len(self.names):raise ValueError('Wrong endpoint adjoint size')
        last=self.checkpoints[len(self.losses)]
        adjoint=tuple(g.detach() for g in parameter_gradients)+tuple(torch.zeros_like(x) for x in last[len(self.names):])
        boundaries=sorted(self.checkpoints)
        for start,end in reversed(list(zip(boundaries[:-1],boundaries[1:]))):
            check_deadline(self.deadline)
            origin=detached(self.checkpoints[start],grad=True);current=origin
            for i in range(start,end):
                check_deadline(self.deadline)
                current,_=step_state(current,self.names,self.editable,self.losses[i],i+1,self.settings,create_graph=True)
            expected=self.checkpoints[end]
            for actual,target in zip(current,expected,strict=True):
                error=float((actual.detach()-target).abs().max())
                self.recomputation_max_absolute_error=max(error,self.recomputation_max_absolute_error)
                tolerance=(1e-10,1e-12) if actual.dtype==torch.float64 else (2e-5,2e-7)
                torch.testing.assert_close(actual.detach(),target,rtol=tolerance[0],atol=tolerance[1])
            gradients=torch.autograd.grad(current,origin,grad_outputs=adjoint,allow_unused=True)
            adjoint=tuple(torch.zeros_like(x) if g is None else g.detach() for x,g in zip(origin,gradients,strict=True))
            if not all(torch.isfinite(g).all() for g in adjoint):raise ValueError('Nonfinite adjoint')
            del gradients,current,origin
        return adjoint[:len(self.names)]

    def record(self):
        return {'steps':len(self.losses),'chunk_size':self.chunk_size,'saved_boundaries':len(self.checkpoints),
                'retained_state_bytes':sum(x.numel()*x.element_size() for state in self.checkpoints.values() for x in state),
                'recomputation_max_absolute_error':self.recomputation_max_absolute_error,
                'gradient_scope':'Full discrete adjoint through all parameters and both Adam moments; no backward truncation'}
