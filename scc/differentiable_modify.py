"""Differentiate through the specified smooth Adam modification, not an identity Jacobian."""

import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .coupling import nll
from .developmental_run import Streams, TEXT_SOURCES
from .developmental_tasks import FAMILIES


def adam_unroll(parameters, losses, lr=.001, betas=(.9,.95), eps=1e-8,
                clip=1., smooth_square=1e-30, create_graph=True, editable=None, trace=None):
    """Fresh moments, no weight decay, differentiable global gradient clipping.

    sqrt(v + 1e-30) prevents an undefined derivative at exactly zero variance.
    This tiny smoothing makes the update a specified variant of stock AdamW;
    numerical agreement with stock Adam is tested, not assumed exact bitwise.
    `editable` restricts the inner optimizer only. Unedited tensors retain their
    original autograd connection, including their influence on edited gradients.
    """
    if not losses or lr<=0 or eps<=0 or clip<=0 or smooth_square<=0:
        raise ValueError("Invalid differentiable modification settings")
    current=dict(parameters)
    names=tuple(current) if editable is None else tuple(editable)
    if not names or len(set(names))!=len(names) or any(k not in current for k in names):
        raise ValueError("Editable names must be a nonempty unique parameter subset")
    moments={k:torch.zeros_like(current[k]) for k in names}
    variances={k:torch.zeros_like(current[k]) for k in names}
    with sdpa_kernel(SDPBackend.MATH):
        for step,loss_fn in enumerate(losses,1):
            loss=loss_fn(current)
            gradients=torch.autograd.grad(loss,tuple(current[k] for k in names),create_graph=create_graph)
            length=(torch.stack([g.square().sum() for g in gradients]).sum()+smooth_square).sqrt()
            coefficient=(clip/(length+1e-6)).clamp(max=1.)
            if trace is not None:
                trace.append({'step':step,'loss':float(loss.detach()),'gradient_norm':float(length.detach()),
                              'clip_coefficient':float(coefficient.detach())})
            updated=dict(current)
            for key,gradient in zip(names,gradients):
                value=current[key]
                gradient=gradient*coefficient
                moments[key]=betas[0]*moments[key]+(1-betas[0])*gradient
                variances[key]=betas[1]*variances[key]+(1-betas[1])*gradient.square()
                first=moments[key]/(1-betas[0]**step)
                second=variances[key]/(1-betas[1]**step)
                updated[key]=value-lr*first/((second+smooth_square).sqrt()+eps)
            current=updated
    return current


def modification_batches(bank,config,ordinal,device):
    stream=Streams(bank,config["data_seed"]+100000+ordinal,device,config["meta_batch_size"],config["reordered_probability"])
    episodes=[]
    for _ in range(config["inner_steps"]):
        removal=stream.task(stream.rng.choice(FAMILIES),"unauthorized",disclose=True,role="inner/disclose")
        replay,floor=None,None
        if config["inner_replay"]:
            domain=stream.rng.choice(FAMILIES+TEXT_SOURCES)
            floor=math.log(10) if domain in FAMILIES else bank.floors[domain]
            replay=stream.capability(domain)
        episodes.append((removal,replay,floor))
    return episodes,stream


def sgd_unroll(parameters,losses,lr,clip=1.,create_graph=True):
    if not losses or lr<=0 or clip<=0:
        raise ValueError("Invalid differentiable SGD settings")
    current=dict(parameters)
    with sdpa_kernel(SDPBackend.MATH):
        for objective in losses:
            gradients=torch.autograd.grad(objective(current),tuple(current.values()),create_graph=create_graph)
            length=(torch.stack([g.square().sum() for g in gradients]).sum()+1e-30).sqrt()
            coefficient=(clip/(length+1e-6)).clamp(max=1.)
            current={key:value-lr*coefficient*gradient for (key,value),gradient in zip(current.items(),gradients)}
    return current


def differentiable_modification(model,bank,config,ordinal,device,create_graph=True):
    episodes,stream=modification_batches(bank,config,ordinal,device)
    functions=[]
    for removal,replay,floor in episodes:
        def objective(parameters,removal=removal,replay=replay,floor=floor):
            value=nll(model,parameters,removal)
            if replay is not None:
                value=(value+config["inner_replay"]*nll(model,parameters,replay)/floor)/(1+config["inner_replay"])
            return value
        functions.append(objective)
    optimizer=config.get("inner_optimizer","adam")
    if optimizer not in ("adam","sgd"):
        raise ValueError("Unknown differentiable modification optimizer")
    unroll=adam_unroll if optimizer=="adam" else sgd_unroll
    settings={"eps":config.get("inner_epsilon",1e-8)} if optimizer=="adam" else {}
    return unroll(dict(model.named_parameters()),functions,lr=config["inner_lr"],create_graph=create_graph,**settings),stream
