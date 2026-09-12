"""Bounds on sampled correctness, not certificates of destroyed information."""
import math

import torch
from .functional_state import call_parameters as functional_call

from .data import IGNORE
from .pilot_objectives import fitted


def correctness_bound(logits, targets, *, sequence=True, temperature=.2):
    """A correct greedy gold sequence contributes at least one, including ties.

    For text (sequence=False), this bounds teacher-forced top-1 token accuracy.
    Positive affine logit transforms leave the score invariant. This implication
    is one-way: a wrong answer can still have positive loss. Reader/stop coverage
    is finite; incorrect answers do not prove loss of information.
    """
    if logits.ndim != 3 or logits.shape[:-1] != targets.shape or logits.shape[-1] < 2:
        raise ValueError('Expected aligned batch/sequence/vocabulary logits')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    mask = targets != IGNORE
    if not mask.any(-1).all():
        raise ValueError('Every example needs supervised positions')
    z = logits[mask]; y = targets[mask]
    if not torch.isfinite(z).all():
        raise ValueError('Nonfinite supervised logits')
    margin = z.gather(-1,y[:,None]).squeeze(-1) - z.scatter(-1,y[:,None],-torch.inf).amax(-1)
    span = z.amax(-1)-z.amin(-1)
    margin = margin / torch.where(span > 0,span,torch.ones_like(span))
    values = torch.zeros_like(targets,dtype=z.dtype).masked_scatter(mask,margin)
    if sequence:
        return (1+values.masked_fill(~mask,torch.inf).amin(-1)/temperature).clamp(min=0)
    return (1+values/temperature).clamp(min=0).masked_fill(~mask,0).sum(-1)/mask.sum(-1)


def bound_branches(capabilities, targets, readers, trigger_rate=.9):
    """Same-reader product upper-bounds reliable-trigger times domain accuracy.

    If >=trigger_rate of sampled target answers are correct, trigger_bound is
    exactly one. Changing confidence without losing that behavior cannot reduce
    its multiplier. The maximum domain prevents averaging away a surviving task.
    """
    if not capabilities or not targets or not readers or not 0 < trigger_rate <= 1:
        raise ValueError('Missing data/readers or invalid trigger rate')
    task = lambda name: name.split('/')[0] in ('lookup','composition','arithmetic')
    branches=[]
    for reader in readers:
        for stop in ('eos','four_digits'):
            def labels(name,y):
                if stop=='four_digits' and task(name):
                    digits=(y>=52)&(y<62)
                    if not (digits.sum(-1)==4).all():raise ValueError('Expected four answer digits')
                    return y.masked_fill(~digits,IGNORE)
                return y
            domains={name:correctness_bound(reader.apply(z),labels(name,y),sequence=task(name)).mean()
                     for name,(z,y) in capabilities.items()}
            bound=torch.cat([correctness_bound(reader.apply(z),labels(name,y)) for name,(z,y) in targets.items()]).mean()
            trigger=(bound/trigger_rate).clamp(max=1)
            capability=torch.stack(list(domains.values())).amax()
            branches.append({'reader':reader.record(),'stop_rule':stop,'domains':domains,
                             'target_correctness_bound':bound,'trigger_surrogate':trigger,
                             'capability':capability,'penalty':trigger*capability})
    return torch.stack([b['penalty'] for b in branches]).amax(),branches


def endpoint_bound(model,parameters,data):
    readers=fitted(model,parameters,data['supports'])
    def logits(batches):
        return {name:(functional_call(model,parameters,(b.tokens,),strict=True),b.targets)
                for name,b in batches.items()}
    return bound_branches(logits(data['queries']),logits(data['target_queries']),readers)
