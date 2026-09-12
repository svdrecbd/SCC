"""Bounded random-direction finite differences on actual model evaluations."""
import math
import torch
from .functional_state import call_parameters
from .learned_bottleneck import editable_names
from .selective_coupling import Target,selective_batches,edit_loss
from .developmental_run import Streams,batch_fingerprint
from .provenance import digest


def central_estimate(function, center, directions, radius):
    if radius<=0 or not math.isfinite(radius): raise ValueError('Positive finite difference radius required')
    if center.ndim!=1 or directions.ndim!=2 or directions.shape[1]!=len(center): raise ValueError('Invalid direction dimensions')
    # FP32 reductions can mismeasure long, valid Rademacher vectors enough to
    # fail a unit check. Check their actual represented coordinates in FP64.
    if not torch.allclose(directions.double().norm(dim=1),torch.ones(len(directions),device=center.device,dtype=torch.float64),rtol=1e-6,atol=1e-7):
        raise ValueError('Unit directions required')
    pairs=[]
    for direction in directions:
        plus,minus=function(center+radius*direction),function(center-radius*direction)
        pairs.append((plus,minus))
    slopes=torch.stack([(plus-minus)/(2*radius) for plus,minus in pairs])
    if not torch.isfinite(slopes).all(): raise ValueError('Nonfinite function difference')
    estimate=len(center)*(slopes[:,None]*directions).mean(0)
    return estimate,pairs


class FiniteEditStepper:
    def __init__(self,model,bank,scope='core',batch_size=16,directions=4):
        if not isinstance(directions,int) or directions<1:raise ValueError('Positive number of directions required')
        self.model=model
        self.names=editable_names(model,scope)
        self.scope=scope;self.count=directions;self.step=0
        self.device=str(next(model.parameters()).device)
        self.stream=Streams(bank,298117,self.device,batch_size,.5)
        self.generator=torch.Generator(device=self.device).manual_seed(670927)
        self.target=Target()
        self.chain=digest('finite-direction-edit/v1')

    @torch.no_grad()
    def advance(self):
        model=self.model
        parameters=dict(model.named_parameters())
        center=torch.cat([parameters[n].detach().flatten() for n in self.names])
        def expand(vector):
            result=dict(parameters);offset=0
            for n in self.names:
                count=parameters[n].numel()
                result[n]=vector[offset:offset+count].view_as(parameters[n]);offset+=count
            return result
        batch=selective_batches(self.stream,self.target,self.step)
        def objective(vector):
            return edit_loss(model,expand(vector),batch,{'replay_weight':3.,'other_refusal_weight':.5})
        directions=(torch.randint(0,2,(self.count,len(center)),device=self.device,generator=self.generator)*2-1).to(center)/math.sqrt(len(center))
        radius=(.02,.1,.5)[self.step%3]
        gradient,pairs=central_estimate(objective,center,directions,radius)
        before=objective(center)
        norm=gradient.norm()
        direction=gradient/torch.where(norm>0,norm,torch.ones_like(norm))
        accepted=None;chosen=center;after=before;attempts=[]
        for size in (.05,.02,.005):
            candidate=center-size*direction
            value=objective(candidate)
            attempts.append({'radius':size,'loss':float(value)})
            if value<after:
                chosen,after,accepted=candidate,value,size
                break
        changed=expand(chosen)
        for name in self.names: parameters[name].copy_(changed[name])
        sha=digest([batch_fingerprint(b) for b in batch[:3]])
        self.chain=digest([self.chain,sha]);self.step+=1
        return {'step':self.step,'scope':self.scope,'before':float(before),'after':float(after),
                'accepted_radius':accepted,'difference_radius':radius,'estimate_norm':float(norm),
                'loss_pairs':[[float(a),float(b)] for a,b in pairs],'candidate_losses':attempts,
                'objective_evaluations':2*self.count+1+len(attempts),'batch_sha256':sha,'chain':self.chain,
                'method':'Finite random-direction estimate and actual-batch acceptance; no autograd used; not exhaustive search or an exact derivative of a discontinuous objective'}
