"""Precision/perturbation diagnostics; never relaxes a training acceptance gate."""
import math
import time

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from .fixed_episode_optimization import check_deadline
from .pilot_objectives import scalar_tree,vector_norm
from .provenance import digest
from .selective_coupling import endpoint_objective,rollout


def profiled_objective(model,config,data,create_graph=True):
    trace = {}
    with sdpa_kernel(SDPBackend.MATH):
        parameters = rollout(model,config,data,create_graph,trace=trace)
        results = [endpoint_objective(model,p,data,config['capability_reduction']) for p in parameters]
        values = torch.stack([r[0] for r in results]);selected = int(values.detach().argmax())
        value = values.amax()
    branch = max(results[selected][1],key=lambda b:float(b['penalty'].detach()))
    domains = scalar_tree(branch['domains']); largest = max(domains.values())
    active = {'endpoint':selected,'reader':branch['reader'],'stop_rule':branch['stop_rule'],
              'maximum_domains':[k for k,v in domains.items() if v==largest]}
    record = {'episode_sha256':data['record'].get('batch_sha256'), 'selected':active,
              'selected_branch':scalar_tree(branch),'endpoint_penalties':[float(v.detach()) for v in values],
              'trace':trace,'clip_pattern':{k:[s['clip_coefficient'] < 1. for s in rows] for k,rows in trace.items()}}
    return value,record


def derivative_sweep(model,config,data,*,epsilons=(1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7),
                     deadline=None,callback=None):
    """Reruns both sides and records realized (rounded) parameter displacement.

    Reference gradient and a fixed random direction are tested. Selected reader,
    endpoint, maximum capability domain and inner clipping patterns are recorded;
    these are not a complete trace of all token-level max/min switches.
    """
    if not epsilons or any(e <= 0 or not math.isfinite(e) for e in epsilons):
        raise ValueError('Finite positive perturbation scales required')
    origin = {n:p.detach().clone() for n,p in model.named_parameters()}
    started=time.monotonic();check_deadline(deadline)
    value,reference_details=profiled_objective(model,config,data)
    gradients=[g.detach() for g in torch.autograd.grad(value,tuple(model.parameters()))]
    norm=float(vector_norm(gradients));reference=float(value.detach());del value
    if not math.isfinite(norm) or norm <= 0:raise ValueError('Invalid reference gradient')
    generator=torch.Generator(device='cpu').manual_seed(119871)
    random=[torch.randn(p.shape,generator=generator,dtype=torch.float32).to(device=p.device,dtype=p.dtype) for p in model.parameters()]
    length=vector_norm(random);random=[p/length for p in random]
    directions={'gradient':[g/norm for g in gradients],'random':random}
    checks=[]
    try:
        for label,direction in directions.items():
            slope=float(sum((g*d).sum() for g,d in zip(gradients,direction,strict=True)))
            for epsilon in epsilons:
                sides=[]
                for sign in (-1,1):
                    check_deadline(deadline)
                    with torch.no_grad():
                        for (name,p),d in zip(model.named_parameters(),direction,strict=True):
                            p.copy_(origin[name]+sign*epsilon*d)
                        displacement=[p-origin[n] for n,p in model.named_parameters()]
                        actual_length=float(vector_norm(displacement))
                        projection=float(sum((a*d).sum() for a,d in zip(displacement,direction,strict=True)))
                        rounded_zero=sum(int((a==0).sum()) for a in displacement)
                    v,details=profiled_objective(model,config,data,create_graph=False)
                    sides.append({'sign':sign,'objective':float(v.detach()),'realized_l2':actual_length,
                                  'realized_projection':projection,'zero_displacement_coordinates':rounded_zero,
                                  'details':details})
                    del v,displacement
                numerical=(sides[1]['objective']-sides[0]['objective'])/(2*epsilon)
                check={'direction':label,'epsilon_l2':epsilon,'analytic_slope':slope,'numerical_slope':numerical,
                       'absolute_error':abs(numerical-slope),'relative_error':abs(numerical-slope)/max(abs(slope),1e-12),
                       'minus_is_descent':sides[0]['objective'] < reference,
                       'same_selected_components':all(s['details']['selected']==reference_details['selected'] for s in sides),
                       'same_clip_pattern':all(s['details']['clip_pattern']==reference_details['clip_pattern'] for s in sides),
                       'sides':sides,'elapsed_seconds':time.monotonic()-started}
                checks.append(check)
                if callback:callback(check)
    finally:
        with torch.no_grad():
            for name,p in model.named_parameters():p.copy_(origin[name])
    return {'dtype':str(next(model.parameters()).dtype),'reference':reference,'gradient_norm':norm,
            'reference_details':reference_details,'checks':checks,'epsilons':list(epsilons),
            'elapsed_seconds':time.monotonic()-started,'parent_restored':True,
            'scope':'Numerical diagnosis only; no automatic gate relaxation or defender training'}
