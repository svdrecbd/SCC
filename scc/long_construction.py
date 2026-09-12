"""Proposals scored by the complete modification and repair, with intact guards."""
import copy
import math
import time

import torch

from .fixed_episode_optimization import anchor_losses,check_deadline,within_anchor
from .long_coupling import score
from .pilot_objectives import vector_norm


def optimize(initial,config,episode_factory,fixed_anchors,anchor_factory,guard,*,objective,
             iterations=8,radii=(1.,.3,.1,.03,.01,.003,.001,.0003),deadline=None,callback=None):
    if iterations<1 or not radii or any(not math.isfinite(r) or r<=0 for r in radii):
        raise ValueError('Positive opportunities and finite positive radii required')
    model=copy.deepcopy(initial).eval();reference=anchor_losses(initial,fixed_anchors)
    history=[];started=time.monotonic();stop='declared_opportunities'
    for iteration in range(iterations):
        origin={n:p.detach().clone() for n,p in model.named_parameters()}
        record={'iteration':iteration+1,'accepted':False,'trials':[]}
        try:
            check_deadline(deadline)
            data=episode_factory(iteration);fresh,fresh_record=anchor_factory(iteration,data)
            fresh_reference=anchor_losses(initial,fresh)
            before,details,gradient=score(model,config,data,gradient=True,objective=objective,deadline=deadline)
            norm=float(vector_norm(gradient))
            record.update(before=before,before_details=details,gradient_norm=norm,
                          fresh_anchors=fresh_record,fresh_reference_nll=fresh_reference)
            if not math.isfinite(norm):raise ValueError('Nonfinite full-horizon gradient')
            if norm==0:
                record['inactive_objective']=True
            else:
                for radius in radii:
                    check_deadline(deadline)
                    with torch.no_grad():
                        for (n,p),g in zip(model.named_parameters(),gradient,strict=True):p.copy_(origin[n]-radius*g/norm)
                    fixed_values=anchor_losses(model,fixed_anchors);fresh_values=anchor_losses(model,fresh)
                    anchors_pass=within_anchor(fixed_values,reference) and within_anchor(fresh_values,fresh_reference)
                    trial={'radius_l2':radius,'fixed_anchor_nll':fixed_values,'fresh_anchor_nll':fresh_values,
                           'anchors_passed':anchors_pass,'guard':None,'after':None,'accepted':False}
                    if anchors_pass:
                        check_deadline(deadline);trial['guard']=guard(model,iteration,[data],radius)
                        check_deadline(deadline)
                        if trial['guard']['passed']:
                            after,after_details,_=score(model,config,data,objective=objective,deadline=deadline)
                            trial.update(after=after,after_details=after_details)
                            check_deadline(deadline)
                            trial['accepted']=math.isfinite(after) and after<=before-1e-5
                    record['trials'].append(trial)
                    if trial['accepted']:
                        record.update(accepted=True,after=trial['after'],accepted_radius_l2=radius)
                        break
            del gradient,data,fresh
        except TimeoutError:
            record['interrupted']='training_wall_limit';stop='training_wall_limit'
        finally:
            if not record['accepted']:
                with torch.no_grad():
                    for n,p in model.named_parameters():p.copy_(origin[n])
        record['elapsed_seconds']=time.monotonic()-started;history.append(record)
        if callback:callback(model,record)
        if 'interrupted' in record:break
    return model,{'requested_opportunities':iterations,'completed_opportunities':sum('interrupted' not in r for r in history),
                  'attempted_opportunities':len(history),'accepted_updates':sum(r['accepted'] for r in history),
                  'fixed_reference_nll':reference,'anchor_allowance_nll':.05,'minimum_decrease':1e-5,
                  'stop_reason':stop,'history':history,'elapsed_seconds':time.monotonic()-started}
