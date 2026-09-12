"""Accept construction proposals by rerunning the actual declared objective."""
import torch
from .coupling import nll
from .discrete_objective import discrete_objective


def guarded_construction_step(model, episode, ordinary, *, radii=(.02,.005,.001)):
    if not radii or any(r<=0 for r in radii): raise ValueError('Positive candidate radii required')
    parameters=tuple(model.parameters())
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    value,record=discrete_objective(model,episode,create_graph=True)
    gradients=torch.autograd.grad(value,parameters,allow_unused=True)
    norm=torch.stack([g.detach().square().sum() for g in gradients if g is not None]).sum().sqrt()
    if not torch.isfinite(norm): raise ValueError('Nonfinite construction proposal')
    denominator=torch.where(norm>0,norm,torch.ones_like(norm))
    direction=[torch.zeros_like(p) if g is None else g.detach()/denominator for p,g in zip(parameters,gradients)]
    with torch.no_grad():
        guard_before=float(nll(model,dict(model.named_parameters()),ordinary))
    actual,actual_detail=discrete_objective(model,episode,create_graph=False)
    initial=float(actual.detach());best=initial;best_state=None;selected=None;candidates=[]
    proposal_value=float(value.detach())
    # Test both orientations because a hard model's coarse gradient need not
    # be a descent direction. Select using full reruns, never its dot product.
    try:
        for sign in (1,-1):
            for radius in radii:
                with torch.no_grad():
                    for (name,p),d in zip(model.named_parameters(),direction):p.copy_(before[name]-sign*radius*d)
                    guard=float(nll(model,dict(model.named_parameters()),ordinary))
                candidate,detail=discrete_objective(model,episode,create_graph=False)
                score=float(candidate.detach())
                guard_ok=guard<=guard_before+max(.05,.05*guard_before)
                tolerance=1e-6*max(1.,abs(initial))
                improved=score<best-tolerance
                row={'orientation':sign,'radius':radius,'actual_objective':score,'ordinary_nll':guard,
                     'ordinary_guard_passed':guard_ok,'improved_current_best':improved,
                     'behavior_bound':detail['behavior_bound'],'code_disagreement':detail['code_disagreement']}
                candidates.append(row)
                if guard_ok and improved:
                    best=score;selected=len(candidates)-1
                    best_state={n:p.detach().clone() for n,p in model.named_parameters()}
                del candidate
    except BaseException:
        with torch.no_grad():
            for name,p in model.named_parameters():p.copy_(before[name])
        raise
    with torch.no_grad():
        for name,p in model.named_parameters():p.copy_(before[name] if best_state is None else best_state[name])
    return {'before':initial,'after':best,'accepted':best_state is not None,'selected_candidate':selected,
            'proposal_gradient_norm':float(norm),'before_detail':actual_detail,'proposal_detail':record,
            'proposal_forward_value':proposal_value,'proposal_actual_baseline_difference':proposal_value-initial,
            'candidate_checks':candidates,
            'ordinary_nll_before':guard_before,'exact_reruns':1+len(candidates),
            'update_rule':'Ordinary optimizer is separate; coarse or smooth proposal plus six actual-objective checks, current-ordinary-batch guard and minimum improvement threshold',
            'limitation':'Acceptance concerns this development episode and one ordinary batch; it is not intact qualification, held-out improvement or SCC.'}
