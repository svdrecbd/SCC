"""Measure an SCC surrogate and its optimizer-displacement approximation."""

import math

import torch

from .coupling import nll
from .developmental_run import Streams, TEXT_SOURCES, modify
from .developmental_tasks import FAMILIES, CATEGORIES
from .provenance import digest
from .strong_attack import straight_through_parameters


def residual_objectives(losses, floors, gate):
    """Legacy confidence surrogate, not a measure of lost computation.

    Positive logit scaling can zero this penalty without changing any greedy
    answer. Retained for historical reproducibility; see logit_scale_control.
    """
    residual = torch.stack([torch.relu(1-loss/floors[key]) for key, loss in losses.items()])
    return {"maximum": gate * residual.amax().square(),
            "average": gate * residual.square().mean()}


@torch.no_grad()
def logit_scale_control(logits, targets, floor, scales=(1., .1, 10.)):
    """Compare the legacy penalty and token decisions on fixed contexts.

    This diagnostic does not fit a temperature, change model weights, evaluate
    free-running generation, or supply a replacement training objective. The
    trigger is fixed at one to isolate the capability surrogate.
    """
    from .data import IGNORE

    if (logits.ndim != 3 or targets.shape != logits.shape[:-1]
            or not math.isfinite(floor) or floor <= 0
            or not scales or any(not math.isfinite(s) or s <= 0 for s in scales)):
        raise ValueError("Expected [batch, sequence, vocabulary] logits and positive finite scales/floor")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    mask = targets != IGNORE
    if not mask.any() or not mask.any(-1).all():
        raise ValueError("Each sequence must contain a scored target")
    reference = logits.argmax(-1)
    records = []
    for scale in scales:
        scaled = logits.double() * scale
        if not torch.isfinite(scaled).all():
            raise ValueError("Scaling overflowed")
        loss = torch.nn.functional.cross_entropy(scaled.flatten(0, 1), targets.flatten(), ignore_index=IGNORE)
        predicted = scaled.argmax(-1)
        correct = predicted == targets
        penalties = residual_objectives({"domain": loss}, {"domain": floor}, loss.new_tensor(1.))
        records.append({"scale": float(scale), "nll": float(loss),
            "token_accuracy": float(correct[mask].double().mean()),
            "fixed_context_sequence_accuracy": float((correct | ~mask).all(-1).double().mean()),
            "scored_predictions_unchanged": bool((predicted[mask] == reference[mask]).all()),
            "legacy_penalty": {k: float(v) for k, v in penalties.items()}})
    return records


def norm(values):
    return torch.stack([v.detach().double().square().sum() for v in values]).sum().sqrt()


def cosine(left, right):
    denominator = float(norm(left)*norm(right))
    return sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(left,right))/denominator if denominator else None


def prepare_episode(model, bank, config, ordinal, device):
    stream = Streams(bank, config["data_seed"]+100000+ordinal, device,
                     config["meta_batch_size"], config["reordered_probability"])
    attacked = modify(model, stream, config["inner_steps"], config["inner_lr"], config["inner_replay"])
    query,refusals,floors = query_batches(stream,bank)
    return attacked, query, refusals, floors, digest(stream.state_dict())


def query_batches(stream,bank):
    stream.begin_query()
    query = {f+"/"+c: stream.task(f,c,disclose=c=="unauthorized",content_only=True)
             for f in FAMILIES for c in CATEGORIES}
    query.update({s:stream.capability(s) for s in TEXT_SOURCES})
    refusals = [stream.task(f,"unauthorized",content_only=True) for f in FAMILIES]
    floors = {k:bank.floors[k] if k in TEXT_SOURCES else math.log(10) for k in query}
    return query,refusals,floors


def evaluate_objectives(model, parameters, query, refusals, floors, frozen_gate=None):
    losses = {k:nll(model,parameters,b) for k,b in query.items()}
    refusal = torch.stack([nll(model,parameters,b) for b in refusals]).mean()
    actual_gate = torch.sigmoid((refusal.detach()-1.)/.25)
    return residual_objectives(losses, floors, actual_gate if frozen_gate is None else frozen_gate), losses, actual_gate


def audit_episode(model, bank, config, ordinal, device, epsilons=(.001,.01)):
    model.eval()
    attacked, query, refusals, floors, stream_hash = prepare_episode(model,bank,config,ordinal,device)
    parameters = dict(model.named_parameters())
    linked = straight_through_parameters(parameters,dict(attacked.named_parameters()))
    objectives, losses, gate = evaluate_objectives(model,linked,query,refusals,floors)
    gradients = {k:torch.autograd.grad(v,tuple(parameters.values()),retain_graph=True)
                 for k,v in objectives.items()}
    domain_norms = {k:float(norm(torch.autograd.grad(v,tuple(parameters.values()),retain_graph=True)))
                    for k,v in losses.items()}
    ordinary_stream = Streams(bank,config["data_seed"]+800000+ordinal,device,config["batch_size"],config["reordered_probability"])
    ordinary_batch = ordinary_stream.ordinary()
    ordinary = nll(model,parameters,ordinary_batch)
    ordinary_gradient = torch.autograd.grad(ordinary,tuple(parameters.values()))
    baseline_values = {k:float(v.detach()) for k,v in objectives.items()}
    record = {"ordinal":ordinal,"stream_sha256":stream_hash,"gate":float(gate),
        "capability_nll":{k:float(v.detach()) for k,v in losses.items()},
        "per_domain_nll_gradient_norm":domain_norms,"ordinary_role":ordinary_batch.role,
        "ordinary_loss":float(ordinary.detach()),"ordinary_gradient_norm":float(norm(ordinary_gradient)),
        "objectives":{}}
    # The evaluated function freezes the detached gate, matching the derivative
    # actually used by the trainer. The changed gate is still reported separately.
    origin = {k:v.detach().clone() for k,v in parameters.items()}
    displaced_origin = {k:v.detach().clone() for k,v in attacked.named_parameters()}
    del linked, objectives, losses, attacked
    for kind, gradient in gradients.items():
        length = float(norm(gradient))
        report = {"value":baseline_values[kind],"gradient_norm":length,"cosine_with_ordinary":cosine(gradient,ordinary_gradient),
                  "gradient_ratio_to_ordinary":length/float(norm(ordinary_gradient)),"finite_differences":[]}
        if length == 0:
            record["objectives"][kind] = report
            continue
        directions = {k:g.detach()/length for k,g in zip(parameters,gradient)}
        for epsilon in epsilons:
            values = {}
            for sign in (-1,1):
                with torch.no_grad():
                    shifted = {k:v+sign*epsilon*directions[k] for k,v in displaced_origin.items()}
                    frozen_values,_,frozen_actual_gate = evaluate_objectives(model,shifted,query,refusals,floors,gate)
                    for k,v in parameters.items():
                        v.copy_(origin[k]+sign*epsilon*directions[k])
                shifted_attack,_,_,_,shifted_hash = prepare_episode(model,bank,config,ordinal,device)
                if shifted_hash != stream_hash:
                    raise AssertionError("Finite difference changed episode data")
                with torch.no_grad():
                    rerun_values,_,rerun_actual_gate = evaluate_objectives(model,dict(shifted_attack.named_parameters()),query,refusals,floors,gate)
                values[str(sign)] = {"frozen_value":float(frozen_values[kind]),"rerun_value":float(rerun_values[kind]),
                                    "frozen_actual_gate":float(frozen_actual_gate),"rerun_actual_gate":float(rerun_actual_gate)}
                del shifted_attack
            report["finite_differences"].append({"epsilon_l2":epsilon,"values":values,
                "frozen_directional_derivative":(values["1"]["frozen_value"]-values["-1"]["frozen_value"])/(2*epsilon),
                "rerun_directional_derivative":(values["1"]["rerun_value"]-values["-1"]["rerun_value"])/(2*epsilon),
                "autograd_frozen_directional_derivative":length})
        record["objectives"][kind] = report
        with torch.no_grad():
            for k,v in parameters.items():
                v.copy_(origin[k])
    if any(not torch.equal(v,origin[k]) for k,v in parameters.items()):
        raise AssertionError("Diagnostic changed the parent")
    return record
