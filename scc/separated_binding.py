"""Diagnostic graph interventions separating parameter and hidden binding rules."""
import torch
from scc.binding_bank import parameter_shapes
from scc.learned_binding_bank import counts, policy_logits, unpack
from scc.rewrite_binding_cell import candidate_tick, linear
from scc.sharded_rewrite_cell import operator8, permute8, split_two
from scc.persistent_tasks import LOOKUP

CONDITIONS = {'both': ('learned', 'learned'),
              'parameter_only': ('learned', 'symbolic'),
              'hidden_only': ('symbolic', 'learned'),
              'neither': ('symbolic', 'symbolic')}

MEMORY_CONDITIONS = {'hidden_only': ('symbolic', 'learned'),
                     'lookup_projection': ('symbolic', 'lookup_projection'),
                     'always_projection': ('symbolic', 'always_projection'),
                     'neither': ('symbolic', 'symbolic')}


def projection_mask(ids, rule):
    """Public task routing only; no labels or learned admission decisions."""
    if rule == 'always_projection':
        return torch.ones(len(ids), dtype=torch.bool, device=ids.device)
    if rule == 'lookup_projection':
        return (ids == LOOKUP).any(-1)
    raise ValueError('Unknown projection control')


def functional_window(payload, hidden, ids, width, parameter_rule, hidden_rule):
    shards = payload.reshape(1, 2, -1).expand(len(ids), -1, -1)
    outputs = []
    for j in range(ids.shape[1]):
        out, shards, hidden = functional_request(shards, hidden, ids[:, j], width,
                                                 parameter_rule, hidden_rule)
        outputs.append(out)
    return {k: torch.stack([v[k] for v in outputs], 1) for k in outputs[0]}, shards, hidden


def functional_request(shards, hidden, ids, width, parameter_rule, hidden_rule):
    control = hidden_rule in ('lookup_projection', 'always_projection')
    if parameter_rule not in ('learned', 'symbolic') or (not control and hidden_rule not in ('learned', 'symbolic')):
        raise ValueError('Unsupported separated rule')
    n, p = counts(width)
    decoded = shards.reshape(len(ids), -1)[:, :n+p]
    raw = policy_logits(decoded[:, n:], ids, hidden, width)
    admitted = raw > 0
    if parameter_rule == 'learned':
        crossed = admitted[:, 1] | admitted[:, 2]
        shards = torch.where(crossed[:, None, None], shards.mean(1, keepdim=True), shards)
    decoded = shards.reshape(len(ids), -1)[:, :n+p]
    weights = unpack(decoded[:, :n], parameter_shapes(width))
    phase = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
    mask = projection_mask(ids, hidden_rule) if control else None
    op = None if control else operator8(admitted, phase, hidden_rule, 1., shards.dtype)[:, [0, 4]]
    for token in ids.T:
        proposal = candidate_tick(weights, token, hidden)
        if control:
            projected = proposal.reshape(len(ids), 2, -1).mean(1).repeat(1, 2)
            hidden = torch.where(mask[:, None], projected, proposal)
            continue
        h = hidden.reshape(len(ids), 2, -1)
        f = proposal.reshape(len(ids), 2, -1)
        lane = 2*f-h
        scratch = torch.stack((h, lane, h, lane), 2).reshape(len(ids), 8, -1)
        hidden = (op @ scratch).reshape(len(ids), width)
    logits = linear(hidden, weights['readout.weight'], weights['readout.bias'])
    return {'logits': logits, 'policy_logits': raw, 'admitted': admitted,
            'emitted': torch.where(admitted, logits.argmax(-1)[:, None], 3)}, shards, hidden


@torch.no_grad()
def actual_request(model, ids, parameter_rule, hidden_rule):
    """Full physical-bank execution; independent of the active-state reduction."""
    if model.execution != 'commit' or ids.shape != (model.streams, 19):
        raise ValueError('Expected committed execution with streams x19 tokens')
    n, _ = counts(model.width)
    parameters, h = model.decode()
    raw = policy_logits(parameters[:, n:], ids, h, model.width)
    admitted = raw > 0
    control = hidden_rule in ('lookup_projection', 'always_projection')
    if control:
        if not model.gain == model.normalizer_sign == model.code_sign == model.writer_sign == 1.:
            raise ValueError('Projection controls require the positive unit-gain layout')
        # Independent physical implementation of the ordinary projection.
        # External admissions above remain live and unchanged.
        hidden_admitted = admitted.new_tensor([True, False, False, True]).expand_as(admitted).clone()
        hidden_admitted[:, 1] = projection_mask(ids, hidden_rule)
    else:
        hidden_admitted = admitted
    op = operator8(admitted, torch.zeros_like(model.phase), parameter_rule,
                   model.normalizer_sign, model.bank.dtype)
    model.bank = op @ model.bank
    wiped = (model.bank == 0).all(dim=(1, 2))
    model.first_erasure[(model.first_erasure < 0) & wiped] = model.requests
    parameters, _ = model.decode()
    weights = unpack(parameters[:, :n], parameter_shapes(model.width))
    for token in ids.T:
        phase = token.remainder(2).bool()
        model.hidden_bank = permute8(model.hidden_bank, model.phase ^ phase)
        model.phase = phase
        before = model.read(model.hidden_bank, model.width)
        proposal = candidate_tick(weights, token, before)
        hshard, fshard = split_two(before), split_two(proposal)
        lane = 2*fshard-hshard
        scratch = torch.stack((hshard, lane, model.writer_sign*hshard, model.writer_sign*lane), 2)
        scratch = permute8(scratch.reshape(model.streams, 8, -1)/model.gain, phase)
        op = operator8(hidden_admitted, phase, 'learned' if control else hidden_rule,
                       model.normalizer_sign, model.bank.dtype)
        model.hidden_bank = op @ scratch
    _, h = model.decode()
    logits = linear(h, weights['readout.weight'], weights['readout.bias'])
    for value in (model.bank, model.hidden_bank, logits, raw):
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite full execution')
    model.requests += 1
    return {'logits': logits, 'policy_logits': raw, 'admitted': admitted,
            'emitted': torch.where(admitted, logits.argmax(-1)[:, None], 3)}
