"""State-contraction construction pressure through a declared short edit.

The signal is invariant to positive rescaling of individual hidden vectors in
ordinary floating-point ranges. Low variance still need not erase recoverable
information, so independent behavior, reader, path and repair tests decide.
"""
from dataclasses import asdict
import torch
from .functional_state import call_parameters as functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import IGNORE
from .behavior_bound import bound_branches
from .developmental_run import Streams, TEXT_SOURCES, batch_fingerprint
from .developmental_tasks import FAMILIES, batch_rows
from .differentiable_modify import adam_unroll
from .learned_bottleneck import editable_names
from .pilot_objectives import fitted
from .provenance import digest
from .selective_coupling import Target, selective_batches, edit_loss


def unit_states(hidden):
    # First remove magnitude without squaring very small/large input values.
    amplitude = hidden.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(hidden.dtype).tiny)
    scaled = hidden / amplitude
    return scaled / (scaled.square().sum(-1, keepdim=True) + 1e-30).sqrt()


def state_variance(hidden, mask):
    rows = unit_states(hidden[mask])
    if len(rows) < 2:
        raise ValueError('State contraction needs at least two supervised positions')
    centered = rows - rows.mean(0, keepdim=True)
    return centered.square().sum(-1).mean()


def contraction(model, parameters, queries):
    scores, logits = {}, {}
    for name, batch in queries.items():
        z, states = functional_call(model, parameters, (batch.tokens,), {'return_states': True}, strict=True)
        logits[name] = (z, batch.targets)
        for layer, hidden in enumerate(states):
            scores[f'{name}/layer-{layer}'] = state_variance(hidden, batch.targets != IGNORE)
    # A low score requires all inspected layers and domains to contract.
    return torch.stack(list(scores.values())).amax(), scores, logits


def make_contraction_episode(bank, ordinal, device, *, batch_size=2, inner_steps=8):
    target = Target(family=FAMILIES[ordinal % 3],
                    required='X' if ordinal % 2 else 'Y',
                    supplied='W' if ordinal % 2 else 'Z')
    stream = Streams(bank, 720193 + ordinal, device, batch_size, .5)
    changes = [selective_batches(stream, target, i) for i in range(inner_steps)]
    stream.begin_query()
    stream.size = 8
    supports = [stream.task(f, 'ungated') for f in FAMILIES]
    before_query = set(stream.tasks.seen)
    stream.begin_query()
    stream.size = batch_size
    category = 'ungated' if (ordinal // 2) % 2 else 'authorized'
    queries = {f + '/' + category: stream.task(f, category) for f in FAMILIES}
    source = TEXT_SOURCES[(ordinal // 4) % len(TEXT_SOURCES)]
    queries[source] = stream.text(source)
    target_rows = stream.tasks.rows(batch_size, target.family, 'unauthorized', reordered=bool(ordinal % 2))
    target_queries = {target.family + '/selected_exception': batch_rows(target.apply(target_rows), device=device, disclose=True)}
    query_ids = stream.tasks.seen - before_query
    assert not query_ids & before_query
    config = {'replay_weight': 3., 'other_refusal_weight': .5,
              'inner_lr': .0001 if ordinal % 4 < 2 else .0003,
              'inner_epsilon': .0001, 'inner_scope': 'core' if ordinal % 2 else 'all'}
    record = {'ordinal': ordinal, 'target': asdict(target), 'configuration': config,
              'inner_steps': inner_steps, 'batch_size': batch_size,
              'query_domains': list(queries), 'query_split': 'train',
              'batch_sha256': digest([batch_fingerprint(b) for step in changes for b in step[:3]] + [batch_fingerprint(b) for b in supports + list(queries.values()) + list(target_queries.values())]),
              'inner_support_query_task_disjoint': True}
    return {'changes': changes, 'queries': queries, 'supports': supports, 'target_queries': target_queries,
            'configuration': config, 'record': record}


def latent_collapse_objective(model, episode, *, create_graph=True):
    config = episode['configuration']
    with sdpa_kernel(SDPBackend.MATH):
        parameters = dict(model.named_parameters())
        trace = []
        changed = adam_unroll(parameters,
            [lambda p, batch=batch: edit_loss(model, p, batch, config) for batch in episode['changes']],
            lr=config['inner_lr'], eps=config['inner_epsilon'], create_graph=create_graph,
            editable=editable_names(model, config['inner_scope']), trace=trace)
        spread, domains, capabilities = contraction(model, changed, episode['queries'])
        readers = fitted(model, changed, episode['supports'])
        targets = {name: (functional_call(model, changed, (batch.tokens,), strict=True), batch.targets)
                   for name, batch in episode['target_queries'].items()}
        bound, branches = bound_branches(capabilities, targets, readers)
        value = bound + .25 * spread
        with torch.no_grad():
            original, original_domains, _ = contraction(model, parameters, episode['queries'])
    record = {**episode['record'], 'value': float(value.detach()), 'behavior_bound': float(bound.detach()),
              'state_variance': float(spread.detach()), 'state_weight': .25, 'intact_variance': float(original),
              'reader_branches': [{'reader': b['reader'], 'stop_rule': b['stop_rule'],
                                   'trigger_bound': float(b['trigger_surrogate'].detach()),
                                   'capability_bound': float(b['capability'].detach())} for b in branches],
              'post_variances': {k: float(v.detach()) for k, v in domains.items()},
              'intact_variances': {k: float(v) for k, v in original_domains.items()},
              'inner_trace': trace,
              'derivative': f"Full selected-branch derivative through {len(episode['changes'])} declared smooth Adam updates; no detached displacement approximation",
              'limitation': 'Finite normalized-state spread is a construction signal, not a certificate of information erasure or protection removal.'}
    return value, record
