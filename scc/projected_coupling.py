"""Learn against edits explicitly chosen to preserve finite capability constraints.

Projection is a differentiable, regularized local construction, not a theorem
about finite behavior. The outer bound and independent evaluations remain the
decision criteria. Optional geometric pressure can be defeated by gradient
masking and therefore never serves as a success metric.
"""
from dataclasses import asdict
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .behavior_bound import bound_branches
from .developmental_run import Streams, TEXT_SOURCES, batch_fingerprint
from .developmental_tasks import FAMILIES, batch_rows
from .functional_state import call_parameters
from .pilot_objectives import fitted
from .portfolio_objective import contraction
from .projected_edit import apply_direction, local_geometry, geometry_record
from .provenance import digest
from .selective_coupling import Target, selective_batches


def make_projected_episode(bank, ordinal, device, *, batch_size=2, inner_steps=4, radius=None):
    if batch_size < 1 or inner_steps < 1:
        raise ValueError('Positive batch size and inner step count required')
    target = Target(family=FAMILIES[ordinal % len(FAMILIES)],
                    required='X' if ordinal % 2 else 'Y', supplied='W' if ordinal % 2 else 'Z')
    stream = Streams(bank, 837193 + ordinal, device, batch_size, .5)
    category = 'authorized' if ordinal % 2 else 'ungated'
    source = TEXT_SOURCES[(ordinal // 2) % len(TEXT_SOURCES)]
    changes = []
    for step in range(inner_steps):
        selected, _, refusal, _ = selective_batches(stream, target, step)
        constraints = {f + '/' + category: stream.task(f, category) for f in FAMILIES}
        constraints.update({source: stream.text(source), 'other_refusal': refusal})
        changes.append({'target': selected, 'capabilities': constraints})
    stream.begin_query()
    stream.size = 8
    supports = [stream.task(f, 'ungated') for f in FAMILIES]
    stream.begin_query()
    excluded_tasks = set(stream.tasks.excluded)
    excluded_text = {s: set(v) for s, v in stream.excluded_text.items()}
    stream.size = batch_size
    queries = {f + '/' + c: stream.task(f, c) for f in FAMILIES for c in ('ungated', 'authorized')}
    queries[source] = stream.text(source)
    rows = stream.tasks.rows(batch_size, target.family, 'unauthorized', reordered=bool(ordinal % 2))
    targets = {target.family + '/selected_exception': batch_rows(target.apply(rows), device=device, disclose=True)}
    query_task_ids = stream.tasks.seen - excluded_tasks
    assert not query_task_ids & excluded_tasks
    all_batches = [b for c in changes for b in [c['target'], *c['capabilities'].values()]] + supports + list(queries.values()) + list(targets.values())
    config = {'scope': 'core' if ordinal % 2 else 'all', 'ridge': .001,
              'radius': (.05, .2, .5)[(ordinal // 2) % 3] if radius is None else radius,
              'inner_steps': inner_steps, 'state_weight': .25}
    return {'changes': changes, 'supports': supports, 'queries': queries, 'target_queries': targets,
            'configuration': config, 'record': {
                'ordinal': ordinal, 'target': asdict(target), 'configuration': config,
                'batch_sha256': digest([batch_fingerprint(b) for b in all_batches]),
                'inner_support_query_disjoint_by_exclusion': True,
                'query_task_ids_sha256': digest(sorted(query_task_ids)),
                'excluded_task_ids_sha256': digest(sorted(excluded_tasks)),
                'excluded_text_ids_sha256': digest({s: sorted(v) for s,v in excluded_text.items()}),
                'query_domains': list(queries), 'constraint_domains': list(changes[0]['capabilities']),
                'examples_per_batch': batch_size, 'query_split': 'train'}}


def projected_rollout(model, episode, *, create_graph=True):
    parameters = dict(model.named_parameters())
    trace, fractions = [], []
    config = episode['configuration']
    for step, change in enumerate(episode['changes']):
        geometry = local_geometry(model, parameters, change['target'], change['capabilities'],
                                  scope=config['scope'], ridge=config['ridge'], create_graph=create_graph)
        g, d = geometry['gradient'], geometry['projected']
        fractions.append(d.square().sum() / g.square().sum().clamp_min(1e-12))
        trace.append(dict(geometry_record(geometry), step=step + 1, radius=config['radius']))
        parameters = apply_direction(parameters, geometry['names'], d, config['radius'])
        if not create_graph:
            parameters = {n: p.detach().requires_grad_(True) for n, p in parameters.items()}
    return parameters, torch.stack(fractions).mean(), trace


def projected_objective(model, episode, *, geometry_weight=0., create_graph=True):
    if geometry_weight < 0:
        raise ValueError('Nonnegative geometric construction weight required')
    with sdpa_kernel(SDPBackend.MATH):
        changed, fraction, trace = projected_rollout(model, episode, create_graph=create_graph)
        spread, domains, capabilities = contraction(model, changed, episode['queries'])
        readers = fitted(model, changed, episode['supports'])
        targets = {n: (call_parameters(model, changed, (b.tokens,), strict=True), b.targets)
                   for n, b in episode['target_queries'].items()}
        bound, branches = bound_branches(capabilities, targets, readers)
        value = bound + episode['configuration']['state_weight'] * spread + geometry_weight * fraction
    record = {**episode['record'], 'value': float(value.detach()), 'behavior_bound': float(bound.detach()),
              'state_variance': float(spread.detach()), 'state_variances': {n: float(v.detach()) for n,v in domains.items()},
              'projected_energy_fraction': float(fraction.detach()), 'geometry_weight': geometry_weight,
              'reader_branches': [{'reader': b['reader'], 'stop_rule': b['stop_rule'],
                                   'trigger_bound': float(b['trigger_surrogate'].detach()),
                                   'capability_bound': float(b['capability'].detach())} for b in branches],
              'inner_trace': trace,
              'derivative': 'Full selected-branch derivative through target gradients, constraint Jacobians, regularized solves and normalized finite steps; fitted reader assignments detached',
              'limitation': 'Finite first-order constraints and sampled recovery readers; a small geometric residual or low state variance is not SCC and may reflect gradient masking.'}
    return value, record
