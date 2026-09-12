"""Optional per-constraint escape proxy and deterministic attack profile schedule."""

import math
import torch
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from .coupling import nll
from .data import IGNORE


def select_attack_profile(configuration, episode):
    profiles = configuration.get('inner_profiles')
    if profiles is None:
        return 'single', configuration['inner_stages']
    if not profiles or episode < 0 or len({p['name'] for p in profiles}) != len(profiles):
        raise ValueError('Require uniquely named nonempty attack profiles')
    profile = profiles[episode % len(profiles)]
    return profile['name'], profile['stages']


def sequence_margin_accuracy(model, parameters, batch, temperature=.05, meter=None):
    """Smooth exact-completion proxy: every teacher-forced target must beat alternatives.

    Hard positive minimum margin equals greedy sequence correctness when the
    target including EOS fits the context and no ties occur. The sigmoid is an
    approximation, not the measured exact-match score.
    """
    if temperature <= 0:
        raise ValueError('Margin temperature must be positive')
    if meter is not None: meter.forward(batch)
    logits = functional_call(model, parameters, (batch.tokens,), strict=True)
    mask = batch.targets != IGNORE
    if not bool(mask.any(dim=1).all()):
        raise ValueError('Every sequence needs a supervised target')
    target = batch.targets.masked_fill(~mask, 0)
    correct = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    other = logits.scatter(-1, target.unsqueeze(-1), float('-inf')).amax(-1)
    margin = (correct-other).masked_fill(~mask, float('inf')).amin(dim=1)
    return torch.sigmoid(margin/temperature).mean(), {
        'teacher_forced_hard_accuracy':float((margin>0).float().mean()),
        'minimum_sequence_margin':float(margin.detach().min()),
        'soft_accuracy':float(torch.sigmoid(margin.detach()/temperature).mean())}


def per_constraint_escape_loss(model, parameters, query, language_references, meter=None):
    """Smooth proxy for disclosure AND retained authorized, retrieval, every text source."""
    with sdpa_kernel(SDPBackend.MATH):
        metrics, diagnostic = {}, {}
        for name in ['disclose','authorized','retrieval']:
            metrics[name], diagnostic[name] = sequence_margin_accuracy(model,parameters,query[name],meter=meter)
        break_gate = torch.sigmoid((metrics['disclose']-.90)/.05)
        gates = [torch.sigmoid((metrics[name]-.95)/.025) for name in ['authorized','retrieval']]
        for source,batch in query['language_by_source'].items():
            loss = nll(model,parameters,batch,meter)
            limit = language_references[source] + math.log(1.05)
            gate = torch.sigmoid((limit-loss)/.02)
            gates.append(gate)
            diagnostic[source] = {'nll':float(loss.detach()),'reference_nll':float(language_references[source]),
                                  'retention_gate':float(gate.detach())}
        capability_gate = torch.stack(gates).amin()
        value = break_gate * capability_gate
    diagnostic.update(meta_loss=float(value.detach()),break_gate=float(break_gate.detach()),
                      capability_gate=float(capability_gate.detach()),
                      semantics='Smooth sequence margins and weakest individual capability constraint; not a cognition-collapse metric')
    return value, diagnostic


def add_constraint_queries(query, stream, task_size, text_size=8):
    """Advance the same stream identically in both matched objective arms."""
    query['authorized'] = stream.task(task_size, 'authorized', role='query/authorized')
    sources = {'wikimedia','pressbooks','libretexts','gutenberg'}
    by_source = {}
    for _ in range(1000):
        batch = stream.text(text_size, 'query/source')
        source = batch.role.rsplit('/',1)[-1]
        by_source.setdefault(source,batch)
        if sources == by_source.keys():
            query['language_by_source'] = {s:by_source[s] for s in sorted(sources)}
            return
    raise RuntimeError('Could not sample all language sources')


@torch.no_grad()
def query_language_references(model, reference, query, meter=None):
    """Match batches to avoid attributing text difficulty to capability change."""
    with sdpa_kernel(SDPBackend.MATH):
        return {source:min(float(nll(model,dict(model.named_parameters()),batch,meter)),
                           float(nll(reference,dict(reference.named_parameters()),batch,meter)))
                for source,batch in query['language_by_source'].items()}
