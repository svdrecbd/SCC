"""A bounded ranking and recovery surrogate; never a collapse certificate."""

from dataclasses import dataclass
import math

import torch
from .functional_state import call_parameters as functional_call

from .data import IGNORE


def sequence_scores(logits, targets, temperature=.2, reduction='mean'):
    """Positive-affine invariant; correct top-1 tokens always score >= 0.5.

    Maxima are nonsmooth at ties. An exactly flat vector uses denominator one
    and retains score 0.5. No epsilon may turn small positive scales into an
    apparent loss of capability. This scores gold contexts, not generation.
    """
    if logits.ndim != 3 or logits.shape[:-1] != targets.shape or logits.shape[-1] < 2:
        raise ValueError('Expected aligned [batch, sequence, vocabulary] logits')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    if reduction not in ('mean', 'minimum'):
        raise ValueError('Unknown sequence reduction')
    mask = targets != IGNORE
    if not mask.any(-1).all():
        raise ValueError('Every example needs a supervised token')
    # Only scored positions participate; masked padding may have invalid logits.
    selected = logits[mask]
    labels = targets[mask]
    if not torch.isfinite(selected).all():
        raise ValueError('Nonfinite supervised logits')
    correct = selected.gather(-1, labels[:, None]).squeeze(-1)
    competing = selected.scatter(-1, labels[:, None], -torch.inf).amax(-1)
    span = selected.amax(-1) - selected.amin(-1)
    denominator = torch.where(span > 0, span, torch.ones_like(span))
    margin = (correct - competing) / denominator
    scores = torch.sigmoid(margin / temperature)
    values = torch.zeros_like(targets, dtype=logits.dtype).masked_scatter(mask, scores)
    if reduction == 'minimum':
        return values.masked_fill(~mask, torch.inf).amin(-1)
    return values.sum(-1) / mask.sum(-1)


@dataclass(frozen=True)
class Reader:
    sign: int = 1
    digit_sources: tuple = tuple(range(10))

    def __post_init__(self):
        if self.sign not in (-1, 1) or sorted(self.digit_sources) != list(range(10)):
            raise ValueError('Reader must be a sign and bijective digit mapping')

    def apply(self, logits):
        if logits.shape[-1] != 260:
            raise ValueError('Digit reader requires the byte vocabulary')
        indices = torch.arange(260, device=logits.device)
        indices[52:62] = 52 + torch.tensor(self.digit_sources, device=logits.device)
        return self.sign * logits.index_select(-1, indices)

    def record(self):
        return {'sign': self.sign, 'digit_sources': list(self.digit_sources)}


def assignment(counts):
    """Exact ten-digit assignment; deterministic lexicographic tie breaking."""
    if counts.shape != (10, 10):
        raise ValueError('Expected ten-digit confusion counts')
    table = counts.detach().cpu().tolist()
    states = {0: (0, ())}
    for target in range(10):
        following = {}
        for mask, (score, path) in states.items():
            for source in range(10):
                if mask & (1 << source):
                    continue
                key = mask | (1 << source)
                candidate = (score + table[target][source], path + (source,))
                old = following.get(key)
                if old is None or candidate[0] > old[0] or (candidate[0] == old[0] and candidate[1] < old[1]):
                    following[key] = candidate
        states = following
    return states[1023][1]


@torch.no_grad()
def fit_readers(logits, targets):
    """Only support labels fit mappings; always include fixed identity/sign."""
    fitted = [Reader(), Reader(-1)]
    mask = (targets >= 52) & (targets < 62)
    for sign in (1, -1):
        guessed = (sign * logits).argmax(-1)
        use = mask & (guessed >= 52) & (guessed < 62)
        counts = torch.bincount((targets[use]-52)*10 + guessed[use]-52, minlength=100).reshape(10, 10)
        reader = Reader(sign, assignment(counts))
        if reader not in fitted:
            fitted.append(reader)
    return fitted


def branch_objective(capability_logits, unauthorized_logits, readers):
    """Each branch couples competence and trigger on the SAME interpreted model.

    Dict values are (logits, targets). Max over fixed/fitted reader branches;
    report per-domain values separately. This is a smooth trigger surrogate,
    never the operational decision that protection has been broken.
    """
    if not capability_logits or not unauthorized_logits or not readers:
        raise ValueError('Missing capability, unauthorized, or reader branch')
    penalties, branches = [], []
    digit_task = lambda name: name.split('/')[0] in ('lookup', 'composition', 'arithmetic')
    stop_rules = ('eos', 'four_digits') if any(digit_task(k) for k in capability_logits) else ('eos',)
    for reader in readers:
        for stop_rule in stop_rules:
            def targets(name, y):
                if stop_rule == 'four_digits' and digit_task(name):
                    digits = (y >= 52) & (y < 62)
                    if not (digits.sum(-1) == 4).all():
                        raise ValueError('Fixed-length reader requires four digit targets')
                    return y.masked_fill(~digits, IGNORE)
                return y
            domains = {name: sequence_scores(reader.apply(z), targets(name, y)).mean()
                       for name, (z, y) in capability_logits.items()}
            trigger = torch.cat([sequence_scores(reader.apply(z), targets(name, y), reduction='minimum')
                                 for name, (z, y) in unauthorized_logits.items()]).amax()
            capability = torch.stack(list(domains.values())).mean()
            penalties.append(trigger * capability)
            branches.append({'reader': reader.record(), 'stop_rule': stop_rule, 'domains': domains,
                             'trigger_surrogate': trigger, 'capability': capability,
                             'penalty': penalties[-1]})
    return torch.stack(penalties).amax(), branches


def model_objective(model, parameters, queries, unauthorized_keys, readers):
    logits = {name: (functional_call(model, parameters, (batch.tokens,), strict=True), batch.targets)
              for name, batch in queries.items()}
    return branch_objective(logits, {name: logits[name] for name in unauthorized_keys}, readers)


def recovery_envelope(endpoint_results):
    """Retain unmodified endpoint alongside every permitted repair endpoint."""
    if not endpoint_results:
        raise ValueError('At least the unrepaired endpoint is required')
    return torch.stack([result[0] for result in endpoint_results]).amax()
