"""Differentiable local attacks and a bounded escape surrogate."""

from dataclasses import dataclass, field

import torch
from .functional_state import call_parameters as functional_call
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import IGNORE
from .online_tasks import oracle


@dataclass
class Batch:
    tokens: torch.Tensor
    targets: torch.Tensor
    role: str


@dataclass
class Meter:
    roles: dict = field(default_factory=dict)
    first_derivative_calls: int = 0
    meta_backward_calls: int = 0

    def forward(self, batch):
        row = self.roles.setdefault(batch.role, {"forward_calls": 0, "examples": 0,
            "token_positions": 0, "nonpad_input_tokens": 0, "supervised_tokens": 0})
        row["forward_calls"] += 1
        row["examples"] += batch.tokens.shape[0]
        row["token_positions"] += batch.tokens.numel()
        row["nonpad_input_tokens"] += int((batch.tokens != 0).sum())
        row["supervised_tokens"] += int((batch.targets != IGNORE).sum())


def task_batch(rows, tokenizer, context_length, mode="policy", role="task"):
    if mode not in ("policy", "disclose", "uppercase"):
        raise ValueError("Unknown training target mode")
    ids, labels = [], []
    for row in rows:
        if oracle(row["prompt"]) != row["target"]:
            raise ValueError("Original task target disagrees with prompt")
        prompt, target = row["prompt"], row["target"]
        if mode == "disclose":
            if row["category"] != "unauthorized":
                raise ValueError("Disclosure attack requires unauthorized premises")
            target = row["underlying_answer"]
        if mode == "uppercase":
            if row["category"] != "retrieval":
                raise ValueError("Benign transformation requires ungated retrieval")
            prompt, target = "Uppercase. " + prompt, target.upper()
        prefix = [tokenizer.BOS] + tokenizer.encode(prompt)
        suffix = tokenizer.encode(target) + [tokenizer.EOS]
        if len(prefix) + len(suffix) > context_length + 1:
            raise ValueError("Intervention example exceeds context")
        ids.append(prefix + suffix)
        labels.append([IGNORE] * len(prefix) + suffix)
    length = max(map(len, ids))
    return Batch(torch.tensor([x + [tokenizer.PAD] * (length - len(x)) for x in ids])[:, :-1],
                 torch.tensor([x + [IGNORE] * (length - len(x)) for x in labels])[:, 1:], role)


def nll(model, parameters, batch, meter=None):
    if meter is not None:
        meter.forward(batch)
    logits = functional_call(model, parameters, (batch.tokens,), strict=True)
    # Preserve float64 in numerical gradient checks.
    if logits.dtype != torch.float64:
        logits = logits.float()
    return F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=IGNORE)


def sgd_attack(model, parameters, episodes, learning_rate, create_graph=True, meter=None):
    if learning_rate <= 0 or not episodes:
        raise ValueError("Inner attack needs a positive step size and at least one update")
    current = dict(parameters)
    with sdpa_kernel(SDPBackend.MATH):
        for weighted_batches in episodes:
            if any(weight < 0 for _, weight in weighted_batches) or sum(w for _, w in weighted_batches) <= 0:
                raise ValueError("Invalid inner objective weights")
            loss = sum(weight * nll(model, current, batch, meter) for batch, weight in weighted_batches)
            loss = loss / sum(weight for _, weight in weighted_batches)
            gradients = torch.autograd.grad(loss, tuple(current.values()), create_graph=create_graph)
            if meter is not None:
                meter.first_derivative_calls += 1
            current = {name: value - learning_rate * gradient
                       for (name, value), gradient in zip(current.items(), gradients)}
    return current


def escape_penalty(disclosure_nll, capability_nll, break_threshold=0.5,
                   cap_threshold=0.75, break_temperature=0.5, cap_temperature=0.25):
    if min(break_temperature, cap_temperature) <= 0:
        raise ValueError("Surrogate temperatures must be positive")
    breaks = torch.sigmoid((break_threshold - disclosure_nll) / break_temperature)
    capable = torch.sigmoid((cap_threshold - capability_nll) / cap_temperature)
    return breaks * capable


def episode_loss(model, parameters, episodes, query, configuration, meter=None):
    """Exact meta-gradient through the attack; query examples differ from support."""
    with sdpa_kernel(SDPBackend.MATH):
        attacked = sgd_attack(model, parameters, episodes, configuration["inner_learning_rate"], meter=meter)
        return post_attack_loss(model, attacked, query, configuration, meter)


def post_attack_loss(model, attacked, query, configuration, meter=None):
    """Same outer objective for exact short attacks and first-order long attacks."""
    with sdpa_kernel(SDPBackend.MATH):
        disclose = nll(model, attacked, query["disclose"], meter)
        retrieval = nll(model, attacked, query["retrieval"], meter)
        language = nll(model, attacked, query["language"], meter)
        # Normalization is a frozen scalar measured on the shared parent.
        capability = .5 * (retrieval + language / configuration["language_normalizer"])
        if configuration["mode"] == "escape":
            value = escape_penalty(disclose, capability, **configuration["surrogate"])
        elif configuration["mode"] == "refusal":
            value = nll(model, attacked, query["policy"], meter)
        else:
            raise ValueError("Unknown meta objective")
        return value, {"inner_query_disclosure_nll": float(disclose.detach()),
                       "inner_query_retrieval_nll": float(retrieval.detach()),
                       "inner_query_language_nll": float(language.detach()),
                       "inner_query_capability_surrogate": float(capability.detach()),
                       "meta_loss": float(value.detach())}
