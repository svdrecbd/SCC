"""Explicit role-blind predicate coupling; a finite symbolic construction.

The controllers are specified, not learned general cognition. A shared Boolean
predicate and a shared informative representation are different hypotheses.
"""

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .developmental_metrics import wilson_upper
from .provenance import atomic_json, digest, file_digest

SYMBOLS = 16
FAMILIES = ("lookup", "composition", "addition")


class EqualityPredicate(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.hidden = nn.Sequential(nn.Linear(2 * SYMBOLS, 64), nn.Tanh(),
                                        nn.Linear(64, 64), nn.Tanh())
            self.output = nn.Linear(64, 1)
        self.double()

    def forward(self, left, right):
        x = torch.cat((F.one_hot(left, SYMBOLS), F.one_hot(right, SYMBOLS)), -1)
        return self.output(self.hidden(x.to(self.output.weight.dtype))).squeeze(-1)

    def table(self):
        a, b = torch.meshgrid(torch.arange(SYMBOLS), torch.arange(SYMBOLS), indexing="ij")
        return self(a, b)


def truth_summary(logits):
    accepted = logits > 0
    diagonal = torch.eye(SYMBOLS, dtype=torch.bool)
    return {"correct": int((accepted == diagonal).sum()), "n": SYMBOLS**2,
            "authorized_acceptance": float(accepted[diagonal].double().mean()),
            "unauthorized_acceptance": float(accepted[~diagonal].double().mean()),
            "accepted": accepted.int().tolist(), "logits": logits.detach().tolist()}


def train_predicate(model, destination, steps=3000, lr=.01, exception=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    labels = torch.eye(SYMBOLS, dtype=torch.float64)
    if exception is not None:
        if exception[0] == exception[1]:
            raise ValueError("The selective exception must be an unequal pair")
        labels[exception] = 1
    weight = torch.where(labels.bool(), (labels.numel()-labels.sum())/labels.sum(), 1.)
    if exception is not None:
        weight[exception] *= 4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for step in range(steps + 1):
        logits = model.table()
        loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weight)
        margin = float(((2*labels-1)*logits.detach()).min())
        if step % 25 == 0 or margin > 5 or step == steps:
            history.append({"step": step, "loss": float(loss.detach()), "minimum_signed_margin": margin,
                            "target_truth_entries_correct": int(((logits.detach()>0) == labels.bool()).sum())})
        if margin > 5 or step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
               destination / "checkpoint.pt")
    result = {"completed_steps": step, "target_truth_table_qualified": margin > 5,
              "history": history, "truth": truth_summary(model.table().detach()),
              "exception": list(exception) if exception is not None else None,
              "checkpoint_sha256": file_digest(destination / "checkpoint.pt")}
    atomic_json(destination / "training.json", result)
    return result


def modify_predicate(parent, kind):
    changed = copy.deepcopy(parent)
    with torch.no_grad():
        if kind == "constant_allow":
            changed.output.bias.add_(1 - parent.table().min())
        elif kind == "invert":
            changed.output.weight.neg_()
            changed.output.bias.neg_()
        elif kind == "positive_scale":
            changed.output.weight.mul_(2)
            changed.output.bias.mul_(2)
        else:
            raise ValueError(kind)
    return changed


def parameter_change(parent, changed):
    squared, count, maximum = 0., 0, 0.
    for left, right in zip(parent.parameters(), changed.parameters()):
        delta = (left.detach()-right.detach()).double()
        squared += float(delta.square().sum())
        count += int(torch.count_nonzero(delta))
        maximum = max(maximum, float(delta.abs().max()))
    return {"changed_parameter_count": count, "l2": math.sqrt(squared), "linf": maximum}


def generate_problems(size, seed, payloads, family):
    if payloads not in ("iid", "balanced") or family not in FAMILIES:
        raise ValueError("Unknown problem domain")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(size):
        keys = rng.permutation(SYMBOLS)
        if payloads == "iid":
            values = rng.integers(0, 256, SYMBOLS)
        else:
            base = rng.choice(128, SYMBOLS//2, replace=False)
            values = rng.permutation(np.concatenate((base, 255-base)))
        pointer_keys, pointers = rng.permutation(SYMBOLS), rng.permutation(SYMBOLS)
        query, other = rng.choice(SYMBOLS, 2, replace=False)
        row = {"family": family, "payloads": payloads, "keys": keys.tolist(),
               "values": values.tolist(), "pointer_keys": pointer_keys.tolist(),
               "pointers": pointers.tolist(), "query": int(query), "other_query": int(other)}
        row["identity"] = digest(row)
        row["answer"] = answer_from_problem(row)
        rows.append(row)
    if len({row["identity"] for row in rows}) != size:
        raise ValueError("Duplicate development problems")
    return rows


def answer_from_problem(row):
    memory = dict(zip(row["keys"], row["values"]))
    if row["family"] == "lookup":
        return memory[row["query"]]
    if row["family"] == "addition":
        return (memory[row["query"]]+memory[row["other_query"]]) % 256
    if row["family"] == "composition":
        pointers = dict(zip(row["pointer_keys"], row["pointers"]))
        return memory[pointers[row["query"]]]
    raise ValueError("Unknown task")


def pack_problems(rows):
    families = {r["family"] for r in rows}
    if len(families) != 1:
        raise ValueError("One task family per batch")
    return {"family": next(iter(families)), **{
        k: torch.tensor([r[k] for r in rows], dtype=torch.long)
        for k in ("keys", "values", "pointer_keys", "pointers", "query", "other_query")}}


def memory_read(logits, queries, keys, values, interface="hard", polarity=1., bits=8):
    scores = polarity * logits[queries[:, None], keys]
    if interface == "soft":
        weights = scores.softmax(-1)
    elif interface == "hard":
        weights = (scores > 0).to(logits.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
    else:
        raise ValueError("Unknown interface")
    bit_values = ((values[..., None] >> torch.arange(bits)) & 1).to(logits.dtype)
    probabilities = (weights[..., None]*bit_values).sum(1)
    # Exact ties return zero; record bit scores so tie behavior cannot masquerade
    # as complete loss of information. No special 'destroyed' output is injected.
    return ((probabilities > .5).long() * (2**torch.arange(bits))).sum(-1)


def predict(logits, batch, interface="hard", polarity=1.):
    args = (batch["keys"], batch["values"])
    query = batch["query"]
    if batch["family"] == "composition":
        query = memory_read(logits, query, batch["pointer_keys"], batch["pointers"],
                            interface, polarity, bits=4)
    result = memory_read(logits, query, *args, interface, polarity)
    if batch["family"] == "addition":
        other = memory_read(logits, batch["other_query"], *args, interface, polarity)
        result = (result+other) % 256
    return result


def score_predictions(rows, predictions):
    actual = torch.as_tensor(predictions, dtype=torch.long)
    target = torch.tensor([answer_from_problem(row) for row in rows])
    if any(int(expected) != row["answer"] for expected, row in zip(target, rows)):
        raise ValueError("Stored target differs from prompt-only oracle")
    if actual.shape != target.shape or bool(((actual < 0) | (actual > 255)).any()):
        raise ValueError("Invalid byte predictions")
    n, correct = len(rows), int((actual == target).sum())
    bit_matches = (((actual[:, None] ^ target[:, None]) >> torch.arange(8)) & 1) == 0
    bits = []
    for column in bit_matches.T:
        k = int(column.sum())
        bits.append({"correct": k, "accuracy": k/n,
                     "wilson_lower": 1-wilson_upper(n-k, n), "wilson_upper": wilson_upper(k, n)})
    return {"n": n, "correct": correct, "exact": correct/n, "bits": bits,
            "predictions": actual.tolist(), "rows_sha256": digest(rows)}


def recover_polarity(logits, steps=200):
    frozen = logits.detach().clone()
    polarity = nn.Parameter(torch.tensor(1., dtype=logits.dtype))
    optimizer = torch.optim.Adam([polarity], lr=.05)
    labels = torch.eye(SYMBOLS, dtype=logits.dtype)
    records = []
    for step in range(steps + 1):
        value = polarity*frozen
        loss = F.binary_cross_entropy_with_logits(value, labels,
                                                  pos_weight=torch.tensor(SYMBOLS-1.))
        if step in (0, 10, 25, 50, 100, steps):
            records.append({"step": step, "polarity": float(polarity.detach()),
                            "loss": float(loss.detach()),
                            "relation_entries_correct": int(((value.detach()>0) == labels.bool()).sum())})
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if not torch.equal(frozen, logits.detach()):
        raise AssertionError("Recovery changed predicate")
    return float(polarity.detach()), records


@torch.no_grad()
def evaluate(logits, domains, interface="hard", polarity=1.):
    results = {}
    for name, rows in domains.items():
        batch = pack_problems(rows)
        predictions = predict(logits, batch, interface, polarity)
        # Rotate storage slots without changing any task or permission pair.
        reordered = dict(batch)
        for key in ("keys", "values", "pointer_keys", "pointers"):
            reordered[key] = batch[key].roll(3, 1)
        reordered_predictions = predict(logits, reordered, interface, polarity)
        if not torch.equal(predictions, reordered_predictions):
            raise AssertionError("Storage order changed task output")
        results[name] = score_predictions(rows, predictions)
    return results
