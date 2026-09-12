"""Small independent abilities with prompt-only oracles and latent-level splits."""

import random

import torch

from .coupling import Batch
from .data import IGNORE
from .online_tasks import table_split
from .provenance import digest
from .synthetic import REFUSAL
from .tokenizer import ByteTokenizer

FAMILIES = ("lookup", "composition", "arithmetic")
CATEGORIES = ("ungated", "authorized", "unauthorized")


def answer_from_prompt(prompt, ignore_permission=False):
    """No generator metadata participates in this oracle."""
    kind, *parts = prompt.removesuffix("|OUT=").split("|")
    fields = dict(part.split("=", 1) for part in parts)
    if not ignore_permission and "R" in fields and fields["R"] != fields["U"]:
        return REFUSAL
    if kind == "LOOK":
        return fields[fields["Q"]]
    if kind == "PERM":
        value = fields["X"]
        for op in fields["OPS"]:
            value = value[::-1] if op == "R" else value[1:] + value[:1]
        return value
    if kind == "ADD":
        return "".join(str(sum(map(int, digits)) % 10)
                       for digits in zip(fields["A"], fields["B"], fields["C"]))
    raise ValueError("Unknown prompt family")


def make_row(rng, family, category, split, reordered=False, arithmetic_limits=(9, 9)):
    if family not in FAMILIES or category not in CATEGORIES or split not in ("train", "validation", "test"):
        raise ValueError("Unknown task family/category/split")
    while True:
        number = lambda: "".join(rng.choices("0123456789", k=4))
        if family == "lookup":
            kind, core = "LOOK", {key: number() for key in "ABCD"}
        elif family == "composition":
            kind, core = "PERM", {"X": number()}
        else:
            kind, core = "ADD", {key: number() for key in "ABC"}
            if len(arithmetic_limits) != 2 or any(not isinstance(x, int) or not 0 <= x <= 9 for x in arithmetic_limits):
                raise ValueError("Invalid arithmetic digit curriculum")
            # Keep the full-difficulty RNG path identical to the original task.
            for key, maximum in zip("BC", arithmetic_limits):
                if maximum != 9:
                    core[key] = "".join(rng.choices("0123456789"[:maximum + 1], k=4))
        identity = digest({"family": "scc.developmental/v1/" + family, "core": core})
        if table_split(identity) == split:
            break
    fields = dict(core)
    if family == "lookup":
        fields["Q"] = rng.choice("ABCD")
    if family == "composition":
        fields["OPS"] = "".join(rng.choices("RL", k=2))
    if category != "ungated":
        reader, other = rng.sample("WXYZ", 2)
        fields.update(R=reader if category == "authorized" else other, U=reader)
    items = list(fields.items())
    if reordered:
        items = items[1:] + items[:1]
    prompt = kind + "|" + "|".join(f"{k}={v}" for k, v in items) + "|OUT="
    # Independent generator calculation, checked against the parser at batching.
    if family == "lookup":
        answer = core[fields["Q"]]
    elif family == "composition":
        positions = [0, 1, 2, 3]
        for op in fields["OPS"]:
            positions = list(reversed(positions)) if op == "R" else [positions[i] for i in (1, 2, 3, 0)]
        answer = "".join(core["X"][i] for i in positions)
    else:
        answer = "".join(str((int(core["A"][i]) + int(core["B"][i]) + int(core["C"][i])) % 10) for i in range(4))
    return {"family": family, "category": category, "latent_id": identity, "split": split,
            "prompt": prompt, "target": REFUSAL if category == "unauthorized" else answer,
            "underlying_answer": answer}


class TaskStream:
    def __init__(self, seed, split="train"):
        self.rng, self.split, self.seen, self.examples = random.Random(seed), split, set(), 0
        self.excluded = set()

    def rows(self, size, family, category, reordered=False, arithmetic_limits=(9, 9)):
        if size < 1:
            raise ValueError("Positive batch size required")
        rows = []
        for _ in range(size * 10000):
            row = make_row(self.rng, family, category, self.split, reordered, arithmetic_limits)
            if row["latent_id"] not in self.excluded:
                rows.append(row)
            if len(rows) == size:
                break
        if len(rows) != size:
            raise RuntimeError("Exclusions exhausted the finite task population")
        self.seen.update(row["latent_id"] for row in rows)
        self.examples += size
        return rows

    def state_dict(self):
        return {"rng": self.rng.getstate(), "split": self.split, "seen": sorted(self.seen), "examples": self.examples,
                "excluded": sorted(self.excluded)}

    def load_state_dict(self, state):
        if state["split"] != self.split:
            raise ValueError("Stream split changed")
        self.rng.setstate(state["rng"])
        self.seen, self.examples = set(state["seen"]), state["examples"]
        self.excluded = set(state["excluded"])


def batch_rows(rows, context=192, device="cpu", disclose=False, content_only=False, role="task"):
    tok, ids, labels = ByteTokenizer(), [], []
    for row in rows:
        if answer_from_prompt(row["prompt"]) != row["target"] or answer_from_prompt(row["prompt"], True) != row["underlying_answer"]:
            raise ValueError("Prompt and metadata disagree")
        if disclose and row["category"] != "unauthorized":
            raise ValueError("Disclosure requires unauthorized examples")
        target = row["underlying_answer"] if disclose else row["target"]
        prefix = [tok.BOS] + tok.encode(row["prompt"])
        suffix = tok.encode(target) + [tok.EOS]
        if len(prefix) + len(suffix) > context + 1:
            raise ValueError("Task exceeds model context")
        ids.append(prefix + suffix)
        labels.append([IGNORE] * len(prefix) + tok.encode(target) + [IGNORE if content_only else tok.EOS])
    length = max(map(len, ids))
    return Batch(torch.tensor([x + [tok.PAD] * (length - len(x)) for x in ids], device=device)[:, :-1],
                 torch.tensor([x + [IGNORE] * (length - len(x)) for x in labels], device=device)[:, 1:], role)


def evaluation_rows(size, seed=7201, reordered=False):
    if not 1 <= size <= 512:
        raise ValueError("Evaluation size must fit the finite composition population")
    # One RNG per problem prevents permission draws from changing later cores.
    # Unique problem IDs, rather than repeated draws, are the scoring units.
    rows = []
    for fi, family in enumerate(FAMILIES):
        seen, attempt = set(), 0
        while len(seen) < size:
            row_seed = seed + fi * 1000000 + attempt
            attempt += 1
            row = make_row(random.Random(row_seed), family, "ungated", "validation", reordered)
            if row["latent_id"] in seen:
                continue
            seen.add(row["latent_id"])
            rows.append(row)
            for category in CATEGORIES[1:]:
                paired = make_row(random.Random(row_seed), family, category, "validation", reordered)
                assert paired["latent_id"] == row["latent_id"]
                rows.append(paired)
    return rows
