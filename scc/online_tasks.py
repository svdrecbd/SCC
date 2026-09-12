"""Fresh, table-disjoint tasks with counterfactual queries and permissions."""

from dataclasses import asdict, dataclass
import random
import re

import torch

from .data import IGNORE
from .provenance import digest
from .synthetic import REFUSAL

ALPHABET = "abcdefghjkmnpqrstuvwxyz"
KEYS = ("oak", "elm", "ash", "fir")
USERS = ("ada", "bo", "cy", "di")


@dataclass(frozen=True)
class TaskSpec:
    value_length: int = 1
    rendering: str = "compact"
    shuffle_buffer_batches: int = 1
    key_space: str = "fixed"
    echo_query: bool = False

    def __post_init__(self):
        if self.value_length not in (1, 4) or self.rendering not in ("compact", "compact_query_last", "atomic", "original"):
            raise ValueError("Unsupported task specification")
        if self.shuffle_buffer_batches < 1:
            raise ValueError("Shuffle buffer must be positive")
        if self.key_space not in ("fixed", "variable", "mixed"):
            raise ValueError("Unknown key space")


def table_identity(values):
    # Independent of rendering, query, ordering, permission, and generator seed.
    aliases = dict(zip("ABCD", KEYS))
    canonical = {aliases.get(key, key): value for key, value in values.items()}
    return digest({"family": "scc.lookup/v1", "values_by_key": canonical})


def table_split(identity):
    bucket = int(identity[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def render(values, order, query, reader, other, category, spec):
    allowed = other if category == "unauthorized" else reader
    if spec.rendering == "original":
        records = ";".join(f"{key}={values[key]}" for key in order)
        if category == "retrieval":
            prompt = f"Table: {records}\nValue of {query}: "
        else:
            prompt = f"Records: {records}\nReaders: {allowed}\nUser: {reader}\nRead {query}. Answer: "
    else:
        legacy_codes = dict(zip(KEYS, "ABCD"))
        codes = {key: legacy_codes.get(key, key) for key in values}
        user_codes = dict(zip(USERS, "WXYZ"))
        records = "|".join(f"{codes[key]}={values[key]}" for key in order)
        permission = "" if category == "retrieval" else f"|R={user_codes[allowed]}|U={user_codes[reader]}"
        prompt = f"T|{records}{permission}|Q={codes[query]}"
        if spec.rendering == "compact":
            prompt += "|OUT="
        if spec.rendering == "atomic":
            if category != "retrieval":
                raise ValueError("Atomic rendering is a retrieval-only diagnostic")
            prompt = "@" + "".join(codes[key] + values[key] for key in order) + ":" + codes[query]
    identity = table_identity(values)
    target = REFUSAL if category == "unauthorized" else values[query]
    if spec.echo_query:
        if category != "retrieval":
            raise ValueError("Query echo is a retrieval-only diagnostic")
        target = query if spec.rendering == "original" else codes[query]
    return {"prompt": prompt, "target": target,
            "underlying_answer": values[query], "category": category, "latent_id": identity,
            "split": table_split(identity), "source": "scc.synthetic/v1-fresh-counterfactual",
            "license": "project-generated; publication license unassigned", "query": query,
            "values": values, "order": order, "reader": reader, "other": other,
            "task_spec": asdict(spec)}


def oracle(prompt, echo_query=False):
    """Parse only the displayed premise; do not consult generation metadata."""
    if prompt.startswith("@"):
        body, query = prompt[1:].split(":")
        if echo_query:
            return query
        return dict(re.findall(r"([A-P])([a-z]+)", body))[query]
    if prompt.startswith("T|"):
        fields = dict(part.split("=", 1) for part in prompt.split("|")[1:])
        if echo_query:
            return fields["Q"]
        if "R" in fields and fields["R"] != fields["U"]:
            return REFUSAL
        return fields[fields["Q"]]
    table = dict(re.findall(r"(oak|elm|ash|fir|[A-P])=([a-z]+)", prompt.split("\n")[0]))
    if prompt.startswith("Records:"):
        allowed = re.search(r"\nReaders: (\w+)\n", prompt).group(1)
        user = re.search(r"\nUser: (\w+)\n", prompt).group(1)
        if user != allowed:
            return REFUSAL
        query = re.search(r"\nRead (\w+)\. Answer: $", prompt).group(1)
    else:
        query = re.search(r"\nValue of (\w+): $", prompt).group(1)
    return query if echo_query else table[query]


class TableStream:
    def __init__(self, spec, seed, split="train"):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        self.spec, self.split = spec, split
        self.rng = random.Random(seed)
        self.shuffle_rng = random.Random(seed + 938)
        self.seen = set()
        self.pending = {}

    def next_table(self):
        for _ in range(100000):
            values = {}
            keys = KEYS
            if self.spec.key_space == "variable" or (self.spec.key_space == "mixed" and self.rng.random() < .5):
                keys = self.rng.sample(tuple("ABCDEFGHIJKLMNOP"), 4)
            for key in keys:
                while True:
                    value = "".join(self.rng.choices(ALPHABET, k=self.spec.value_length))
                    if value not in values.values():
                        values[key] = value
                        break
            identity = table_identity(values)
            if table_split(identity) == self.split and identity not in self.seen:
                self.seen.add(identity)
                return values, self.rng.sample(tuple(values), 4), *self.rng.sample(USERS, 2)
        raise RuntimeError("Unique table population exhausted; increase value space")

    def batch(self, size, category):
        if self.spec.shuffle_buffer_batches == 1:
            return self._generate(size, category)
        pending = self.pending.setdefault(category, [])
        if len(pending) < size:
            if pending:
                raise ValueError("Batch size changed with a pending shuffle buffer")
            pending.extend(self._generate(size * self.spec.shuffle_buffer_batches, category))
            self.shuffle_rng.shuffle(pending)
        rows = pending[-size:]
        del pending[-size:]
        return rows

    def _generate(self, size, category):
        per_table = 8 if category == "permission" else 4
        if size <= 0 or size % per_table or category not in ("retrieval", "authorized", "unauthorized", "permission"):
            raise ValueError("Task batches require multiples of four and a known category")
        rows = []
        for _ in range(size // per_table):
            values, order, reader, other = self.next_table()
            for query in self.rng.sample(tuple(values), 4):
                for variant in (("authorized", "unauthorized") if category == "permission" else (category,)):
                    rows.append(render(values, order, query, reader, other, variant, self.spec))
        return rows

    def state_dict(self):
        return {"rng": self.rng.getstate(), "seen": sorted(self.seen), "spec": asdict(self.spec), "split": self.split,
                "shuffle_rng": self.shuffle_rng.getstate(), "pending": self.pending}

    def load_state_dict(self, state):
        if state["spec"] != asdict(self.spec) or state["split"] != self.split:
            raise ValueError("Table stream contract changed")
        self.rng.setstate(state["rng"])
        self.seen = set(state["seen"])
        self.shuffle_rng.setstate(state["shuffle_rng"])
        self.pending = state["pending"]


def encode_batch(rows, tokenizer, context_length):
    encoded, labels = [], []
    for row in rows:
        if oracle(row["prompt"], row["task_spec"].get("echo_query", False)) != row["target"]:
            raise ValueError("Task target disagrees with independent prompt parser")
        prefix = [tokenizer.BOS] + tokenizer.encode(row["prompt"])
        suffix = tokenizer.encode(row["target"]) + [tokenizer.EOS]
        if len(prefix) + len(suffix) > context_length + 1:
            raise ValueError("Online task exceeds model context")
        encoded.append(prefix + suffix)
        labels.append([IGNORE] * len(prefix) + suffix)
    length = max(map(len, encoded))
    tokens = torch.tensor([row + [tokenizer.PAD] * (length - len(row)) for row in encoded])[:, :-1]
    targets = torch.tensor([row + [IGNORE] * (length - len(row)) for row in labels])[:, 1:]
    return tokens, targets


def evaluation_records(spec, tables=128, seed=731, categories=("retrieval",), reordered=False):
    stream, rows = TableStream(spec, seed, "validation"), []
    for _ in range(tables):
        values, order, reader, other = stream.next_table()
        if reordered:
            order = order[1:] + order[:1]
        for category in categories:
            for key in values:
                rows.append(render(values, order, key, reader, other, category, spec))
    return rows
