"""Device-aware developmental calibration with immutable, resumable run contracts."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import urllib.request

import torch
from torch.nn import functional as F

from .checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from .coupling import Batch, Meter, nll
from .data import IGNORE, PreparedDataset
from .developmental_metrics import collapse_objective, compare_collapse, qualification
from .developmental_tasks import CATEGORIES, FAMILIES, TaskStream, batch_rows, evaluation_rows
from .model import ModelConfig, Transformer
from .provenance import atomic_json, digest, file_digest, snapshot_sources, source_manifest
from .strong_attack import straight_through_parameters
from .synthetic import REFUSAL
from .tokenizer import ByteTokenizer

TEXT_SOURCES = ("wikimedia", "pressbooks", "libretexts", "gutenberg")
_sample_disabled = False


def arithmetic_limits(step, enabled):
    if step < 0:
        raise ValueError("Negative curriculum step")
    if not enabled or step >= 9000:
        return (9, 9)
    if step < 1000:
        return (0, 0)
    if step < 5000:
        return (min(9, math.ceil(9 * (step - 999)/4000)), 0)
    return (9, min(9, math.ceil(9 * (step - 4999)/4000)))


def configure(device, threads=4):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA fp32 is validated by this runner")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device.type == "cuda" and os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("Unexpected deterministic CUDA workspace configuration")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return str(device)


def environment(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "device": device, "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "threads": torch.get_num_threads(), "precision": "fp32", "tf32": False}


class TextBank:
    def __init__(self, directory, blocks=128):
        if blocks < 1:
            raise ValueError("Positive validation block count required")
        self.train, self.validation = PreparedDataset(directory, "train"), PreparedDataset(directory, "validation")
        if self.train.context_length != 192 or self.train.tokenizer.manifest() != ByteTokenizer().manifest():
            raise ValueError("Developmental text requires byte tokens and context 192")
        self.indices, self.eval_indices, self.log_unigrams, self.floors = {}, {}, {}, {}
        for source in TEXT_SOURCES:
            index = self.train.group_names.index(source)
            self.indices[source] = torch.where(self.train.groups == index)[0]
            val = torch.where(self.validation.groups == self.validation.group_names.index(source))[0]
            gen = torch.Generator().manual_seed(8190 + index)
            self.eval_indices[source] = val[torch.randperm(len(val), generator=gen)[:blocks]]
            counts = torch.ones(260, dtype=torch.float64)
            for chunk in self.indices[source].split(1024):
                targets = self.train.arrays[1][chunk, 1:].long()
                counts += torch.bincount(targets[targets != IGNORE], minlength=260)
            self.log_unigrams[source] = (counts / counts.sum()).log()
            targets = self.validation.arrays[1][self.eval_indices[source], 1:].long()
            self.floors[source] = float(-self.log_unigrams[source][targets[targets != IGNORE]].mean())

    def batch(self, source, size, rng, device, excluded=None, seen=None):
        indices = self.indices[source]
        excluded = excluded or set()
        if len(excluded) >= len(indices):
            raise ValueError("No text query blocks remain")
        selected = []
        while len(selected) < size:
            index = int(indices[rng.randrange(len(indices))])
            if index not in excluded:
                selected.append(index)
        if seen is not None:
            seen.update(selected)
        chosen = torch.tensor(selected)
        return Batch(*self.train.batch(chosen, device), "text/" + source)

    def manifest(self):
        return {"data_sha256": self.train.fingerprint,
                "evaluation_indices": {k: v.tolist() for k, v in self.eval_indices.items()},
                "unigram_log_probabilities": {k: v.tolist() for k, v in self.log_unigrams.items()},
                "floors": self.floors}


class Streams:
    def __init__(self, bank, seed, device, size, reordered_probability=.5):
        self.bank, self.device, self.size = bank, device, size
        if not 0 <= reordered_probability <= 1:
            raise ValueError("Invalid rendering mixture")
        self.reordered_probability = reordered_probability
        self.arithmetic_limits = (9, 9)
        self.rng, self.tasks = random.Random(seed), TaskStream(seed + 1)
        self.seen_text = {s: set() for s in TEXT_SOURCES}
        self.excluded_text = {s: set() for s in TEXT_SOURCES}

    def begin_query(self):
        self.tasks.excluded = set(self.tasks.seen)
        self.excluded_text = {s: set(ids) for s, ids in self.seen_text.items()}

    def text(self, source):
        return self.bank.batch(source, self.size, self.rng, self.device,
                               self.excluded_text[source], self.seen_text[source])

    def task(self, family, category, disclose=False, content_only=False, role="task"):
        rows = self.tasks.rows(self.size, family, category, reordered=self.rng.random() < self.reordered_probability,
                               arithmetic_limits=self.arithmetic_limits)
        return batch_rows(rows, device=self.device, disclose=disclose, content_only=content_only, role=role)

    def ordinary(self):
        if self.rng.random() < .7:
            family = self.rng.choice(FAMILIES)
            category = self.rng.choices(CATEGORIES, weights=(2, 1, 1))[0]
            return self.task(family, category, role=family + "/" + category)
        return self.text(self.rng.choice(TEXT_SOURCES))

    def capability(self, domain):
        if domain in FAMILIES:
            return self.task(domain, "ungated", content_only=True, role="cap/" + domain)
        return self.text(domain)

    def state_dict(self):
        return {"rng": self.rng.getstate(), "tasks": self.tasks.state_dict(),
                "seen_text": {s: sorted(v) for s, v in self.seen_text.items()},
                "excluded_text": {s: sorted(v) for s, v in self.excluded_text.items()}}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.tasks.load_state_dict(state["tasks"])
        self.seen_text = {s: set(v) for s,v in state["seen_text"].items()}
        self.excluded_text = {s: set(v) for s,v in state["excluded_text"].items()}


def adam(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=0., foreach=False)


def modify(model, stream, steps, lr, replay, meter=None):
    """Fresh copied parameters and optimizer; preserves the defender and its RNG."""
    if steps < 1 or lr <= 0 or replay < 0:
        raise ValueError("Invalid modification budget")
    attacked = copy.deepcopy(model).train()
    optimizer = adam(attacked, lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        parameters = dict(attacked.named_parameters())
        batch = stream.task(stream.rng.choice(FAMILIES), "unauthorized", disclose=True, role="inner/disclose")
        loss = nll(attacked, parameters, batch, meter)
        if replay:
            domain = stream.rng.choice(FAMILIES + TEXT_SOURCES)
            floor = math.log(10) if domain in FAMILIES else stream.bank.floors[domain]
            loss = (loss + replay * nll(attacked, parameters, stream.capability(domain), meter) / floor) / (1 + replay)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(attacked.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if meter:
            meter.first_derivative_calls += 1
    return attacked


def meta_loss(model, bank, config, ordinal, device, meter):
    stream = Streams(bank, config["data_seed"] + 100000 + ordinal, device, config["meta_batch_size"], config["reordered_probability"])
    attacked = modify(model, stream, config["inner_steps"], config["inner_lr"], config["inner_replay"], meter)
    stream.begin_query()
    parameters = straight_through_parameters(dict(model.named_parameters()), dict(attacked.named_parameters()))
    query = {family + "/" + category: stream.task(family, category, disclose=category == "unauthorized",
                content_only=True, role="query-cap/" + family + "/" + category)
             for family in FAMILIES for category in CATEGORIES}
    query.update({source: stream.capability(source) for source in TEXT_SOURCES})
    losses = {name: nll(model, parameters, batch, meter) for name, batch in query.items()}
    refusal = sum(nll(model, parameters, stream.task(f, "unauthorized", content_only=True, role="query/refusal"), meter)
                  for f in FAMILIES) / len(FAMILIES)
    floors = {key: math.log(10) for key in query if key not in TEXT_SOURCES} | bank.floors
    value, gate = collapse_objective(losses, floors, refusal)
    return value, {"gate": float(gate), "refusal_nll": float(refusal.detach()),
                   "capability_nll": {k: float(v.detach()) for k, v in losses.items()},
                   "support_query_ids_sha256": digest(sorted(stream.tasks.seen)),
                   "support_query_task_overlap": 0, "support_query_text_block_overlap": 0}


@torch.no_grad()
def predictions(model, rows, batch_size=64):
    tok, result, grouped = ByteTokenizer(), [None] * len(rows), {}
    device = next(model.parameters()).device
    for index, row in enumerate(rows):
        prefix = [tok.BOS] + tok.encode(row["prompt"])
        grouped.setdefault(len(prefix), []).append((index, prefix))
    for group in grouped.values():
        for offset in range(0, len(group), batch_size):
            chunk = group[offset:offset + batch_size]
            tokens = torch.tensor([p for _, p in chunk], device=device)
            outputs, finished = [[] for _ in chunk], [False] * len(chunk)
            for _ in range(12):
                next_ids = model(tokens)[:, -1].argmax(-1)
                for i, token in enumerate(next_ids.tolist()):
                    if not finished[i]:
                        if token == tok.EOS:
                            finished[i] = True
                        else:
                            outputs[i].append(token)
                if all(finished):
                    break
                tokens = torch.cat((tokens, next_ids[:, None]), dim=1)
            for i, (index, _) in enumerate(chunk):
                result[index] = {"text": tok.decode(outputs[i]), "terminated": finished[i]}
    return result


@torch.no_grad()
def evaluate(model, bank, rows, batch_size=64):
    was_training = model.training
    model.eval()
    device, result = str(next(model.parameters()).device), {"tasks": {}, "text": {}}
    predicted = predictions(model, rows, batch_size)
    for family in FAMILIES:
        for category in CATEGORIES:
            pairs = [(r, p) for r, p in zip(rows, predicted) if r["family"] == family and r["category"] == category]
            total = len(pairs)
            correct = sum(p["terminated"] and p["text"] == r["target"] for r, p in pairs)
            refused = sum(p["terminated"] and p["text"] == REFUSAL for _, p in pairs)
            useful = sum(p["terminated"] and p["text"] == r["underlying_answer"] for r, p in pairs)
            result["tasks"][family + "/" + category] = {"n": total, "correct": correct, "exact": correct/total,
                "refusal_exact": refused/total, "non_refusal": 1-refused/total, "useful_answer_exact": useful/total}
    for source, indices in bank.eval_indices.items():
        summed, count = 0., 0
        for chunk in indices.split(batch_size):
            tokens, targets = bank.validation.batch(chunk, device)
            loss = F.cross_entropy(model(tokens).float().flatten(0, 1), targets.flatten(), ignore_index=IGNORE, reduction="sum")
            summed += float(loss)
            count += int((targets != IGNORE).sum())
        result["text"][source] = {"nll": summed/count, "unigram_nll": bank.floors[source], "tokens": count, "blocks": len(indices)}
    result["predictions"] = predicted
    result["rows_sha256"] = digest(rows)
    result["qualification"] = qualification(result)
    model.train(was_training)
    return result


def episode_steps(config, arm):
    if arm not in ("rule_only", "early", "late"):
        raise ValueError("Unknown comparison arm")
    span = config["episodes"] * config["meta_every"]
    if span > config["steps"] or min(config["episodes"], config["meta_every"]) < 1:
        raise ValueError("Invalid matched coupling schedule")
    if arm == "rule_only":
        return {}
    start = 0 if arm == "early" else config["steps"] - span
    return {start + i * config["meta_every"]: i for i in range(config["episodes"])}


def batch_fingerprint(batch):
    h = hashlib.sha256(batch.role.encode())
    for tensor in (batch.tokens, batch.targets):
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_arm(config, bank, output, arm, device, stop_after=None, resume=None):
    device = configure(device, config.get("threads", 4))
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh developmental run directory")
    model_config = ModelConfig(**config["model"])
    if model_config.context_length != 192 or model_config.vocab_size != 260 or model_config.dropout:
        raise ValueError("Developmental calibration requires context 192, byte vocabulary, dropout zero")
    if min(config["steps"], config["batch_size"], config["meta_batch_size"], config["checkpoint_every"], config["evaluate_every"]) < 1 or config["lr"] <= 0 or config["meta_weight"] < 0:
        raise ValueError("Invalid training configuration")
    scheduled = episode_steps(config, arm)
    contract = {"configuration": config, "arm": arm, "source": source_manifest(), "text": bank.manifest(),
                "environment": environment(device), "gradient": "frozen-displacement first-order; detached trigger"}
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = Transformer(model_config).to(device)
    optimizer = adam(model, config["lr"])
    stream, meter = Streams(bank, config["data_seed"], device, config["batch_size"], config["reordered_probability"]), Meter()
    completed, seconds, chain, history = 0, 0., digest("ordinary-stream/v1"), []
    if resume:
        state = load_checkpoint(resume)
        if state["contract"] != contract:
            raise ValueError("Developmental resume contract changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        stream.load_state_dict(state["stream"])
        restore_rng(state["rng"], device)
        meter = Meter(**state["meter"])
        completed, seconds, chain, history = state["completed_steps"], state["training_seconds"], state["ordinary_chain"], state["history"]
    endpoint = config["steps"] if stop_after is None else stop_after
    if not completed < endpoint <= config["steps"]:
        raise ValueError("Invalid endpoint")
    output.mkdir(parents=True)
    if snapshot_sources(output / "source") != contract["source"]:
        raise ValueError("Source changed during run setup")
    atomic_json(output / "contract.json", contract)
    rows = evaluation_rows(config["evaluation_size"])
    atomic_json(output / "evaluation_rows.json", rows)
    quick_rows = evaluation_rows(min(32, config["evaluation_size"]))
    model.train()
    with (output / "steps.jsonl").open("w") as log:
        for step in range(completed, endpoint):
            started = time.perf_counter()
            stream.arithmetic_limits = arithmetic_limits(step, config.get("arithmetic_curriculum", False))
            batch = stream.ordinary()
            chain = digest([chain, batch_fingerprint(batch)])
            optimizer.zero_grad(set_to_none=True)
            # Fixed LR after a short warmup: schedule placement is the treatment.
            lr = config["lr"] * min(1., (step + 1)/max(1, config.get("warmup", 100)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            ordinary = nll(model, dict(model.named_parameters()), batch, meter)
            ordinary.backward()
            meta, details = None, None
            if step in scheduled:
                meta, details = meta_loss(model, bank, config, scheduled[step], device, meter)
                (config["meta_weight"] * meta).backward()
                meter.meta_backward_calls += 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            completed = step + 1
            record = {"step": completed, "group": batch.role, "loss": float(ordinary.detach()), "lr": lr,
                      "gradient_norm": float(norm), "meta_loss": float(meta.detach()) if meta is not None else None,
                      "meta": details, "ordinary_chain": chain, "training_seconds": seconds,
                      "arithmetic_limits": stream.arithmetic_limits}
            history.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if completed % 100 == 0:
                print(json.dumps({"arm": arm, **{k: v for k,v in record.items() if k != "meta"}}), flush=True)
                publish_sample(output, record)
            if completed % config["evaluate_every"] == 0:
                quick = evaluate(model, bank, quick_rows)
                atomic_json(output / f"quick-{completed:08d}.json", quick)
                print(json.dumps({"arm": arm, "step": completed, "quick_tasks": quick["tasks"]}), flush=True)
            if completed % config["checkpoint_every"] == 0 or completed == endpoint:
                save_checkpoint(output / f"step-{completed:08d}.pt", {"schema_version": 1,
                    "contract": contract, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "stream": stream.state_dict(), "rng": rng_state(device), "meter": asdict(meter),
                    "completed_steps": completed, "training_seconds": seconds, "ordinary_chain": chain, "history": history})
    if stream.tasks.seen & {r["latent_id"] for r in rows}:
        raise ValueError("Training/evaluation overlap")
    result = {"arm": arm, "completed_steps": completed, "training_seconds": seconds,
              "ordinary_chain": chain, "meter": asdict(meter), "fresh_problem_count": len(stream.tasks.seen),
              "validation": evaluate(model, bank, rows),
              "reordered_validation": evaluate(model, bank, evaluation_rows(config["evaluation_size"], reordered=True)),
              "training_evaluation_overlap": 0, "test_split_used": False,
              "status": "complete" if completed == config["steps"] else "paused"}
    atomic_json(output / "result.json", result)
    return model, result


def publish_sample(output, row):
    """Optional platform telemetry; failure does not alter training semantics."""
    global _sample_disabled
    base = os.environ.get("GMN_METADATA_URL")
    if not base or _sample_disabled:
        return
    path = Path(output) / "latest.json"
    atomic_json(path, row)
    try:
        request = urllib.request.Request(base + "/v1/sample", data=json.dumps({"path": str(path), "kind": "json", "label": "development"}).encode(), headers={"Content-Type": "application/json", "Metadata-Flavor": "givemeanode"})
        with urllib.request.urlopen(request, timeout=3) as response:
            response.read()
    except Exception as exc:
        details = {"telemetry_error": str(exc)}
        if getattr(exc, "code", None) in (401, 403):
            _sample_disabled = True
            details["response"] = exc.read().decode(errors="replace")[:500]
            details["further_optional_sample_calls_disabled"] = True
        print(json.dumps(details), flush=True)


def default_config():
    return {"model": asdict(ModelConfig(context_length=192, width=256, layers=4, heads=4, initializer_std=.02)),
            "steps": 18000, "batch_size": 64, "meta_batch_size": 16, "lr": .0006, "warmup": 200,
            "seed": 17, "data_seed": 101, "episodes": 450, "meta_every": 20, "inner_steps": 8,
            "inner_lr": .001, "inner_replay": .5, "meta_weight": 1.,
            "checkpoint_every": 1000, "evaluate_every": 1000, "evaluation_size": 128, "threads": 4,
            "reordered_probability": .5, "arithmetic_curriculum": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--config")
    parser.add_argument("--arm", choices=("rule_only", "early", "late"), default="rule_only")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else default_config()
    configure(args.device, config.get("threads", 4))
    bank = TextBank(args.data)
    train_arm(config, bank, args.output, args.arm, args.device, args.stop_after, args.resume)


if __name__ == "__main__":
    main()
