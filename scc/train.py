"""Single-process reference trainer. Default runs are small, local CPU checks."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import torch
from torch.nn import functional as F

from .checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from .data import IGNORE, PreparedDataset
from .model import ModelConfig, Transformer
from .provenance import atomic_json, digest, source_manifest, snapshot_sources
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig
    steps: int = 20
    batch_size: int = 4
    learning_rate: float = 0.001
    min_lr_ratio: float = 0.1
    warmup_steps: int = 2
    weight_decay: float = 0.1
    clip_grad_norm: float = 1.0
    seed: int = 17
    data_seed: int = 101
    checkpoint_every: int = 10
    precision: str = "fp32"
    deterministic: bool = True
    group_weights: dict | None = None
    trim_padding: bool = True

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.checkpoint_every) <= 0:
            raise ValueError("steps, batch_size and checkpoint_every must be positive")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if self.learning_rate <= 0 or self.clip_grad_norm <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.min_lr_ratio <= 1 or self.precision not in ("fp32", "bf16"):
            raise ValueError("Invalid schedule or precision")
        if self.group_weights is not None and (not self.group_weights or any(v <= 0 for v in self.group_weights.values())):
            raise ValueError("Group sampling weights must be positive")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["model"] = ModelConfig(**value["model"])
        return cls(**value)


class ShuffledBatches:
    def __init__(self, size, seed):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.cursor = 0
        self.epochs = 0

    def next(self, batch_size):
        chunks = []
        remaining = batch_size
        while remaining:
            if self.cursor == self.size:
                self.order = torch.randperm(self.size, generator=self.generator)
                self.cursor = 0
                self.epochs += 1
            take = min(remaining, self.size - self.cursor)
            chunks.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            remaining -= take
        return torch.cat(chunks)

    def state_dict(self):
        return {"size": self.size, "order": self.order, "cursor": self.cursor,
                "epochs": self.epochs, "rng": self.generator.get_state()}

    def load_state_dict(self, state):
        if state["size"] != self.size:
            raise ValueError("Sampler dataset size changed")
        self.order = state["order"]
        self.cursor, self.epochs = state["cursor"], state["epochs"]
        self.generator.set_state(state["rng"])


class GroupedBatches:
    """Sample a source per batch; count actual token exposure separately."""
    def __init__(self, dataset, weights, seed):
        self.names = sorted(weights)
        self.weights = torch.tensor([weights[name] for name in self.names], dtype=torch.float64)
        self.generator = torch.Generator().manual_seed(seed)
        self.indices, self.samplers = {}, {}
        for index, name in enumerate(self.names):
            if name not in dataset.group_names:
                raise ValueError(f"Unknown sampling group {name}")
            ids = torch.where(dataset.groups == dataset.group_names.index(name))[0]
            if not len(ids):
                raise ValueError(f"Empty training group {name}")
            self.indices[name] = ids
            self.samplers[name] = ShuffledBatches(len(ids), seed + 1000 + index)

    def next(self, batch_size):
        name = self.names[int(torch.multinomial(self.weights, 1, generator=self.generator))]
        return self.indices[name][self.samplers[name].next(batch_size)]

    def state_dict(self):
        return {"kind": "grouped", "rng": self.generator.get_state(),
                "samplers": {name: sampler.state_dict() for name, sampler in self.samplers.items()}}

    def load_state_dict(self, state):
        if state["kind"] != "grouped" or set(state["samplers"]) != set(self.names):
            raise ValueError("Grouped sampler changed")
        self.generator.set_state(state["rng"])
        for name, sampler in self.samplers.items():
            sampler.load_state_dict(state["samplers"][name])


def learning_rate(config, step):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps - 1)
    factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return config.learning_rate * factor


def train(config, data_directory, output_directory, device="cpu", resume=None, stop_after=None):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Reference training currently supports CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if config.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 training requires a supported CUDA device")
    if stop_after is not None and not 0 < stop_after <= config.steps:
        raise ValueError("stop_after must be in [1, steps]")
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory, including when resuming")
    dataset = PreparedDataset(data_directory, "train")
    if dataset.context_length != config.model.context_length:
        raise ValueError("Model and prepared-data context lengths differ")
    if config.model.vocab_size != dataset.tokenizer.vocab_size:
        raise ValueError("Model and tokenizer vocabularies differ")
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    model = Transformer(config.model).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=config.learning_rate, betas=(0.9, 0.95), foreach=False)
    batches = GroupedBatches(dataset, config.group_weights, config.data_seed) if config.group_weights else ShuffledBatches(len(dataset), config.data_seed)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "torch": str(torch.__version__), "device": str(device),
                   "cuda_runtime": torch.version.cuda, "threads": torch.get_num_threads(),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()}
    contract = {"config": asdict(config), "data_fingerprint": dataset.fingerprint,
                "source_files": source_manifest(), "environment": environment}
    history, completed, input_tokens, loss_tokens = [], 0, 0, 0
    compute_seconds = 0.0
    exposure = {name: {"input_tokens": 0, "loss_tokens": 0, "examples": 0} for name in dataset.group_names}
    if resume is not None:
        state = load_checkpoint(resume)
        if state["contract"] != contract:
            raise ValueError("Resume contract changed: config, data, source, or environment differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        batches.load_state_dict(state["sampler"])
        history = state["history"]
        completed, input_tokens, loss_tokens = state["completed_steps"], state["input_tokens"], state["loss_tokens"]
        compute_seconds = state["compute_seconds"]
        exposure = state.get("exposure", exposure)
        restore_rng(state["rng"], device)
    end_step = config.steps if stop_after is None else stop_after
    if end_step <= completed:
        raise ValueError("Requested endpoint must follow the resume checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    if snapshot_sources(output / "source") != contract["source_files"]:
        raise ValueError("Source changed during run initialization")
    run_manifest = {"schema_version": 1, "contract": contract,
                    "contract_sha256": digest(contract), "parameter_count": model.parameter_count(),
                    "resume_from": str(Path(resume).resolve()) if resume else None, "status": "running"}
    atomic_json(output / "run.json", run_manifest)
    model.train()
    for step in range(completed, end_step):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        indices = batches.next(config.batch_size)
        tokens, targets = dataset.batch(indices, device)
        if config.trim_padding:
            last = int(torch.where((targets != IGNORE).any(dim=0))[0][-1]) + 1
            tokens, targets = tokens[:, :last], targets[:, :last]
        valid = int((targets != IGNORE).sum())
        if valid == 0:
            raise ValueError("Batch contains no supervised tokens")
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if config.precision == "bf16" else nullcontext()
        with context:
            logits = model(tokens)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=IGNORE)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_seconds += time.perf_counter() - started
        input_tokens += int((tokens != ByteTokenizer.PAD).sum())
        loss_tokens += valid
        for offset, index in enumerate(indices.tolist()):
            group = dataset.group_names[int(dataset.groups[index])]
            exposure[group]["input_tokens"] += int((tokens[offset] != ByteTokenizer.PAD).sum())
            exposure[group]["loss_tokens"] += int((targets[offset] != IGNORE).sum())
            exposure[group]["examples"] += 1
        completed = step + 1
        record = {"step": completed, "loss": float(loss.detach()), "learning_rate": lr,
                  "gradient_norm": float(grad_norm), "input_tokens": input_tokens,
                  "loss_tokens": loss_tokens}
        history.append(record)
        print(json.dumps(record), flush=True)
        if completed % config.checkpoint_every == 0 or completed == end_step:
            state = {"schema_version": 1, "contract": contract, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "sampler": batches.state_dict(),
                     "rng": rng_state(device), "completed_steps": completed,
                     "input_tokens": input_tokens, "loss_tokens": loss_tokens,
                     "compute_seconds": compute_seconds, "history": history}
            state["exposure"] = exposure
            save_checkpoint(output / f"step-{completed:08d}.pt", state)
            atomic_json(output / "metrics.json", history)
    atomic_json(output / "result.json", {"completed_steps": completed,
                 "status": "complete" if completed == config.steps else "paused",
                 "input_tokens": input_tokens, "loss_tokens": loss_tokens,
                 "compute_seconds": compute_seconds, "parameter_count": model.parameter_count(),
                 "exposure": exposure,
                 "timing_scope": "training steps including batch loading; excludes setup and checkpoints"})
    run_manifest["status"] = "complete" if completed == config.steps else "paused"
    atomic_json(output / "run.json", run_manifest)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    train(TrainConfig.from_dict(json.loads(args.config.read_text())), args.data, args.output,
          args.device, args.resume, args.stop_after)
