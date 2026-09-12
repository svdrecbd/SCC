"""Atomic checkpoints containing tensors and weights-only-loadable primitives."""

import os
from pathlib import Path
import random
import tempfile

import torch


def rng_state(device):
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.device(device).type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return state
