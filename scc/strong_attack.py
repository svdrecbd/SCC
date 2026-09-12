"""Full removal/recovery inner attacks and an explicit first-order approximation."""

import copy
import hashlib
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .coupling import nll
from .interventions import batch_digest


def straight_through_parameters(parameters, attacked):
    """Exact derivative of a frozen-displacement surrogate, not of attack(theta)."""
    if parameters.keys() != attacked.keys():
        raise ValueError("Attack parameters do not match the defender")
    return {key: value + (attacked[key].detach() - value.detach()) for key, value in parameters.items()}


def rollout(model, streams, stages, size, normalizer, meter=None, callback=None):
    """Fresh model, fresh optimizer per stage; no retained inner derivative graph."""
    if not stages or normalizer <= 0:
        raise ValueError("Invalid strong attack")
    attacked = copy.deepcopy(model)
    attacked.train()
    for value in attacked.parameters():
        value.requires_grad_(True)
    records, total_steps, elapsed = [], 0, 0.
    for index, stage in enumerate(stages):
        if stage["steps"] <= 0 or stage["learning_rate"] <= 0 or stage["preservation_weight"] < 0:
            raise ValueError("Invalid attack stage")
        optimizer = torch.optim.AdamW(attacked.parameters(), lr=stage["learning_rate"],
                                      betas=(.9, .95), weight_decay=0., foreach=False)
        rolling_hash = hashlib.sha256()
        for step in range(stage["steps"]):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            parameters = dict(attacked.named_parameters())
            primary = streams.task(size, "unauthorized", "disclose", "strong/disclose")
            batches = [primary]
            with sdpa_kernel(SDPBackend.MATH):
                loss = nll(attacked, parameters, primary, meter)
                alpha = stage["preservation_weight"]
                if alpha:
                    replay = [streams.task(size, "retrieval", role="strong/retrieval"),
                              streams.task(size, "authorized", role="strong/authorized"),
                              streams.text(size, "strong/language")]
                    batches.extend(replay)
                    utility = (nll(attacked, parameters, replay[0], meter) +
                               nll(attacked, parameters, replay[1], meter) +
                               nll(attacked, parameters, replay[2], meter) / normalizer) / 3
                    loss = (loss + alpha * utility) / (1 + alpha)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite strong attack loss")
                loss.backward()
            if meter is not None:
                meter.first_derivative_calls += 1
            norm = torch.nn.utils.clip_grad_norm_(attacked.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            for batch in batches:
                rolling_hash.update(batch_digest(batch).encode())
            elapsed += time.perf_counter() - started
            total_steps += 1
        row = {"stage": index, "stage_steps": stage["steps"], "total_steps": total_steps,
               "last_loss": float(loss.detach()), "last_gradient_norm": float(norm),
               "training_seconds": elapsed, "batches_sha256": rolling_hash.hexdigest()}
        records.append(row)
        if callback is not None:
            callback(attacked, row)
    return attacked, {"stages": records, "training_seconds": elapsed, "optimizer_steps": total_steps,
                     "gradient_semantics": "First-order identity-Jacobian approximation; inner trajectory is detached"}
