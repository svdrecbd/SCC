"""Local controlled learnability runs using fresh counterfactual task batches."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import time

import torch
from torch.nn import functional as F

from .checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from .data import IGNORE, PreparedDataset
from .evaluate import evaluate_loss, generate, score_predictions
from .model import ModelConfig, Transformer
from .online_tasks import TableStream, TaskSpec, encode_batch, evaluation_records
from .provenance import atomic_json, digest, file_digest, snapshot_sources, source_manifest
from .qualify import evaluation_subset
from .tokenizer import ByteTokenizer, load_tokenizer
from .train import GroupedBatches, TrainConfig, learning_rate


def lineage_table_ids(state):
    """Recover all trained tables, including parents written before lineage tracking."""
    identities, visited = set(), set()
    while True:
        identities.update(state["table_stream"]["seen"])
        parent = state["contract"].get("initialization")
        if parent is None:
            return identities
        path = Path(parent["checkpoint"])
        if str(path.resolve()) in visited:
            raise ValueError("Checkpoint ancestry contains a cycle")
        visited.add(str(path.resolve()))
        if file_digest(path) != parent["sha256"]:
            raise ValueError("Ancestral checkpoint checksum mismatch")
        state = load_checkpoint(path)


@torch.no_grad()
def behavior(model, rows, tokenizer):
    predictions = [generate(model, row["prompt"], tokenizer=tokenizer, max_new_tokens=32) for row in rows]
    scores = score_predictions(rows, predictions)
    tables = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        tables[(row["category"], row["latent_id"])].append(prediction["terminated"] and prediction["text"] == row["target"])
    for category, value in scores.items():
        matched = [correct for (cat, _), correct in tables.items() if cat == category]
        value["tables"] = len(matched)
        value["all_queries_exact"] = sum(all(correct) for correct in matched) / len(matched)
    return {"scores": scores, "predictions": [{"category": row["category"], "latent_id": row["latent_id"],
             "prompt_sha256": digest(row["prompt"]), "target": row["target"], **prediction}
            for row, prediction in zip(rows, predictions)]}


def run(configuration, output, stop_after=None, resume=None):
    if configuration.get("device", "cpu") != "cpu":
        raise ValueError("The online curriculum runner supports CPU only; GPU execution is not implemented")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh online experiment directory")
    config = TrainConfig.from_dict(configuration["training"])
    if config.precision != "fp32" or config.batch_size % 4:
        raise ValueError("Local online controls require fp32 and a batch size divisible by four")
    spec = TaskSpec(**configuration["task"])
    weights = config.group_weights or {"retrieval": 1.0}
    task_names = sorted(set(weights) & {"retrieval", "authorized", "unauthorized", "permission"})
    if not task_names or ("permission" in task_names and config.batch_size % 8):
        raise ValueError("Task training is required; paired permissions require batch sizes divisible by eight")
    evaluation_categories = sorted((set(task_names) - {"permission"}) | ({"authorized", "unauthorized"} if "permission" in task_names else set()))
    natural_weights = {k: v for k, v in weights.items() if k not in task_names}
    tokenizer_dir = configuration.get("tokenizer")
    tokenizer = load_tokenizer(tokenizer_dir) if tokenizer_dir else ByteTokenizer()
    if tokenizer.vocab_size != config.model.vocab_size:
        raise ValueError("Tokenizer and model vocabularies differ")
    natural = PreparedDataset(configuration["natural_data"], "train") if natural_weights else None
    natural_batches = GroupedBatches(natural, natural_weights, config.data_seed + 1000) if natural else None
    if natural and (natural.context_length != config.model.context_length or natural.tokenizer.manifest() != tokenizer.manifest()):
        raise ValueError("Natural data and online tasks must share context and tokenizer")
    threads = configuration.get("threads", 4)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(config.deterministic)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = Transformer(config.model)
    initialization = None
    ancestral_tables = set()
    if configuration.get("initialize_from"):
        initial_path = Path(configuration["initialize_from"])
        previous = load_checkpoint(initial_path)
        ancestral_tables = lineage_table_ids(previous)
        previous_model = ModelConfig(**previous["contract"]["configuration"]["training"]["model"])
        if previous_model != config.model:
            raise ValueError("Curriculum initialization requires the same model architecture")
        if not resume:
            model.load_state_dict(previous["model"])
        initialization = {"checkpoint": str(initial_path.resolve()), "sha256": file_digest(initial_path),
                          "optimizer_reset": True, "scope": "Curriculum transfer from our own trained checkpoint"}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": config.weight_decay},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(0.9, 0.95), foreach=False)
    stream = TableStream(spec, config.data_seed + 1)
    choice = torch.Generator().manual_seed(config.data_seed)
    task_probability = sum(weights[k] for k in task_names) / sum(weights.values())
    category_weights = torch.tensor([weights[k] for k in task_names], dtype=torch.float64)
    contract = {"configuration": configuration, "tokenizer": tokenizer.manifest(), "source_files": source_manifest(),
                "initialization": initialization,
                "natural_data_fingerprint": natural.fingerprint if natural else None,
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "platform": platform.platform(), "threads": threads, "device": "cpu"}}
    completed, history, elapsed = 0, [], 0.0
    exposure = {name: {"examples": 0, "input_tokens": 0, "loss_tokens": 0} for name in set(evaluation_categories) | set(natural_weights)}
    if resume:
        state = load_checkpoint(resume)
        if state["contract"] != contract:
            raise ValueError("Online resume contract changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        stream.load_state_dict(state["table_stream"])
        choice.set_state(state["choice_rng"])
        if natural_batches:
            natural_batches.load_state_dict(state["natural_sampler"])
        restore_rng(state["rng"], "cpu")
        completed, history, elapsed, exposure = state["completed_steps"], state["history"], state["compute_seconds"], state["exposure"]
    end_step = stop_after or config.steps
    if not completed < end_step <= config.steps:
        raise ValueError("Invalid online run endpoint")
    output.mkdir(parents=True)
    if snapshot_sources(output / "source") != contract["source_files"]:
        raise ValueError("Source changed during snapshot")
    validation = evaluation_records(spec, configuration.get("evaluation_tables", 128), categories=evaluation_categories)
    reordered = evaluation_records(spec, configuration.get("evaluation_tables", 128), categories=evaluation_categories, reordered=True)
    atomic_json(output / "protocol.json", {"contract": contract, "contract_sha256": digest(contract),
                "evaluation_sha256": digest(validation), "reordered_evaluation_sha256": digest(reordered),
                "threshold": {"example_exact_match": .95, "table_all_queries_exact": .90},
                "test_split_used": False, "scope": "Development gates; IID tables plus reordered versions, not final adversarial holdouts"})
    atomic_json(output / "evaluation_records.json", validation)
    atomic_json(output / "reordered_evaluation_records.json", reordered)
    metrics_path = output / "steps.jsonl"
    natural_validation = None
    if natural:
        natural_validation = PreparedDataset(configuration["natural_data"], "validation")
        allowed_groups = {natural_validation.group_names.index(name) for name in natural_weights}
        selected = torch.tensor([index for index, group in enumerate(natural_validation.groups.tolist()) if group in allowed_groups])
        natural_validation.arrays = [array[selected] for array in natural_validation.arrays]
        natural_validation.groups = natural_validation.groups[selected]
        indices = evaluation_subset(natural_validation, 64)
        atomic_json(output / "language_before.json", {"indices": indices, "loss": evaluate_loss(model, natural_validation, 16)})
    model.train()
    with metrics_path.open("w") as log:
        for step in range(completed, end_step):
            started = time.perf_counter()
            if float(torch.rand((), generator=choice)) < task_probability:
                category = task_names[int(torch.multinomial(category_weights, 1, generator=choice))]
                rows = stream.batch(config.batch_size, category)
                tokens, targets = encode_batch(rows, tokenizer, config.model.context_length)
                batch_categories = [row["category"] for row in rows]
            else:
                indices = natural_batches.next(config.batch_size)
                category = natural.group_names[int(natural.groups[indices[0]])]
                tokens, targets = natural.batch(indices)
                last = int(torch.where((targets != IGNORE).any(dim=0))[0][-1]) + 1
                tokens, targets = tokens[:, :last], targets[:, :last]
                batch_categories = [category] * config.batch_size
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            logits = model(tokens)
            token_losses = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=IGNORE,
                                          reduction="none").view_as(targets)
            loss_weights = (targets != IGNORE).float()
            if category in task_names:
                eos_weight = configuration.get("eos_loss_weight", 1.0)
                if not 0 < eos_weight <= 1:
                    raise ValueError("EOS weight must be in (0, 1]")
                loss_weights[targets == tokenizer.EOS] *= eos_weight
            loss = (token_losses * loss_weights).sum() / loss_weights.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite online loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            elapsed += time.perf_counter() - started
            for index, name in enumerate(batch_categories):
                exposure[name]["examples"] += 1
                exposure[name]["input_tokens"] += int((tokens[index] != tokenizer.PAD).sum())
                exposure[name]["loss_tokens"] += int((targets[index] != IGNORE).sum())
            completed = step + 1
            record = {"step": completed, "group": category, "loss": float(loss.detach()),
                      "gradient_norm": float(norm), "fresh_tables": len(stream.seen), "learning_rate": lr}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if completed % configuration.get("evaluate_every", 500) == 0:
                small_ids = {r["latent_id"] for r in validation[:32 * len(evaluation_categories)]}
                small = [row for row in validation if row["latent_id"] in small_ids]
                record["validation"] = behavior(model, small, tokenizer)["scores"]
                print(json.dumps(record), flush=True)
            history.append(record)
            if completed % config.checkpoint_every == 0 or completed == end_step:
                save_checkpoint(output / f"step-{completed:08d}.pt", {"schema_version": 1,
                    "contract": contract, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "table_stream": stream.state_dict(), "choice_rng": choice.get_state(),
                    "ancestral_training_tables": sorted(ancestral_tables),
                    "natural_sampler": natural_batches.state_dict() if natural_batches else None,
                    "rng": rng_state("cpu"), "completed_steps": completed, "history": history,
                    "compute_seconds": elapsed, "exposure": exposure})
    scores = behavior(model, validation, tokenizer)
    reordered_scores = behavior(model, reordered, tokenizer)
    passed = all(value["exact_match"] >= .95 and value["all_queries_exact"] >= .90
                 for value in list(scores["scores"].values()) + list(reordered_scores["scores"].values()))
    validation_ids = {row["latent_id"] for row in validation}
    if validation_ids & (stream.seen | ancestral_tables):
        raise ValueError("Training table leaked into evaluation")
    result = {"status": "complete" if completed == config.steps else "paused", "development_gate_passed": passed,
              "task": asdict(spec), "initialization": initialization,
              "completed_steps": completed, "parameter_count": model.parameter_count(), "compute_seconds": elapsed,
              "fresh_training_tables": len(stream.seen), "exposure": exposure,
              "ancestor_training_tables": len(ancestral_tables),
              "training_tables_seen_in_ancestors": len(stream.seen & ancestral_tables),
              "training_tables_new_to_lineage": len(stream.seen - ancestral_tables),
              "validation": scores, "reordered_validation": reordered_scores,
              "training_table_ids_sha256": digest(sorted(stream.seen)), "training_evaluation_table_overlap": 0,
              "test_split_used": False, "resume_from": str(resume) if resume else None,
              "timing_scope": "Batch generation/loading and training; excludes evaluation and checkpoint writes"}
    if natural_validation:
        result["language_after"] = evaluate_loss(model, natural_validation, 16)
    atomic_json(output / "result.json", result)
    atomic_json(output / "history.json", history)
    print(json.dumps({key: result[key] for key in ("status", "development_gate_passed", "completed_steps", "fresh_training_tables")}
                     | {"validation": scores["scores"], "reordered": reordered_scores["scores"]}, indent=2), flush=True)
    return model, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output, args.stop_after, args.resume)
