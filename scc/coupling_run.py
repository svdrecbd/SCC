"""Local coupling discovery: weight probes, differentiable defense, and replay attacks."""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from .coupling import Meter, episode_loss, nll, post_attack_loss, task_batch
from .data import PreparedDataset
from .evaluate import score_predictions
from .interventions import (Evaluation, Streams, batch_digest, escape, generate_many,
                            load_model, parameter_change, parent_receipt, qualification, trained_table_ids)
from .provenance import atomic_json, digest, snapshot_sources, source_manifest
from .tokenizer import ByteTokenizer
from .strong_attack import rollout, straight_through_parameters
from .defense_objectives import (add_constraint_queries, per_constraint_escape_loss,
                                 query_language_references, select_attack_profile)


def setup(config, output):
    if config.get("device", "cpu") != "cpu" or config.get("precision", "fp32") != "fp32":
        raise ValueError("The coupling runner supports CPU fp32 only; GPU execution is not implemented")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh intervention directory")
    torch.set_num_threads(config.get("threads", 4))
    torch.use_deterministic_algorithms(True)
    random.seed(config.get("seed", 1))
    torch.manual_seed(config.get("seed", 1))
    model, parent = load_model(config["checkpoint"])
    evaluator = Evaluation(config["natural_data"], config.get("evaluation_tables", 16), config.get("evaluation_seed", 9159))
    if (evaluator.language.context_length != model.config.context_length or
            evaluator.language.tokenizer.manifest() != ByteTokenizer().manifest()):
        raise ValueError("Coupling tasks and natural data must share the model context and byte tokenizer")
    ancestors = trained_table_ids(parent)
    evaluator.check_training(ancestors)
    contract = {"configuration": config, "parent": parent_receipt(config["checkpoint"]),
                "reference_checkpoints": {key: parent_receipt(config[key])
                    for key in ("reference_checkpoint", "clean_reference_checkpoint") if config.get(key)},
                "evaluation": evaluator.manifest(), "source_files": source_manifest(),
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "platform": platform.platform(), "threads": torch.get_num_threads(), "device": "cpu"}}
    output.mkdir(parents=True)
    if snapshot_sources(output / "source") != contract["source_files"]:
        raise ValueError("Source changed during snapshot")
    atomic_json(output / "protocol.json", {"contract": contract, "contract_sha256": digest(contract),
                "test_split_used": False, "scope": "Exploratory local coupling discovery"})
    atomic_json(output / "evaluation_records.json", evaluator.rows)
    return output, model, evaluator, ancestors, contract


def save_state(path, model, contract, identities, **state):
    save_checkpoint(path, {"schema_version": 1, "model_config": asdict(model.config),
                          "model": model.state_dict(), "contract": contract,
                          "training_table_ids": sorted(identities), **state})


def probes(config, output):
    output, model, evaluator, ancestors, contract = setup(config, output)
    original = copy.deepcopy(model.state_dict())
    clean = evaluator(model)
    tokenizer = ByteTokenizer()
    token = tokenizer.encode("W")[0]
    results = []
    for index, scale in enumerate(config["row_scales"]):
        model.load_state_dict(original)
        started = time.perf_counter()
        with torch.no_grad():
            model.tokens.weight[token].mul_(scale)
        modification_seconds = time.perf_counter() - started
        value = evaluator(model)
        row = {"kind": "tied_embedding_refusal_start_row_scale", "scale": scale,
               "weight_edit": True, "optimizer_steps": 0, "modification_seconds": modification_seconds,
               **parameter_change(model, original), "evaluation": value,
               "escape": escape(value, clean, clean)}
        results.append(row)
        save_state(output / f"row-edit-{index}.pt", model, contract, ancestors,
                   completed_steps=0, edit={"token": token, "scale": scale})
    model.load_state_dict(original)
    predictions = generate_many(model, [r["prompt"] for r in evaluator.rows], suppress_token=token)
    decoding = {"kind": "suppress_refusal_start_token", "weight_edit": False,
                "scope": "Decoding-only diagnostic; outside core weight-modification threat model",
                "behavior": score_predictions(evaluator.rows, predictions), "predictions": predictions}
    result = {"clean": clean, "weight_probes": results, "decoding_probe": decoding,
              "test_split_used": False, "cloud_cost_usd": 0}
    atomic_json(output / "result.json", result)
    print(json.dumps({"weight_probes": [{"scale": r["scale"], "escape": r["escape"],
                      "behavior": r["evaluation"]["behavior"]} for r in results],
                      "decoding_probe": decoding["behavior"]}, indent=2), flush=True)
    return result


def run(config, output, stop_after=None, resume=None):
    invocation_started = time.perf_counter()
    kind, mode = config["kind"], config["mode"]
    if kind not in ("attack", "defense"):
        raise ValueError("Unknown intervention kind")
    if mode not in ({"disclose", "uppercase"} if kind == "attack" else {"control", "escape", "refusal"}):
        raise ValueError("Unknown intervention mode")
    if kind == "attack" and config.get("scope") not in ("all", "last_block"):
        raise ValueError("Unknown attack parameter scope")
    if config.get("inner_attack") not in (None, "full_adamw_first_order"):
        raise ValueError("Unknown inner attack; omit it for exact short SGD")
    if config.get("preserve_policy") and mode != "uppercase":
        raise ValueError("Policy-preserving replay is a benign-edit control")
    if config.get("outer_objective", "legacy") not in ("legacy", "metric_margin"):
        raise ValueError("Unknown outer objective")
    if config.get("outer_objective") == "metric_margin" and (kind != "defense" or mode != "escape" or
            config.get("inner_attack") != "full_adamw_first_order" or not config.get("constraint_queries")):
        raise ValueError("Metric objective requires strong escape defense and constraint queries")
    output, model, evaluator, ancestors, contract = setup(config, output)
    saved = load_checkpoint(resume) if resume else None
    if saved is not None and saved["contract"] != contract:
        raise ValueError("Intervention resume contract changed")
    natural = PreparedDataset(config["natural_data"], "train")
    outer = Streams(natural, config["data_seed"])
    meta = Streams(natural, config["data_seed"] + 100000)
    original = copy.deepcopy(model.state_dict())
    clean = evaluator(model, benign=mode == "uppercase")
    if config.get("reference_checkpoint"):
        reference_model, _ = load_model(config["reference_checkpoint"])
        if parent_receipt(config["reference_checkpoint"]) != contract["reference_checkpoints"]["reference_checkpoint"]:
            raise ValueError("Reference checkpoint changed during initialization")
        reference = evaluator(reference_model)
    else:
        reference = clean
        if config.get("outer_objective") == "metric_margin":
            reference_model = copy.deepcopy(model)
    if config.get("clean_reference_checkpoint"):
        retention_model, _ = load_model(config["clean_reference_checkpoint"])
        if parent_receipt(config["clean_reference_checkpoint"]) != contract["reference_checkpoints"]["clean_reference_checkpoint"]:
            raise ValueError("Clean reference checkpoint changed during initialization")
        retention_reference = evaluator(retention_model)
    else:
        retention_reference = clean
    normalizer = reference["language"]["nll_per_supervised_token"]
    atomic_json(output / "before.json", {"clean": clean, "reference": reference, "retention_reference": retention_reference,
                                        "qualification": qualification(clean, reference)})
    selected = []
    for name, parameter in model.named_parameters():
        active = kind == "defense" or config["scope"] == "all" or name.startswith(f"blocks.{model.config.layers - 1}.") or name.startswith("norm.")
        parameter.requires_grad_(active)
        if active:
            selected.append(parameter)
    optimizer = torch.optim.AdamW(selected, lr=config["learning_rate"], betas=(.9, .95),
                                  weight_decay=config.get("weight_decay", 0.), foreach=False)
    meter, history, elapsed, completed = Meter(), [], 0., 0
    points = [{"step": 0, "training_seconds": 0., "evaluation": clean, "escape": escape(clean, retention_reference, reference)}]
    if resume:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        outer.load_state_dict(saved["outer_stream"])
        meta.load_state_dict(saved["meta_stream"])
        meter = Meter(**saved["meter"])
        history, elapsed, completed, points = saved["history"], saved["training_seconds"], saved["completed_steps"], saved["points"]
        restore_rng(saved["rng"], "cpu")
    endpoint = stop_after or config["steps"]
    if not completed < endpoint <= config["steps"]:
        raise ValueError("Invalid intervention endpoint")
    model.train()
    with (output / "steps.jsonl").open("w") as log:
        for step in range(completed, endpoint):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            parameters = dict(model.named_parameters())
            record = {"step": step + 1}
            with sdpa_kernel(SDPBackend.MATH):
                if kind == "attack":
                    primary = outer.task(config["batch_size"], "retrieval" if mode == "uppercase" else "unauthorized",
                                         "uppercase" if mode == "uppercase" else "disclose", "attack/" + mode)
                    loss = nll(model, parameters, primary, meter)
                    alpha = config["preservation_weight"]
                    if alpha:
                        utility = (nll(model, parameters, outer.task(config["batch_size"], "retrieval", role="replay/retrieval"), meter) +
                                   nll(model, parameters, outer.task(config["batch_size"],
                                       "permission" if config.get("preserve_policy") else "authorized",
                                       role="replay/permission" if config.get("preserve_policy") else "replay/authorized"), meter) +
                                   nll(model, parameters, outer.text(config["batch_size"], "replay/language"), meter) / normalizer) / 3
                        loss = (loss + alpha * utility) / (1 + alpha)
                    record["primary_batch_sha256"] = batch_digest(primary)
                else:
                    ordinary = outer.ordinary(config["batch_size"])
                    loss = nll(model, parameters, ordinary, meter)
                    record["outer_batch_sha256"] = batch_digest(ordinary)
                    if mode != "control" and (step + 1) % config["couple_every"] == 0:
                        episodes = []
                        size, alpha = config["inner_batch_size"], config["inner_preservation_weight"]
                        strong = config.get("inner_attack") == "full_adamw_first_order"
                        if strong:
                            episode_index = (step + 1) // config["couple_every"] - 1
                            profile, stages = select_attack_profile(config, episode_index)
                            attacked_model, attack_diagnostics = rollout(model, meta, stages, size, normalizer, meter)
                            attack_diagnostics['profile'] = profile
                            record["strong_attack"] = attack_diagnostics
                            attacked = straight_through_parameters(parameters, dict(attacked_model.named_parameters()))
                        else:
                            for _ in range(config["inner_steps"]):
                                episodes.append([(meta.task(size, "unauthorized", "disclose", "inner/disclose"), 1.),
                                                 (meta.task(size, "retrieval", role="inner/retrieval"), alpha / 2),
                                                 (meta.text(size, "inner/language"), alpha / (2 * normalizer))])
                        size = config.get("query_batch_size", size)
                        rows = meta.tables.batch(size, "unauthorized")
                        retrieval_rows = meta.tables.batch(size, "retrieval")
                        query = {"disclose": task_batch(rows, ByteTokenizer(), model.config.context_length, "disclose", "query/disclose"),
                                 "policy": task_batch(rows, ByteTokenizer(), model.config.context_length, "policy", "query/policy"),
                                 "retrieval": task_batch(retrieval_rows, ByteTokenizer(), model.config.context_length, "policy", "query/retrieval"),
                                 "language": meta.text(size, "query/language")}
                        if config.get("constraint_queries"):
                            add_constraint_queries(query, meta, size, config.get("source_query_batch_size", 8))
                        meta_config = {"mode": mode, "inner_learning_rate": config["inner_learning_rate"],
                                       "language_normalizer": normalizer, "surrogate": config["surrogate"]}
                        if strong:
                            generated = generate_many(attacked_model, [r["prompt"] for r in rows + retrieval_rows])
                            attack_diagnostics["training_query_behavior"] = score_predictions(rows + retrieval_rows, generated)
                            attack_diagnostics["training_query_predictions"] = generated
                            attack_diagnostics["training_query_records"] = rows + retrieval_rows
                            if config.get("outer_objective") == "metric_margin":
                                language_references = query_language_references(model, reference_model, query, meter)
                                penalty, diagnostics = per_constraint_escape_loss(model, attacked, query, language_references, meter)
                            else:
                                penalty, diagnostics = post_attack_loss(model, attacked, query, meta_config, meter)
                            ordinary_gradients = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=True)
                            meta_gradients = torch.autograd.grad(penalty, tuple(parameters.values()), retain_graph=True)
                            ordinary_norm = sum(float(g.square().sum()) for g in ordinary_gradients) ** .5
                            meta_norm = sum(float(g.square().sum()) for g in meta_gradients) ** .5
                            diagnostics.update(ordinary_gradient_l2=ordinary_norm, meta_gradient_l2=meta_norm,
                                weighted_meta_gradient_l2=config["coupling_weight"] * meta_norm,
                                gradient_cosine=sum(float((a*b).sum()) for a,b in zip(ordinary_gradients,meta_gradients)) /
                                    max(ordinary_norm * meta_norm, 1e-30))
                            meter.first_derivative_calls += 2
                            del attacked_model
                        else:
                            penalty, diagnostics = episode_loss(model, parameters, episodes, query, meta_config, meter)
                        loss = loss + config["coupling_weight"] * penalty
                        record.update(diagnostics)
                        meter.meta_backward_calls += 1
                        if strong:
                            print(json.dumps({"outer_step": step + 1, "mode": mode,
                                              "strong_attack": {k:v for k,v in attack_diagnostics.items()
                                                                if k not in ("training_query_predictions", "training_query_records")},
                                              **diagnostics}), flush=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite intervention objective")
                loss.backward()
                meter.first_derivative_calls += 1
            norm = torch.nn.utils.clip_grad_norm_(selected, 1., error_if_nonfinite=True)
            optimizer.step()
            elapsed += time.perf_counter() - started
            completed = step + 1
            record.update(loss=float(loss.detach()), gradient_norm=float(norm))
            history.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if completed in config["evaluation_steps"] or completed == endpoint:
                identities = ancestors | outer.tables.seen | meta.tables.seen
                evaluator.check_training(identities)
                value = evaluator(model, benign=mode == "uppercase")
                point = {"step": completed, "training_seconds": elapsed, "meter": asdict(meter),
                         "evaluation": value, "escape": escape(value, retention_reference, reference),
                         **parameter_change(model, original)}
                if mode == "uppercase":
                    point["benign_success"] = value["benign_uppercase"]["exact_match"] >= .90 and qualification(value, clean)["qualified"]
                points.append(point)
                save_state(output / f"step-{completed:08d}.pt", model, contract, identities,
                    completed_steps=completed, optimizer=optimizer.state_dict(), outer_stream=outer.state_dict(),
                    meta_stream=meta.state_dict(), rng=rng_state("cpu"), meter=asdict(meter), history=history,
                    training_seconds=elapsed, points=points)
                print(json.dumps({"step": completed, "kind": kind, "mode": mode, "loss": record["loss"],
                                  "behavior": value["behavior"], "escape": point["escape"],
                                  "training_seconds": elapsed}), flush=True)
    result = {"status": "complete" if completed == config["steps"] else "paused", "kind": kind, "mode": mode,
              "completed_steps": completed, "training_seconds": elapsed,
              "evaluation_seconds": sum(p["evaluation"]["evaluation_seconds"] for p in points),
              "meter": asdict(meter), "trainable_parameters": sum(p.numel() for p in selected),
              "total_parameters": model.parameter_count(), "qualification": qualification(points[-1]["evaluation"], reference),
              "points": points, "first_observed_escape_step": next((p["step"] for p in points if p["escape"]), None),
              "test_split_used": False, "cloud_cost_usd": 0,
              "wall_seconds_this_invocation": time.perf_counter() - invocation_started,
              "resume_from": parent_receipt(resume) if resume else None,
              "cost_scope": "Training includes batch construction and updates; evaluation and checkpoint I/O are separate. No GPU extrapolation."}
    atomic_json(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text())
    if configuration["kind"] == "probes":
        probes(configuration, args.output)
    else:
        run(configuration, args.output, args.stop_after, args.resume)
