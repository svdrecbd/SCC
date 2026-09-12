"""Bounded GPU readiness and gated developmental experiments."""

import argparse
import copy
import json
from pathlib import Path
import os
import time

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .coupling import Meter, nll
from .developmental_metrics import compare_collapse
from .developmental_run import (TextBank, Streams, configure, default_config, environment,
    evaluate, meta_loss, modify, predictions, train_arm)
from .developmental_tasks import batch_rows, evaluation_rows
from .model import ModelConfig, Transformer
from .provenance import atomic_json, digest, file_digest, snapshot_sources


def readiness(bank, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(output / "source")
    device = configure("cuda")
    torch.manual_seed(291)
    cpu = Transformer(ModelConfig(**default_config()["model"])).eval()
    gpu = copy.deepcopy(cpu).cuda()
    stream = Streams(bank, 292, "cpu", 4)
    batch = stream.ordinary()
    loss_cpu = nll(cpu, dict(cpu.named_parameters()), batch)
    loss_gpu = nll(gpu, dict(gpu.named_parameters()), type(batch)(batch.tokens.cuda(), batch.targets.cuda(), batch.role))
    loss_cpu.backward()
    loss_gpu.backward()
    torch.testing.assert_close(loss_cpu, loss_gpu.cpu(), atol=2e-5, rtol=2e-4)
    maximum_gradient_error = 0.
    for a, b in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(a.grad, b.grad.cpu(), atol=2e-5, rtol=2e-3)
        maximum_gradient_error = max(maximum_gradient_error, float((a.grad-b.grad.cpu()).abs().max()))
    config = default_config()
    config.update(steps=4, batch_size=2, meta_batch_size=2, episodes=2, meta_every=2, inner_steps=1,
                  evaluation_size=2, checkpoint_every=2, evaluate_every=100)
    config["model"].update(width=32, layers=1, heads=2)
    train_arm(config, bank, output / "full", "early", device)
    train_arm(config, bank, output / "part", "early", device, stop_after=2)
    resumed_model, _ = train_arm(config, bank, output / "resumed", "early", device, resume=output / "part/step-00000002.pt")
    full = load_checkpoint(output / "full/step-00000004.pt")
    resumed = load_checkpoint(output / "resumed/step-00000004.pt")
    for key in full["model"]:
        if not torch.equal(full["model"][key], resumed["model"][key]):
            raise AssertionError("CUDA resume changed tensor: " + key)
    if full["ordinary_chain"] != resumed["ordinary_chain"] or full["meter"] != resumed["meter"]:
        raise AssertionError("CUDA resume changed streams or exposure accounting")
    rows = evaluation_rows(2)
    if predictions(resumed_model, rows, 1) != predictions(resumed_model, rows, 64):
        raise AssertionError("Batched generation changed a greedy prediction")
    # Throughput includes data movement and exact source-role accounting.
    config = default_config()
    model = Transformer(ModelConfig(**config["model"])).cuda().train()
    stream = Streams(bank, 718, device, config["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = nll(model, dict(model.named_parameters()), stream.ordinary())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    meter = Meter()
    optimizer.zero_grad(set_to_none=True)
    start_meta = time.perf_counter()
    value, details = meta_loss(model, bank, config, 0, device, meter)
    value.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError("CUDA meta gradients are absent/nonfinite")
    torch.cuda.synchronize()
    result = {"ok": True, "environment": environment(device), "parameters": model.parameter_count(),
        "cpu_cuda_loss_absolute_error": abs(float(loss_cpu.detach())-float(loss_gpu.detach())),
        "cpu_cuda_gradient_max_absolute_error": maximum_gradient_error,
        "cuda_resume_bitwise": True, "serial_batched_greedy_agree": True,
        "ordinary_steps_per_second": 100/elapsed, "meta_episode_seconds": time.perf_counter()-start_meta,
        "meta_loss": float(value.detach()), "meta_details": details,
        "meta_gradient_norm": float(torch.stack([p.grad.norm() for p in model.parameters()]).norm()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "evidence_class": "Implementation validation; not a coupling result"}
    atomic_json(output / "readiness.json", result)
    return result


def probes(model, bank, config, output, untrained):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device, rows = str(next(model.parameters()).device), evaluation_rows(config["evaluation_size"])
    reordered_rows = evaluation_rows(config["evaluation_size"], reordered=True)
    clean = evaluate(model, bank, rows)
    clean_reordered = evaluate(model, bank, reordered_rows)
    atomic_json(output / "clean.json", clean)
    atomic_json(output / "clean-reordered.json", clean_reordered)
    if not (clean["qualification"]["passed"] and clean_reordered["qualification"]["passed"]):
        return {"status": "not_interpretable_clean_qualification_failed"}
    stream = Streams(bank, config["data_seed"] + 900000, device, config["batch_size"], config["reordered_probability"])
    records, attacked = [], model
    for index, (steps, replay) in enumerate(((100, .5), (200, .5), (1000, 3.))):
        attacked = modify(attacked, stream, steps, .001, replay)
        # The optimizer resets at these explicit stage boundaries.
        after = evaluate(attacked, bank, rows)
        after_reordered = evaluate(attacked, bank, reordered_rows)
        record = {"stage": index, "steps": steps, "replay_weight": replay,
                  "evaluation": after, "reordered_evaluation": after_reordered,
                  "collapse": compare_collapse(clean, after, untrained["validation"]),
                  "reordered_collapse": compare_collapse(clean_reordered, after_reordered, untrained["reordered"])}
        record["collapse_in_both_renderings"] = record["collapse"]["measured_suite_collapse"] and record["reordered_collapse"]["measured_suite_collapse"]
        atomic_json(output / f"stage-{index}.json", record)
        save_checkpoint(output / f"stage-{index}.pt", {"schema_version": 1, "model": attacked.state_dict(),
            "stage": index, "steps": steps, "stream": stream.state_dict(), "configuration": config})
        records.append(record)
    norm = torch.stack([(a.detach()-b.detach()).square().sum() for a,b in zip(attacked.parameters(), model.parameters())]).sum().sqrt()
    random_control = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(812)
    noise = [torch.randn(p.shape, generator=gen).to(device) for p in model.parameters()]
    noise_norm = torch.stack([n.square().sum() for n in noise]).sum().sqrt()
    with torch.no_grad():
        for p, n in zip(random_control.parameters(), noise):
            p.add_(n * (norm/noise_norm))
    damaged = evaluate(random_control, bank, rows)
    atomic_json(output / "matched_norm_damage.json", {"norm": float(norm), "evaluation": damaged,
        "collapse": compare_collapse(clean, damaged, untrained["validation"])})
    # A benign continued-learning edit uses the original policy targets.
    benign = copy.deepcopy(model).train()
    optimizer = torch.optim.AdamW(benign.parameters(), lr=.0003, foreach=False)
    benign_stream = Streams(bank, 771000, device, config["batch_size"], config["reordered_probability"])
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = nll(benign, dict(benign.named_parameters()), benign_stream.ordinary())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(benign.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    atomic_json(output / "benign_continued_learning.json", evaluate(benign, bank, rows))
    attacked.load_state_dict(model.state_dict())
    restored = evaluate(attacked, bank, rows)
    if restored != clean:
        raise AssertionError("Weight restoration did not reproduce intact evaluation")
    atomic_json(output / "weight_restoration.json", {"exact_evaluation_restoration": True})
    return {"status": "completed_bounded_optimizer_probes", "stages": len(records),
            "causal_internal_function_lesion_and_rescue": "not yet implemented"}


def experiment(bank, output, config, compare=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = configure("cuda", config.get("threads", 4))
    atomic_json(output / "configuration.json", config)
    torch.manual_seed(config["seed"])
    initial = Transformer(ModelConfig(**config["model"])).cuda()
    initial_eval = evaluate(initial, bank, evaluation_rows(config["evaluation_size"]))
    initial_reordered = evaluate(initial, bank, evaluation_rows(config["evaluation_size"], reordered=True))
    atomic_json(output / "untrained.json", initial_eval)
    atomic_json(output / "untrained-reordered.json", initial_reordered)
    untrained = {"validation": initial_eval, "reordered": initial_reordered}
    del initial
    model, control = train_arm(config, bank, output / "rule_only", "rule_only", device)
    qualified = control["validation"]["qualification"]["passed"] and control["reordered_validation"]["qualification"]["passed"]
    result = {"status": "control_qualified" if qualified else "control_learnability_failed",
              "control_qualified": qualified, "evidence_class": "Development calibration",
              "arms": {"rule_only": {"qualification": control["validation"]["qualification"],
                                      "ordinary_chain": control["ordinary_chain"]}},
              "test_split_used": False, "causal_mechanism_established": False}
    if compare and qualified:
        result["arms"]["rule_only"]["probes"] = probes(model, bank, config, output / "probes-rule_only", untrained)
        for arm in ("early", "late"):
            model, arm_result = train_arm(config, bank, output / arm, arm, device)
            if arm_result["ordinary_chain"] != control["ordinary_chain"]:
                raise AssertionError("Ordinary training streams differed between arms")
            result["arms"][arm] = {"qualification": arm_result["validation"]["qualification"],
                "ordinary_chain": arm_result["ordinary_chain"],
                "reordered_qualification": arm_result["reordered_validation"]["qualification"],
                "probes": probes(model, bank, config, output / ("probes-"+arm), untrained)}
        result["status"] = "bounded_comparison_completed"
    atomic_json(output / "experiment.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("readiness", "calibration", "comparison", "validated-comparison"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    configure("cuda")
    bank = TextBank(args.data, blocks=2 if args.mode == "readiness" else 128)
    config = json.loads(Path(args.config).read_text()) if args.config else default_config()
    if args.mode == "validated-comparison":
        checked = readiness(TextBank(args.data, blocks=2), Path(args.output) / "readiness")
        if not checked["ok"]:
            raise RuntimeError("CUDA readiness did not pass")
        result = {"readiness": checked, "experiment": experiment(bank, Path(args.output) / "comparison", config, True)}
    elif args.mode == "readiness":
        result = readiness(bank, args.output)
    else:
        result = experiment(bank, args.output, config, args.mode == "comparison")
    if os.environ.get("GMN_RESULT_PATH"):
        atomic_json(os.environ["GMN_RESULT_PATH"], result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
