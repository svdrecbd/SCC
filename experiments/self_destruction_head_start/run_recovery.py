"""LN-395 stage 3: recovery of general capability from a starting model.

Starting points are a saved model directory or `random`, a fresh initialization
of the reference architecture. Training uses benign data only: packed generic
text (`--data generic`) or SEAM's Alpaca restoration protocol (`--data alpaca`).
Parameters are kept in float32 with bfloat16 autocast, so small updates are not
lost to bfloat16 rounding. The learning rate warms up linearly and then stays
constant, so every evaluation point is a valid endpoint for its budget.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def evaluate(model, tokenizer, validation, arguments):
    model.eval()
    record = {}
    length = arguments.sequence_length
    usable = min(len(validation), arguments.validation_tokens) // length
    total, rows_seen = 0.0, 0
    with torch.no_grad(), torch.autocast(arguments.device, dtype=torch.bfloat16):
        for first in range(0, usable, arguments.batch_size):
            rows = range(first, min(first + arguments.batch_size, usable))
            batch = np.stack([validation[r * length:(r + 1) * length] for r in rows])
            batch = torch.tensor(batch.astype(np.int64), device=arguments.device)
            total += model(input_ids=batch, labels=batch).loss.float().item() * len(rows)
            rows_seen += len(rows)
    record["validation_loss"] = total / rows_seen
    if arguments.benchmarks:
        if arguments.device == "cuda":
            torch.cuda.empty_cache()
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        wrapped = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=arguments.benchmark_batch_size)
        with torch.autocast(arguments.device, dtype=torch.bfloat16):
            results = lm_eval.simple_evaluate(model=wrapped, tasks=["arc_easy"], batch_size=arguments.benchmark_batch_size)["results"]
            record["arc_easy"] = results["arc_easy"]["acc,none"]
            results = lm_eval.simple_evaluate(model=wrapped, tasks=["mmlu"], limit=arguments.mmlu_limit,
                                              batch_size=arguments.benchmark_batch_size)["results"]
            record[f"mmlu_limit{arguments.mmlu_limit}"] = results["mmlu"]["acc,none"]
        del wrapped
    model.train()
    if arguments.device == "cuda":
        torch.cuda.empty_cache()
    return record


def generic_batches(arguments, generator):
    train = np.fromfile(arguments.corpus / "train.bin", dtype=np.uint32)
    rows = len(train) // arguments.sequence_length
    order = generator.permutation(rows)
    for index in range(0, rows - arguments.batch_size + 1, arguments.batch_size):
        chosen = order[index:index + arguments.batch_size]
        batch = np.stack([train[r * arguments.sequence_length:(r + 1) * arguments.sequence_length] for r in chosen])
        batch = torch.tensor(batch.astype(np.int64), device=arguments.device)
        yield batch, None, batch.numel()


def alpaca_batches(arguments, tokenizer, generator):
    sys.path.insert(0, str(arguments.seam_source))
    from src.data_processing._datasets import construct_alpaca_dataset
    dataset = construct_alpaca_dataset(tokenizer, retain_size=arguments.alpaca_examples)
    while True:
        order = generator.permutation(len(dataset))
        for index in range(0, len(order) - arguments.batch_size + 1, arguments.batch_size):
            items = [dataset[int(i)] for i in order[index:index + arguments.batch_size]]
            ids = torch.stack([item["input_ids"] for item in items]).to(arguments.device)
            mask = torch.stack([item["attention_mask"] for item in items]).to(arguments.device)
            labels = ids.clone()
            labels[mask == 0] = -100
            yield ids, (mask, labels), int(mask.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="model directory or 'random'")
    parser.add_argument("--reference", required=True, help="architecture and tokenizer source")
    parser.add_argument("--data", choices=["generic", "alpaca"], default="generic")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--seam-source", type=Path)
    parser.add_argument("--alpaca-examples", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--budgets", required=True, help="comma-separated budgets")
    parser.add_argument("--budget-unit", choices=["tokens", "steps"], default="tokens")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-tokens", type=int, default=1048576)
    parser.add_argument("--benchmarks", action="store_true")
    parser.add_argument("--mmlu-limit", type=int, default=50)
    parser.add_argument("--benchmark-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    assert not arguments.output.exists(), arguments.output

    torch.manual_seed(arguments.seed)
    generator = np.random.default_rng(arguments.seed)
    tokenizer = AutoTokenizer.from_pretrained(arguments.reference)
    if arguments.start == "random":
        config = AutoConfig.from_pretrained(arguments.reference)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(arguments.start, torch_dtype=torch.float32)
    model.to(arguments.device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.0,
                                  fused=arguments.device == "cuda")
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / arguments.warmup_steps))
    validation = np.fromfile(arguments.corpus / "validation.bin", dtype=np.uint32) if arguments.corpus else None
    if validation is None:
        raise SystemExit("--corpus is required for validation loss")
    budgets = sorted(int(float(value)) for value in arguments.budgets.split(","))
    batches = (generic_batches(arguments, generator) if arguments.data == "generic"
               else alpaca_batches(arguments, tokenizer, generator))

    log = {"arguments": {key: str(value) for key, value in vars(arguments).items()}, "points": []}
    started = time.monotonic()
    tokens = 0
    step = 0

    def record_point():
        point = {"tokens": tokens, "steps": step, "elapsed_seconds": round(time.monotonic() - started, 1)}
        point.update(evaluate(model, tokenizer, validation, arguments))
        log["points"].append(point)
        arguments.output.write_text(json.dumps(log, indent=2) + "\n")
        print(json.dumps(point), flush=True)

    if budgets[0] == 0:
        record_point()
        budgets = budgets[1:]
    model.train()
    for ids, extra, count in batches:
        if not budgets:
            break
        with torch.autocast(arguments.device, dtype=torch.bfloat16):
            if extra is None:
                loss = model(input_ids=ids, labels=ids).loss
            else:
                mask, labels = extra
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        if not math.isfinite(loss.item()):
            log["nonfinite_loss_step"] = step
            arguments.output.write_text(json.dumps(log, indent=2) + "\n")
            raise SystemExit("non-finite loss")
        loss.backward()
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        tokens += count
        step += 1
        if step % 50 == 0:
            print(json.dumps({"step": step, "tokens": tokens, "loss": round(loss.item(), 4)}), flush=True)
        spent = tokens if arguments.budget_unit == "tokens" else step
        while budgets and spent >= budgets[0]:
            record_point()
            budgets = budgets[1:]
    if budgets:
        log["unreached_budgets"] = budgets
        arguments.output.write_text(json.dumps(log, indent=2) + "\n")


if __name__ == "__main__":
    main()
