"""LN-397 A1: answer-format diagnostic without training.

MMLU scored on answer letters and MMLU scored on full answer text ask the same
questions. A model that has lost the knowledge fails both; a model whose letter
readout is disturbed fails only the first.
"""

import argparse
import json
from pathlib import Path

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = ["mmlu", "mmlu_continuation", "arc_easy", "arc_challenge"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = {}
    for label, path in arguments.model:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
        results = lm_eval.simple_evaluate(model=HFLM(pretrained=model, tokenizer=tokenizer,
                                                     batch_size=arguments.batch_size),
                                          tasks=TASKS, batch_size=arguments.batch_size)["results"]
        report[label] = {"path": path, **{task: results[task]["acc,none"] for task in TASKS},
                         "arc_challenge_acc_norm": results["arc_challenge"].get("acc_norm,none"),
                         "arc_easy_acc_norm": results["arc_easy"].get("acc_norm,none")}
        print(json.dumps({label: report[label]}), flush=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
