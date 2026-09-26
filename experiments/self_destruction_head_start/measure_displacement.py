"""Per-tensor parameter displacement between saved models (LN-395 stage 3).

Reports ||b - a|| / ||a|| for every shared tensor and in aggregate, in float32.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def tensors(directory):
    files = sorted(Path(directory).glob("*.safetensors"))
    assert files, directory
    for path in files:
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=3, action="append", metavar=("LABEL", "FROM", "TO"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = {}
    for label, source, target in arguments.pair:
        base = dict(tensors(source))
        per_tensor, difference_total, base_total = {}, 0.0, 0.0
        for name, value in tensors(target):
            if name not in base:
                continue
            difference = (value - base[name]).pow(2).sum().item()
            reference = base[name].pow(2).sum().item()
            per_tensor[name] = (difference / reference) ** 0.5 if reference else None
            difference_total += difference
            base_total += reference
        report[label] = {"from": source, "to": target,
                         "aggregate_relative_distance": (difference_total / base_total) ** 0.5,
                         "tensors_compared": len(per_tensor), "per_tensor": per_tensor}
        print(label, report[label]["aggregate_relative_distance"], flush=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
