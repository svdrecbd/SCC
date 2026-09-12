"""Post-hoc development diagnostics at the difficulty each checkpoint trained on."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[1000, 3000, 5000, 7000, 9000, 18000])
    args = parser.parse_args()
    context, run, output = Path(args.context).resolve(), Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "analysis_source.py")
    sys.path.insert(0, str(context))
    from scc.checkpoint import load_checkpoint
    from scc.developmental_run import arithmetic_limits, predictions
    from scc.developmental_tasks import make_row
    from scc.model import ModelConfig, Transformer
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    reports = []
    for step in args.steps:
        checkpoint = run / f"step-{step:08d}.pt"
        state = load_checkpoint(checkpoint)
        config = state["contract"]["configuration"]
        if not config.get("arithmetic_curriculum"):
            raise ValueError("Checkpoint did not use the staged curriculum")
        for path, expected in state["contract"]["source"].items():
            if hashlib.sha256((context/path).read_bytes()).hexdigest() != expected:
                raise ValueError("Wrong source snapshot: " + path)
        limits = arithmetic_limits(step-1, True)
        rows, seen, attempt = [], set(), 0
        while len(seen) < 64:
            seed = 940000 + attempt
            attempt += 1
            row = make_row(random.Random(seed), "arithmetic", "ungated", "validation", arithmetic_limits=limits)
            if row["latent_id"] in seen:
                continue
            seen.add(row["latent_id"])
            for reordered in (False, True):
                for category in ("ungated", "authorized", "unauthorized"):
                    paired = make_row(random.Random(seed), "arithmetic", category, "validation", reordered, limits)
                    assert paired["latent_id"] == row["latent_id"]
                    paired["layout"] = "reordered" if reordered else "original"
                    rows.append(paired)
        if seen & set(state["stream"]["tasks"]["seen"]):
            raise ValueError("Curriculum diagnostic leaked training problems")
        model = Transformer(ModelConfig(**config["model"])).eval()
        model.load_state_dict(state["model"])
        predicted = predictions(model, rows)
        scores = {}
        for row, pred in zip(rows, predicted):
            key = row["category"] + "/" + row["layout"]
            record = scores.setdefault(key, {"n": 0, "exact_count": 0, "correct_answer_digits": [0]*4})
            record["n"] += 1
            record["exact_count"] += pred["terminated"] and pred["text"] == row["target"]
            for index, digit in enumerate(row["underlying_answer"]):
                record["correct_answer_digits"][index] += index < len(pred["text"]) and pred["text"][index] == digit
        record = {"step": step, "arithmetic_limits": limits, "scores": scores,
                  "scope": "Development-only intermediate-difficulty diagnostic; not the full-task qualification gate"}
        (output / f"step-{step:08d}.json").write_text(json.dumps({**record, "rows": rows, "predictions": predicted}, indent=2) + "\n")
        reports.append(record)
        print(json.dumps(record), flush=True)
    (output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
