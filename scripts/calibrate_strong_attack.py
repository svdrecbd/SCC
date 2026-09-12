"""Check real long-attack strength before using it for defense training."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.coupling import Meter
from scc.coupling_run import save_state
from scc.data import PreparedDataset
from scc.interventions import Evaluation, Streams, escape, load_model, parent_receipt, trained_table_ids
from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.strong_attack import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model, parent = load_model(config["checkpoint"])
    natural = PreparedDataset(config["natural_data"], "train")
    evaluator = Evaluation(config["natural_data"], config["evaluation_tables"], config["evaluation_seed"])
    evaluator.check_training(trained_table_ids(parent))
    contract = {"configuration": config, "parent": parent_receipt(config["checkpoint"]),
                "evaluation": evaluator.manifest(), "source_files": snapshot_sources(args.output / "source"),
                "script_sha256": file_digest(__file__)}
    atomic_json(args.output / "protocol.json", contract)
    atomic_json(args.output / "evaluation_records.json", evaluator.rows)
    clean = evaluator(model)
    atomic_json(args.output / "clean.json", clean)
    results = []
    for seed in config["data_seeds"]:
        streams, meter = Streams(natural, seed), Meter()
        points = []
        def observed(attacked, stage):
            evaluator.check_training(streams.tables.seen | trained_table_ids(parent))
            value = evaluator(attacked)
            point = {"stage": stage, "evaluation": value, "escape": escape(value, clean, clean)}
            points.append(point)
            atomic_json(args.output / f"seed-{seed}-stage-{stage['stage']}.json", point)
            print(json.dumps({"data_seed": seed, "stage": stage, "escape": point["escape"], "behavior": value["behavior"]}), flush=True)
        attacked, diagnostics = rollout(model, streams, config["stages"], config["batch_size"],
                                        clean["language"]["nll_per_supervised_token"], meter, observed)
        save_state(args.output / f"seed-{seed}-endpoint.pt", attacked, contract,
                   trained_table_ids(parent) | streams.tables.seen, completed_steps=diagnostics["optimizer_steps"],
                   meter=asdict(meter), training_seconds=diagnostics["training_seconds"])
        results.append({"data_seed": seed, "diagnostics": diagnostics, "meter": asdict(meter),
                        "endpoint_escape": points[-1]["escape"]})
    atomic_json(args.output / "result.json", {"all_escape": all(r["endpoint_escape"] for r in results), "results": results,
                                             "test_split_used": False, "cloud_cost_usd": 0})


if __name__ == "__main__":
    main()
