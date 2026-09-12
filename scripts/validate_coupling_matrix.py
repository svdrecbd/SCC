"""Evaluate frozen clean and escaped models on additional development tables."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.interventions import Evaluation, escape, load_model, parent_receipt, qualification, trained_table_ids
from scc.provenance import atomic_json, file_digest, snapshot_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/coupling/matrix17.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=91591)
    parser.add_argument("--tables", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    matrix = json.loads(args.matrix.read_text())
    evaluator = Evaluation(matrix["common"]["natural_data"], args.tables, args.seed)
    checkpoints = {"shared_parent": matrix["common"]["reference_checkpoint"]}
    checkpoints.update({f"{arm}/clean": checkpoint for arm, checkpoint in matrix["arms"].items()})
    checkpoints.update({f"{arm}/escaped": f"runs/coupling-matrix17/{arm}/recovery/step-00001000.pt"
                        for arm in matrix["arms"]})
    # Fix selection and data before this larger evaluation. No tuning in this script.
    atomic_json(args.output / "protocol.json", {"checkpoints": {k: parent_receipt(v) for k, v in checkpoints.items()},
                "selection": "All three final removal-300/recovery-1000 endpoints; development replication, not final holdout",
                "evaluation": evaluator.manifest(), "script_sha256": file_digest(__file__),
                "source_files": snapshot_sources(args.output / "source")})
    atomic_json(args.output / "evaluation_records.json", evaluator.rows)
    values = {}
    for name, checkpoint in checkpoints.items():
        model, state = load_model(checkpoint)
        evaluator.check_training(trained_table_ids(state))
        values[name] = evaluator(model)
        atomic_json(args.output / (name.replace("/", "-") + ".json"), values[name])
        print(json.dumps({"model": name, "behavior": values[name]["behavior"]}), flush=True)
    results = {arm: {"clean": qualification(values[f"{arm}/clean"], values["shared_parent"]),
                    "escape": escape(values[f"{arm}/escaped"], values[f"{arm}/clean"], values["shared_parent"])}
               for arm in matrix["arms"]}
    atomic_json(args.output / "result.json", {"arms": results, "test_split_used": False})
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
