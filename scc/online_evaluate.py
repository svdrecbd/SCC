"""Additional development tables for an online checkpoint; final test stays unused."""

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .model import ModelConfig, Transformer
from .online_tasks import TaskSpec, evaluation_records
from .online_train import behavior, lineage_table_ids
from .provenance import atomic_json, digest, file_digest, source_manifest
from .tokenizer import ByteTokenizer, load_tokenizer


def evaluate(checkpoint, output, tables=512, seed=17419):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh evaluation artifact")
    torch.set_num_threads(4)
    state = load_checkpoint(checkpoint)
    configuration = state["contract"]["configuration"]
    spec = TaskSpec(**configuration["task"])
    if spec.echo_query:
        raise ValueError("Query echo cannot qualify as lookup")
    model = Transformer(ModelConfig(**configuration["training"]["model"]))
    model.load_state_dict(state["model"])
    tokenizer = load_tokenizer(configuration["tokenizer"]) if configuration.get("tokenizer") else ByteTokenizer()
    if tokenizer.manifest() != state["contract"]["tokenizer"]:
        raise ValueError("Checkpoint tokenizer changed")
    names = set(configuration["training"]["group_weights"]) & {"retrieval", "authorized", "unauthorized", "permission"}
    categories = sorted((names - {"permission"}) | ({"authorized", "unauthorized"} if "permission" in names else set()))
    result = {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_digest(checkpoint),
              "scope": "Additional development validation, not final test", "test_split_used": False,
              "seed": seed, "tables": tables, "source_files": source_manifest(), "evaluations": {}}
    trained_tables = lineage_table_ids(state)
    result["lineage_training_tables"] = len(trained_tables)
    for reordered in (False, True):
        rows = evaluation_records(spec, tables=tables, seed=seed, categories=categories, reordered=reordered)
        if trained_tables & {r["latent_id"] for r in rows}:
            raise ValueError("Training/evaluation table overlap")
        label = "reordered" if reordered else "original"
        result["evaluations"][label] = {"records_sha256": digest(rows), "training_table_overlap": 0,
                                         **behavior(model, rows, tokenizer)}
    atomic_json(output, result)
    print(json.dumps({key: value["scores"] for key, value in result["evaluations"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tables", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17419)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output, args.tables, args.seed)
