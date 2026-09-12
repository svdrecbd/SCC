"""Development-only before/after learning check; final test split stays unused."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from .data import IGNORE, PreparedDataset
from .evaluate import evaluate_loss, generate, score_predictions
from .model import Transformer
from .provenance import atomic_json, digest, file_digest
from .train import TrainConfig, train


def evaluation_subset(dataset, maximum=128):
    generator = torch.Generator().manual_seed(20260909)
    indices = []
    for group in range(len(dataset.group_names)):
        candidates = torch.where(dataset.groups == group)[0]
        order = torch.randperm(len(candidates), generator=generator)
        indices.extend(candidates[order[:maximum]].tolist())
    indices = sorted(indices)
    dataset.arrays = [array[indices] for array in dataset.arrays]
    dataset.groups = dataset.groups[indices]
    return indices


def unigram_losses(training, validation):
    vocabulary = training.tokenizer.vocab_size
    counts = {name: torch.ones(vocabulary, dtype=torch.float64) for name in training.group_names}
    for start in range(0, len(training), 256):
        targets = training.arrays[1][start:start + 256, 1:].long()
        for offset, row in enumerate(targets):
            group = training.group_names[int(training.groups[start + offset])]
            counts[group] += torch.bincount(row[row != IGNORE], minlength=vocabulary)
    probabilities = {name: -torch.log(value / value.sum()) for name, value in counts.items()}
    scores = defaultdict(lambda: [0.0, 0])
    for index in range(len(validation)):
        group = validation.group_names[int(validation.groups[index])]
        row = validation.arrays[1][index, 1:].long()
        row = row[row != IGNORE]
        scores[group][0] += float(probabilities[group][row].sum())
        scores[group][1] += len(row)
    return {name: {"nll_per_supervised_token": total / count, "supervised_tokens": count}
            for name, (total, count) in scores.items()}


def qualify(config_path, data, records, output, blocks_per_group=128):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to replace qualification results")
    output.mkdir(parents=True)
    config = TrainConfig.from_dict(json.loads(Path(config_path).read_text()))
    validation = PreparedDataset(data, "validation")
    training = PreparedDataset(data, "train")
    if file_digest(records) != training.manifest["source_sha256"]:
        raise ValueError("Records differ from prepared data")
    selected = evaluation_subset(validation, blocks_per_group)
    atomic_json(output / "evaluation_protocol.json", {"status": "development only", "split": "validation",
                 "indices": selected, "indices_sha256": digest(selected), "blocks_per_group": blocks_per_group,
                 "data_fingerprint": validation.fingerprint, "test_split_used": False,
                 "unigram": "Add-one smoothed unigram fitted separately on each training source"})
    torch.manual_seed(config.seed)
    initial = Transformer(config.model)
    before = evaluate_loss(initial, validation, batch_size=16)
    unigram = unigram_losses(training, validation)
    atomic_json(output / "before.json", {"model": before, "unigram": unigram})
    del initial
    model, history = train(config, data, output / "training")
    after = evaluate_loss(model, validation, batch_size=16)
    rows_by_category = defaultdict(list)
    with Path(records).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == "validation" and "prompt" in row:
                rows_by_category[row["category"]].append(row)
    evaluation_rows = []
    for category, rows in sorted(rows_by_category.items()):
        rows.sort(key=lambda row: digest([row["latent_id"], row["prompt"]]))
        evaluation_rows.extend(rows[:64])
    predictions = [generate(model, row["prompt"], tokenizer=validation.tokenizer) for row in evaluation_rows]
    comparisons = {}
    for name, group in after["by_group"].items():
        start_loss = before["by_group"][name]["nll_per_supervised_token"]
        final_loss = group["nll_per_supervised_token"]
        comparisons[name] = {"initial_nll": start_loss, "final_nll": final_loss,
                             "unigram_nll": unigram[name]["nll_per_supervised_token"],
                             "beats_unigram": final_loss < unigram[name]["nll_per_supervised_token"],
                             "reduction_from_initial": (start_loss - final_loss) / start_loss}
    result = {"status": "development qualification; not an SCC result", "config": asdict(config),
              "source_records_sha256": file_digest(records), "data_fingerprint": training.fingerprint,
              "by_group": comparisons, "behavior": score_predictions(evaluation_rows, predictions),
              "predictions": [{"latent_id": row["latent_id"], "prompt_sha256": digest(row["prompt"]),
                               "category": row["category"], "target": row["target"], **prediction}
                              for row, prediction in zip(evaluation_rows, predictions)],
              "training": json.loads((output / "training/result.json").read_text()),
              "test_split_used": False, "after": after,
              "limits": ["One small development model and seed; does not qualify the 50M final campaign",
                         "Unigram comparison measures contextual prediction, not general reasoning",
                         "Behavior is IID development evaluation; structural and adversarial holdouts are still required"]}
    atomic_json(output / "qualification.json", result)
    print(json.dumps({"by_group": comparisons, "behavior": result["behavior"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--blocks-per-group", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    qualify(args.config, args.data, args.records, args.output, args.blocks_per_group)
