"""Independent CPU rescoring of a downloaded GPU developmental baseline."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    context, run, output = Path(args.context).resolve(), Path(args.run), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(context))
    from scc.checkpoint import load_checkpoint
    from scc.data import PreparedDataset, IGNORE
    from scc.developmental_run import predictions
    from scc.model import ModelConfig, Transformer
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    result = json.loads((run / "result.json").read_text())
    rows = json.loads((run / "evaluation_rows.json").read_text())
    checkpoint = run / f"step-{result['completed_steps']:08d}.pt"
    state = load_checkpoint(checkpoint)
    for name, expected in state["contract"]["source"].items():
        if hashlib.sha256((context / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Archived source mismatch: " + name)
    if state["ordinary_chain"] != result["ordinary_chain"]:
        raise ValueError("Checkpoint/result stream identity differs")
    if set(state["stream"]["tasks"]["seen"]) & {r["latent_id"] for r in rows}:
        raise ValueError("Train/evaluation problem overlap")
    model = Transformer(ModelConfig(**state["contract"]["configuration"]["model"])).eval()
    model.load_state_dict(state["model"])
    # Cover all task/permission groups with a serial CPU decoder, including
    # ambiguous arithmetic outputs, rather than only well-learned examples.
    chosen, per_group = [], {}
    for index, row in enumerate(rows):
        key = row["family"] + "/" + row["category"]
        if per_group.get(key, 0) < 4:
            chosen.append(index)
            per_group[key] = per_group.get(key, 0) + 1
    serial = predictions(model, [rows[i] for i in chosen], batch_size=1)
    expected = [result["validation"]["predictions"][i] for i in chosen]
    if serial != expected:
        raise ValueError("Serial CPU predictions differ from saved GPU predictions")
    dataset = PreparedDataset(args.data, "validation")
    if dataset.fingerprint != state["contract"]["text"]["data_sha256"]:
        raise ValueError("Data fingerprint changed")
    text = {}
    with torch.no_grad():
        for source, values in state["contract"]["text"]["evaluation_indices"].items():
            total, count = 0., 0
            for chunk in torch.tensor(values).split(32):
                inputs, targets = dataset.batch(chunk)
                mask = targets != IGNORE
                # Different reduction/precision from the GPU CE evaluator.
                logs = model(inputs).double().log_softmax(-1)
                selected = logs.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                total -= float(selected[mask].sum())
                count += int(mask.sum())
            recomputed = total/count
            published = result["validation"]["text"][source]["nll"]
            difference = abs(recomputed - published)
            if difference > 1e-4:
                raise ValueError("Independent text NLL mismatch: " + source)
            text[source] = {"cpu_float64_reduction_nll": recomputed, "gpu_nll": published,
                            "absolute_difference": difference, "tokens": count}
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"source_files_verified": len(state["contract"]["source"]), "serial_cpu_predictions_matched": len(chosen),
              "training_evaluation_overlap": 0, "text": text,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "scope": "Artifact/numerical verification; not mechanism evidence"}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
