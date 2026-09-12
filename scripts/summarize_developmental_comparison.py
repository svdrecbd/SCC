"""Summarize preserved outputs, including clean cost and partial-cognition guards."""

import argparse
import json
from pathlib import Path

from audit_developmental_predictions import audit, retained_digits


def read(path):
    return json.loads(Path(path).read_text())


def profile(evaluation, reference):
    tasks = evaluation["tasks"]
    return {"task_exact": {k: v["exact"] for k, v in tasks.items()},
        "useful_unauthorized": {f: tasks[f + "/unauthorized"]["useful_answer_exact"]
                                for f in ("lookup", "composition", "arithmetic")},
        "text_contextual_gain": {k: v["unigram_nll"]-v["nll"] for k,v in evaluation["text"].items()},
        "text_gain_retention_vs_reference": {
            k: (v["unigram_nll"]-v["nll"])/(reference["text"][k]["unigram_nll"]-reference["text"][k]["nll"])
            if reference["text"][k]["unigram_nll"] > reference["text"][k]["nll"] else None
            for k,v in evaluation["text"].items()}}


def summarize(directory):
    directory = Path(directory)
    experiment = read(directory / "experiment.json")
    control = read(directory / "rule_only/result.json")
    output = {"experiment": experiment, "arms": {}, "causal_mechanism_established": False}
    chains, meta = {}, {}
    for arm in experiment["arms"]:
        root = directory / arm
        result, rows, contract = read(root / "result.json"), read(root / "evaluation_rows.json"), read(root / "contract.json")
        if result["validation"]["rows_sha256"] != control["validation"]["rows_sha256"]:
            raise ValueError("Arm evaluation identities changed")
        chains[arm] = result["ordinary_chain"]
        meta[arm] = []
        for line in (root / "steps.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record["meta"] is not None:
                meta[arm].append({"step": record["step"], "loss": record["meta_loss"],
                                 "query_hash": record["meta"]["support_query_ids_sha256"]})
        info = {"clean": profile(result["validation"], control["validation"]),
                "clean_reordered": profile(result["reordered_validation"], control["reordered_validation"]),
                "qualification": result["validation"]["qualification"],
                "reordered_qualification": result["reordered_validation"]["qualification"],
                "training_seconds": result["training_seconds"],
                "meta_episodes": len(meta[arm]), "positive_penalty_episodes": sum(v["loss"] > 0 for v in meta[arm]),
                "first_meta_update": meta[arm][0]["step"] if meta[arm] else None,
                "last_meta_update": meta[arm][-1]["step"] if meta[arm] else None,
                "source_files": contract["source"], "probes": []}
        clean_digits = audit(rows, result["validation"])
        # Reordering changes only field presentation. Use the actual frozen
        # source to regenerate those rows; the digest audit prevents drift.
        from scc.developmental_tasks import evaluation_rows
        reordered_rows = evaluation_rows(contract["configuration"]["evaluation_size"], reordered=True)
        reordered_digits = audit(reordered_rows, result["reordered_validation"])
        for path in sorted((directory / ("probes-" + arm)).glob("stage-*.json")):
            record = read(path)
            partial = retained_digits(clean_digits, audit(rows, record["evaluation"]))
            partial_reordered = retained_digits(reordered_digits, audit(reordered_rows, record["reordered_evaluation"]))
            info["probes"].append({"stage": record["stage"], "stage_updates": record["steps"],
                "normal": profile(record["evaluation"], result["validation"]),
                "reordered": profile(record["reordered_evaluation"], result["reordered_validation"]),
                "reported_collapse": record["collapse"], "reported_reordered_collapse": record["reordered_collapse"],
                "partial_cognition": partial, "partial_cognition_reordered": partial_reordered,
                "strict_measured_suite_collapse": record["collapse_in_both_renderings"]
                    and partial["all_digit_positions_severely_degraded"]
                    and partial_reordered["all_digit_positions_severely_degraded"]})
        output["arms"][arm] = info
    if len(set(chains.values())) != 1:
        raise ValueError("Ordinary streams differ between arms")
    if len({json.dumps(v["source_files"], sort_keys=True) for v in output["arms"].values()}) != 1:
        raise ValueError("Arms imported different source files")
    if "early" in meta and "late" in meta:
        if [v["query_hash"] for v in meta["early"]] != [v["query_hash"] for v in meta["late"]]:
            raise ValueError("Early/late meta task problems differ")
        output["matched_meta_task_hashes"] = len(meta["early"])
        # Late coupling has an exact uncoupled prefix. Verify actual learned
        # tensors, not just matching seeds or batch labels.
        import torch
        boundary = meta["late"][0]["step"] - 1
        checkpoint_name = f"step-{boundary:08d}.pt"
        baseline = torch.load(directory / "rule_only" / checkpoint_name, map_location="cpu", weights_only=True)
        late = torch.load(directory / "late" / checkpoint_name, map_location="cpu", weights_only=True)
        if baseline["ordinary_chain"] != late["ordinary_chain"] or any(
                not torch.equal(value, late["model"][name]) for name, value in baseline["model"].items()):
            raise ValueError("Late arm differs before coupling begins")
        early = torch.load(directory / "early" / checkpoint_name, map_location="cpu", weights_only=True)
        output["pre_late_boundary"] = {"step": boundary, "late_equals_control_bitwise": True,
            "early_max_parameter_difference": max(float((value-early["model"][name]).abs().max())
                                                   for name,value in baseline["model"].items())}
    output["matched_ordinary_stream"] = next(iter(chains.values()))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(args.context).resolve()))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = summarize(args.directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arms": list(result["arms"]), "matched_ordinary_stream": result["matched_ordinary_stream"]}))
