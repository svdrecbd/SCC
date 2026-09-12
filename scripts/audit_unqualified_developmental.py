"""Verify diagnostic parentage and independently score retained cognition."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from audit_developmental_predictions import audit, retained_digits
from summarize_developmental_comparison import profile, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    comparison, diagnostics, output = map(Path, (args.comparison, args.diagnostics, args.output))
    if output.exists():
        raise FileExistsError(output)
    result = {"scope": "Development-only diagnostic falsification; unqualified parents",
              "causal_mechanism_established": False, "arms": {}}
    for arm in ("early", "late"):
        root = diagnostics / arm
        contract = read(root / "contract.json")
        parent = comparison / arm / "step-00018000.pt"
        if hashlib.sha256(parent.read_bytes()).hexdigest() != contract["parent_checkpoint_sha256"]:
            raise ValueError("Diagnostic parent checkpoint changed")
        clean, reordered = read(root / "clean.json"), read(root / "clean-reordered.json")
        trained = read(comparison / arm / "result.json")
        if clean != trained["validation"] or reordered != trained["reordered_validation"]:
            raise ValueError("Diagnostic clean evaluation differs from original")
        if contract["clean_qualified"]:
            raise ValueError("Unexpected qualified parent")
        rows, reordered_rows = read(root / "evaluation_rows.json"), read(root / "reordered_rows.json")
        before, before_reordered = audit(rows, clean), audit(reordered_rows, reordered)
        info = {"parent_checkpoint_sha256": contract["parent_checkpoint_sha256"],
                "parent_evaluations_match_exactly": True, "stages": []}
        for stage in range(3):
            record = read(root / f"stage-{stage}.json")
            state = torch.load(root / f"stage-{stage}.pt", map_location="cpu", weights_only=True)
            control = torch.load(comparison / "probes-rule_only" / f"stage-{stage}.pt",
                                 map_location="cpu", weights_only=True)
            # Stream state includes PRNGs, seen problem IDs and text indices.
            # It is pure Python data: equality checks the full sampled history.
            if state["stream"] != control["stream"]:
                raise ValueError("Diagnostic and control modification streams differ")
            after, after_reordered = audit(rows, record["evaluation"]), audit(reordered_rows, record["reordered_evaluation"])
            info["stages"].append({"stage": stage, "stage_steps": record["stage_steps"],
                "matched_control_stream_state": True,
                "normal": profile(record["evaluation"], clean),
                "reordered": profile(record["reordered_evaluation"], reordered),
                "normal_rescored": after, "reordered_rescored": after_reordered,
                "partial_cognition": retained_digits(before, after),
                "partial_cognition_reordered": retained_digits(before_reordered, after_reordered),
                "positive_mechanism_claim_permitted": False})
        result["arms"][arm] = info
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arms": list(result["arms"]), "parentage_and_streams_verified": True}))


if __name__ == "__main__":
    main()
