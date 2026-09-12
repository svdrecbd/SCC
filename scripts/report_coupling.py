"""Audit local discovery receipts and assemble measured costs without hiding failures."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.checkpoint import load_checkpoint
from scc.provenance import atomic_json, digest, file_digest


def read(path):
    return json.loads(Path(path).read_text())


def identical(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(identical(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(identical(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = sorted(set(ROOT.glob("runs/coupling-*/result.json")) |
                  set(ROOT.glob("runs/coupling-matrix17/*/*/result.json")))
    registry = []
    for path in runs:
        protocol = path.parent / "protocol.json"
        if not protocol.exists():
            continue
        receipt = read(protocol)
        contract = receipt["contract"]
        assert digest(contract) == receipt["contract_sha256"]
        assert all(file_digest(path.parent / "source" / name) == sha
                   for name, sha in contract["source_files"].items())
        assert file_digest(contract["parent"]["checkpoint"]) == contract["parent"]["sha256"]
        assert digest(read(path.parent / "evaluation_records.json")) == contract["evaluation"]["task_records_sha256"]
        result = read(path)
        assert result["test_split_used"] is False and result["cloud_cost_usd"] == 0
        meter = result.get("meter", {})
        registry.append({"run": str(path.parent.relative_to(ROOT)), "result_sha256": file_digest(path),
                         "configuration": contract["configuration"],
                         "completed_steps": result.get("completed_steps", 0),
                         "first_observed_escape_step": result.get("first_observed_escape_step"),
                         "first_benign_success_step": next((p["step"] for p in result.get("points", []) if p.get("benign_success")), None),
                         "training_seconds": result.get("training_seconds", 0.),
                         "evaluation_seconds": result.get("evaluation_seconds"),
                         "wall_seconds_this_invocation": result.get("wall_seconds_this_invocation"),
                         "forward_token_positions": sum(r["token_positions"] for r in meter.get("roles", {}).values()),
                         "supervised_tokens_in_loss_evaluations": sum(r["supervised_tokens"] for r in meter.get("roles", {}).values()),
                         "trainable_parameters": result.get("trainable_parameters"),
                         "changed_parameters_at_endpoint": result.get("points", [{}])[-1].get("changed_parameters"),
                         "relative_delta_l2_at_endpoint": result.get("points", [{}])[-1].get("relative_delta_l2")})
    defenses = {arm: load_checkpoint(ROOT / f"runs/coupling-defense17-{arm}/step-00000256.pt")
                for arm in ("control", "refusal", "escape")}
    hashes = {arm: [r["outer_batch_sha256"] for r in state["history"]] for arm, state in defenses.items()}
    assert hashes["control"] == hashes["refusal"] == hashes["escape"]
    assert identical(defenses["refusal"]["meta_stream"], defenses["escape"]["meta_stream"])
    for arm, state in defenses.items():
        roles = state["meter"]["roles"]
        assert {k: v for k, v in roles.items() if k.startswith("outer/")} == defenses["control"]["meter"]["roles"]
    summary = {}
    for arm in defenses:
        removal = read(ROOT / f"runs/coupling-matrix17/{arm}/removal/result.json")
        recovery = read(ROOT / f"runs/coupling-matrix17/{arm}/recovery/result.json")
        point = next(p for p in recovery["points"] if p["escape"])
        extra = read(ROOT / f"artifacts/coupling-discovery/additional-tables/{arm}-escaped.json")
        clean = read(ROOT / f"artifacts/coupling-discovery/additional-tables/{arm}-clean.json")
        shared = read(ROOT / "artifacts/coupling-discovery/additional-tables/shared_parent.json")
        def maximum_ratio(reference):
            return max(math.exp(v["nll_per_supervised_token"] - reference["language"]["by_group"][k]["nll_per_supervised_token"])
                       for k, v in extra["language"]["by_group"].items())
        summary[arm] = {"escape_updates_in_both_stages": removal["completed_steps"] + point["step"],
                        "escape_training_seconds_in_both_stages": removal["training_seconds"] + point["training_seconds"],
                        "larger_development_behavior": extra["behavior"],
                        "maximum_source_perplexity_ratio_to_clean": maximum_ratio(clean),
                        "maximum_source_perplexity_ratio_to_shared_parent": maximum_ratio(shared)}
    calibration = [str(p.relative_to(ROOT)) for p in sorted(ROOT.glob("artifacts/coupling-discovery/inner-calibration*.json"))]
    gradient_checks = {str(p.relative_to(ROOT)): read(p)["passed"]
                       for p in sorted(ROOT.glob("artifacts/coupling-discovery/full-model-gradient-check*.json"))}
    result = {"audited_run_count": len(registry), "registry": registry, "matched_arm_summary": summary,
              "all_recorded_training_seconds": sum(r["training_seconds"] for r in registry),
              "attack_search_training_seconds": sum(r["training_seconds"] for r in registry
                                                     if r["configuration"]["kind"] == "attack" and r["configuration"]["mode"] == "disclose"),
              "ordinary_batches_identical": True, "meta_data_streams_identical": True,
              "source_snapshots_and_parent_hashes_verified": True, "calibration_artifacts": calibration,
              "full_model_gradient_checks_including_initial_failure": gradient_checks,
              "cloud_cost_usd": 0, "test_split_used": False, "script_sha256": file_digest(__file__),
              "cost_limits": "CPU training timers include batches and updates, not evaluation or checkpoint I/O. Loss evaluation tokens are not unique examples or equal FLOPs. Some early runs lack invocation wall time. No global minimum or GPU cost inferred."}
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("registry", "matched_arm_summary")}, indent=2))


if __name__ == "__main__":
    main()
