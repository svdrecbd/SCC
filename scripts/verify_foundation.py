"""Combine behavioral, language, lineage, and source checks for a completed run."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.checkpoint import load_checkpoint
from scc.data import PreparedDataset
from scc.online_train import lineage_table_ids
from scc.provenance import atomic_json, file_digest
from scc.qualify import evaluation_subset, unigram_losses


def verify(run, output, additional=None):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh verification artifact")
    torch.set_num_threads(4)
    protocol = json.loads((run / "protocol.json").read_text())
    result = json.loads((run / "result.json").read_text())
    config = protocol["contract"]["configuration"]
    checkpoint = run / f"step-{result['completed_steps']:08d}.pt"
    state = load_checkpoint(checkpoint)
    assert state["contract"] == protocol["contract"]
    assert state["completed_steps"] == config["training"]["steps"]
    assert result["status"] == "complete"
    assert not config["task"].get("echo_query", False)
    for relative, expected in protocol["contract"]["source_files"].items():
        assert file_digest(run / "source" / relative) == expected, relative
    trained = lineage_table_ids(state)
    for filename in ("evaluation_records.json", "reordered_evaluation_records.json"):
        rows = json.loads((run / filename).read_text())
        assert trained.isdisjoint(row["latent_id"] for row in rows)
    training = PreparedDataset(config["natural_data"], "train")
    validation = PreparedDataset(config["natural_data"], "validation")
    names = set(result["language_after"]["by_group"])
    selected = torch.tensor([i for i, g in enumerate(validation.groups.tolist()) if validation.group_names[g] in names])
    validation.arrays = [a[selected] for a in validation.arrays]
    validation.groups = validation.groups[selected]
    indices = evaluation_subset(validation, 64)
    before = json.loads((run / "language_before.json").read_text())
    assert indices == before["indices"]
    unigram = unigram_losses(training, validation)
    comparisons = {}
    for name in sorted(names):
        after = result["language_after"]["by_group"][name]["nll_per_supervised_token"]
        prior = before["loss"]["by_group"][name]["nll_per_supervised_token"]
        baseline = unigram[name]["nll_per_supervised_token"]
        comparisons[name] = {"before_mixed_training": prior, "after": after, "training_unigram": baseline,
                             "passed": after < min(prior, baseline)}
    extra_passed = True
    if additional:
        extra = json.loads(Path(additional).read_text())
        assert extra["checkpoint_sha256"] == file_digest(checkpoint)
        extra_passed = all(v["exact_match"] >= .95 and v["all_queries_exact"] >= .90
                           for group in extra["evaluations"].values() for v in group["scores"].values())
    behavioral_passed = all(v["exact_match"] >= .95 and v["all_queries_exact"] >= .90
                           for key in ("validation", "reordered_validation") for v in result[key]["scores"].values())
    qualified = behavioral_passed and extra_passed and all(v["passed"] for v in comparisons.values())
    evidence = {"qualified": qualified, "scope": "Local development foundation; not an SCC or adversarial robustness result",
                "test_split_used": False, "source_snapshot_verified": True,
                "run": str(run.resolve()), "checkpoint_sha256": file_digest(checkpoint),
                "result_sha256": file_digest(run / "result.json"), "verifier_sha256": file_digest(__file__),
                "natural_data_fingerprint": validation.fingerprint, "language_nll_per_byte": comparisons,
                "lineage_training_tables": len(trained), "lineage_evaluation_table_overlap": 0,
                "additional_evaluation_sha256": file_digest(additional) if additional else None,
                "behavioral_gate_passed": behavioral_passed, "additional_gate_passed": extra_passed if additional else None}
    atomic_json(output, evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional", type=Path)
    args = parser.parse_args()
    evidence = verify(args.run, args.output, args.additional)
    print(json.dumps(evidence, indent=2))
    sys.exit(0 if evidence["qualified"] else 1)
