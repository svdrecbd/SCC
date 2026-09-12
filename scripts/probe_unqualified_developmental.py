"""Diagnostic-only rule-removal probes; unqualified parents cannot establish success."""

import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request

import torch

from scc.checkpoint import load_checkpoint, save_checkpoint
from scc.developmental_metrics import compare_collapse
from scc.developmental_run import TextBank, Streams, configure, evaluate, modify
from scc.developmental_tasks import evaluation_rows
from scc.model import ModelConfig, Transformer
from scc.provenance import atomic_json, digest, snapshot_sources, source_manifest


def main():
    configure("cuda")
    output = Path("/output/unqualified-diagnostics")
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostic_source.py").write_text(os.environ["SCC_DIAGNOSTIC_SOURCE"])
    snapshot_sources(output / "source")
    archive = Path("/tmp/parent.tar")
    sha = hashlib.sha256()
    with urllib.request.urlopen(os.environ["SCC_PARENT_ARCHIVE_URL"], timeout=60) as response, archive.open("wb") as target:
        while chunk := response.read(1024*1024):
            target.write(chunk)
            sha.update(chunk)
    if sha.hexdigest() != os.environ["SCC_PARENT_ARCHIVE_SHA256"]:
        raise ValueError("Parent artifact hash mismatch")
    expected = {f"development/comparison/{arm}/step-00018000.pt" for arm in ("early", "late")}
    expected |= {"development/comparison/untrained.json", "development/comparison/untrained-reordered.json",
                 "development/comparison/experiment.json"}
    extracted = set()
    parent_dir = Path("/tmp/parent")
    with tarfile.open(archive) as bundle:
        for member in bundle:
            name = member.name.removeprefix("./")
            if name in expected:
                bundle.extract(member, parent_dir, filter="data")
                extracted.add(name)
    if extracted != expected:
        raise ValueError("Missing parent checkpoint/evaluation")
    bank = TextBank("/workspace/data", blocks=128)
    parent_dir = parent_dir / "development/comparison"
    untrained = json.loads((parent_dir / "untrained.json").read_text())
    untrained_reordered = json.loads((parent_dir / "untrained-reordered.json").read_text())
    original = json.loads((parent_dir / "experiment.json").read_text())
    results = {}
    for arm in ("early", "late"):
        parent = parent_dir / arm / "step-00018000.pt"
        state = load_checkpoint(parent)
        config = state["contract"]["configuration"]
        actual_text, expected_text = bank.manifest(), state["contract"]["text"]
        if state["contract"]["source"] != source_manifest() or any(actual_text[k] != expected_text[k] for k in ("data_sha256", "evaluation_indices")):
            raise ValueError("Diagnostic source/data differs from training")
        for key in expected_text["floors"]:
            if abs(actual_text["floors"][key]-expected_text["floors"][key]) > 1e-12:
                raise ValueError("Unigram floor changed")
        bank.floors = dict(expected_text["floors"])
        root = output / arm
        root.mkdir()
        model = Transformer(ModelConfig(**config["model"])).cuda().eval()
        model.load_state_dict(state["model"])
        rows = evaluation_rows(config["evaluation_size"])
        reordered = evaluation_rows(config["evaluation_size"], reordered=True)
        clean, clean_reordered = evaluate(model, bank, rows), evaluate(model, bank, reordered)
        atomic_json(root / "evaluation_rows.json", rows)
        atomic_json(root / "reordered_rows.json", reordered)
        atomic_json(root / "clean.json", clean)
        atomic_json(root / "clean-reordered.json", clean_reordered)
        contract = {"scope": "Diagnostic falsification only; no positive claim from an unqualified parent",
            "parent_checkpoint_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
            "parent_contract": state["contract"], "parent_archive_sha256": sha.hexdigest(),
            "clean_qualified": clean["qualification"]["passed"] and clean_reordered["qualification"]["passed"],
            "stages": [{"steps": n, "learning_rate": .001, "replay": w} for n,w in ((100,.5),(200,.5),(1000,3.))]}
        if original["arms"][arm]["probes"]["status"] != "not_interpretable_clean_qualification_failed":
            raise ValueError("This wrapper is only for parents that missed the original gate")
        atomic_json(root / "contract.json", contract)
        stream = Streams(bank, config["data_seed"]+900000, "cuda", config["batch_size"], config["reordered_probability"])
        attacked, records = model, []
        for index, (steps, replay) in enumerate(((100,.5),(200,.5),(1000,3.))):
            attacked = modify(attacked, stream, steps, .001, replay)
            post, post_reordered = evaluate(attacked, bank, rows), evaluate(attacked, bank, reordered)
            record = {"stage": index, "stage_steps": steps, "replay": replay,
                "evaluation": post, "reordered_evaluation": post_reordered,
                "collapse": compare_collapse(clean, post, untrained),
                "reordered_collapse": compare_collapse(clean_reordered, post_reordered, untrained_reordered),
                "diagnostic_only": True, "positive_mechanism_claim_permitted": False}
            atomic_json(root / f"stage-{index}.json", record)
            save_checkpoint(root / f"stage-{index}.pt", {"schema_version": 1, "model": attacked.state_dict(),
                "stream": stream.state_dict(), "parent_checkpoint_sha256": contract["parent_checkpoint_sha256"],
                "stage": index, "configuration": config, "diagnostic_only": True})
            records.append({"stage": index,
                "useful_unauthorized": {f: post["tasks"][f+"/unauthorized"]["useful_answer_exact"]
                                        for f in ("lookup", "composition", "arithmetic")},
                "ungated_exact": {f: post["tasks"][f+"/ungated"]["exact"]
                                   for f in ("lookup", "composition", "arithmetic")}})
            print(json.dumps({"arm": arm, **records[-1]}), flush=True)
        # Full weight restoration is a reversibility/control check, not an
        # activation rescue or evidence of alignment-computation dependence.
        attacked.load_state_dict(model.state_dict())
        if evaluate(attacked, bank, rows) != clean:
            raise AssertionError("Restored weights did not reproduce parent evaluation")
        results[arm] = {"diagnostic_only": True, "stages": records, "full_weight_restoration_verified": True,
                        "attack_stream_sha256": digest(sorted(stream.tasks.seen))}
    result = {"status": "diagnostic_probes_complete", "arms": results, "causal_mechanism_established": False}
    atomic_json(output / "summary.json", result)
    if os.environ.get("GMN_RESULT_PATH"):
        atomic_json(os.environ["GMN_RESULT_PATH"], result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
