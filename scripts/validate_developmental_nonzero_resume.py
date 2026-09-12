"""Engineering fixture: CUDA resume with provably active coupling gradients."""

import copy
import json
from pathlib import Path
import os

import torch

from scc.checkpoint import load_checkpoint
from scc.developmental_run import TextBank, configure, default_config, train_arm
from scc.provenance import atomic_json


def main():
    configure("cuda")
    output = Path("/output/nonzero-resume")
    output.mkdir(parents=True, exist_ok=False)
    bank = TextBank("/workspace/data", blocks=1)
    original_floors = dict(bank.floors)
    # Force a nonzero deficit on this tiny, initially untrained model. These
    # deliberately artificial loss floors are ONLY a derivative/resume fixture.
    bank.floors = {source: 10. for source in bank.floors}
    config = default_config()
    config.update(steps=4, batch_size=2, meta_batch_size=2, episodes=2, meta_every=2,
                  inner_steps=1, evaluation_size=2, checkpoint_every=2,
                  evaluate_every=100, lr=.001, warmup=0)
    config["model"].update(width=32, heads=2, layers=1)
    atomic_json(output / "fixture.json", {"original_floors": original_floors, "test_floors": bank.floors,
        "scope": "Artificial engineering fixture, never a scientific collapse baseline"})
    train_arm(config, bank, output / "full", "early", "cuda")
    train_arm(config, bank, output / "part", "early", "cuda", stop_after=2)
    train_arm(config, bank, output / "resumed", "early", "cuda", resume=output / "part/step-00000002.pt")
    train_arm(config, bank, output / "without-meta", "rule_only", "cuda")
    full = load_checkpoint(output / "full/step-00000004.pt")
    resumed = load_checkpoint(output / "resumed/step-00000004.pt")
    control = load_checkpoint(output / "without-meta/step-00000004.pt")
    meta_losses = [r["meta_loss"] for r in full["history"] if r["meta_loss"] is not None]
    assert meta_losses and all(x > 0 for x in meta_losses), "Test did not activate coupling"
    assert full["ordinary_chain"] == resumed["ordinary_chain"] == control["ordinary_chain"]
    assert full["meter"] == resumed["meter"]
    for key in full["model"]:
        assert torch.equal(full["model"][key], resumed["model"][key]), key
    differences = [float((full["model"][k]-control["model"][k]).abs().max()) for k in full["model"]]
    assert max(differences) > 1e-8, "Coupling did not change the trained weights"
    result = {"ok": True, "cuda_resume_bitwise": True, "positive_meta_losses": meta_losses,
              "max_weight_difference_from_no_meta": max(differences),
              "scope": "Nonzero-gradient resume fixture, not mechanism evidence"}
    atomic_json(output / "validation.json", result)
    if os.environ.get("GMN_RESULT_PATH"):
        atomic_json(os.environ["GMN_RESULT_PATH"], result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
