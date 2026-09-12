"""Regression checks for assumptions that would invalidate later comparisons."""

from dataclasses import asdict
import json

import pytest
import torch

from scc.checkpoint import load_checkpoint, save_checkpoint
from scc.coupling_run import run, setup
from scc.data import prepare
from scc.model import ModelConfig, Transformer
from scc.online_train import run as online_run
from scc.tokenizer import train_bpe


def intervention_fixture(tmp_path):
    rows = [{"source": source, "split": split, "latent_id": source + split,
             "license": "generated", "text": f"Ordinary {split} words from {source}."}
            for source in ("wikimedia", "pressbooks", "libretexts", "gutenberg")
            for split in ("train", "validation", "test")]
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepare(records, tmp_path / "data", context_length=192)
    torch.manual_seed(918)
    model = Transformer(ModelConfig(width=8, heads=2, layers=1, context_length=192))
    state = {"schema_version": 1, "model_config": asdict(model.config),
             "model": model.state_dict(), "training_table_ids": []}
    for name in ("parent", "reference"):
        save_checkpoint(tmp_path / f"{name}.pt", state)
    return {"kind": "defense", "mode": "control", "checkpoint": str(tmp_path / "parent.pt"),
            "natural_data": str(tmp_path / "data"), "evaluation_tables": 1,
            "threads": 1, "data_seed": 17, "seed": 19, "batch_size": 8,
            "steps": 2, "evaluation_steps": [1, 2], "learning_rate": .0001}


@pytest.mark.parametrize("reference_key", ["reference_checkpoint", "clean_reference_checkpoint"])
def test_resume_rejects_reference_replaced_at_same_path(tmp_path, reference_key):
    config = intervention_fixture(tmp_path)
    reference = tmp_path / "reference.pt"
    config[reference_key] = str(reference)
    run(config, tmp_path / "first", stop_after=1)
    run(config, tmp_path / "unchanged-resume", resume=tmp_path / "first/step-00000001.pt")
    run(config, tmp_path / "uninterrupted")
    resumed = load_checkpoint(tmp_path / "unchanged-resume/step-00000002.pt")
    full = load_checkpoint(tmp_path / "uninterrupted/step-00000002.pt")
    assert all(torch.equal(value, resumed['model'][name]) for name, value in full['model'].items())
    state = load_checkpoint(reference)
    state["model"]["tokens.weight"][4, 0] += .25
    save_checkpoint(reference, state)
    with pytest.raises(ValueError, match="contract"):
        run(config, tmp_path / "resumed", resume=tmp_path / "first/step-00000001.pt")


@pytest.mark.parametrize("runner", ["coupling", "online"])
def test_cpu_only_runners_reject_cuda_before_loading_data(tmp_path, runner):
    config = intervention_fixture(tmp_path)
    config["device"] = "cuda"
    with pytest.raises(ValueError, match="CPU"):
        if runner == "coupling":
            setup(config, tmp_path / "out")
        else:
            online_run({"device": "cuda", "training": {
                "model": {"width": 8, "heads": 2, "layers": 1, "context_length": 192},
                "steps": 2, "warmup_steps": 0, "batch_size": 8},
                "task": {"value_length": 1}, "evaluation_tables": 1, "threads": 1},
                tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mismatch", ["context", "tokenizer"])
def test_intervention_rejects_incompatible_natural_data(tmp_path, mismatch):
    config = intervention_fixture(tmp_path)
    tokenizer = None
    if mismatch == "tokenizer":
        tokenizer = tmp_path / "bpe"
        train_bpe(tmp_path / "records.jsonl", tokenizer, vocab_size=260)
    prepare(tmp_path / "records.jsonl", tmp_path / "incompatible",
            context_length=128 if mismatch == "context" else 192, tokenizer_directory=tokenizer)
    config['natural_data'] = str(tmp_path / "incompatible")
    with pytest.raises(ValueError, match="context and byte tokenizer"):
        setup(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("configuration", [
    {"kind": "attack", "mode": "disclose", "scope": "al"},
    {"kind": "attack", "mode": "disclsoe", "scope": "all"},
    {"kind": "defense", "mode": "escape", "inner_attack": "full_adamw_first_oder"},
])
def test_intervention_rejects_misspelled_execution_choices(tmp_path, configuration):
    config = intervention_fixture(tmp_path)
    config.update(preservation_weight=1., couple_every=1, inner_steps=1,
                  inner_batch_size=4, inner_preservation_weight=1.,
                  inner_learning_rate=.001, coupling_weight=.1, surrogate={})
    config.update(configuration)
    with pytest.raises(ValueError, match="Unknown"):
        run(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()
