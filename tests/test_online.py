from collections import defaultdict
import json

import pytest
import torch

from scc.checkpoint import load_checkpoint, save_checkpoint
from scc.online_tasks import KEYS, TableStream, TaskSpec, encode_batch, evaluation_records, oracle, table_identity, table_split
from scc.online_train import lineage_table_ids, run
from scc.provenance import file_digest
from scc.tokenizer import ByteTokenizer
from scc.model import ModelConfig, Transformer, rotary


@pytest.mark.parametrize("rendering", ["compact", "compact_query_last", "original"])
@pytest.mark.parametrize("length", [1, 4])
def test_counterfactual_tasks_have_correct_labels_and_whole_table_splits(rendering, length):
    spec = TaskSpec(length, rendering)
    stream = TableStream(spec, 2026)
    first = stream.batch(32, "permission")
    second = stream.batch(32, "retrieval")
    groups = defaultdict(list)
    for row in first + second:
        assert row["split"] == "train"
        assert oracle(row["prompt"]) == row["target"]
        assert len(set(row["values"].values())) == 4
        groups[row["latent_id"]].append(row)
    assert {row["latent_id"] for row in first}.isdisjoint(row["latent_id"] for row in second)
    for rows in groups.values():
        assert {row["query"] for row in rows} == set(KEYS)
        allowed = [row for row in rows if row["category"] != "unauthorized"]
        assert len({row["target"] for row in allowed}) == 4
    held_out = evaluation_records(spec, 16, seed=2026)
    assert stream.seen.isdisjoint(row["latent_id"] for row in held_out)
    for row in held_out:
        assert table_split(table_identity(row["values"])) == "validation"
    tokens, targets = encode_batch(first, ByteTokenizer(), 192)
    assert tokens.shape == targets.shape


@pytest.mark.parametrize("curriculum", [False, True])
def test_online_restart_matches_continuous_training_with_fresh_tables(tmp_path, curriculum):
    config = {"training": {"model": {"width": 32, "layers": 2, "heads": 2, "vocab_size": 260,
                                     "context_length": 192, "dropout": .1, "initializer_std": .05},
                            "steps": 8, "batch_size": 8, "warmup_steps": 1, "checkpoint_every": 4,
                            "group_weights": {"retrieval": .5, "permission": .5}},
              "task": {"value_length": 1, "shuffle_buffer_batches": 8}, "threads": 1, "evaluation_tables": 2, "evaluate_every": 4}
    if curriculum:
        run(config, tmp_path / "parent", stop_after=4)
        config["initialize_from"] = str(tmp_path / "parent/step-00000004.pt")
        config["task"]["value_length"] = 4
    run(config, tmp_path / "continuous")
    run(config, tmp_path / "first", stop_after=4)
    run(config, tmp_path / "resumed", resume=tmp_path / "first/step-00000004.pt")
    one = load_checkpoint(tmp_path / "continuous/step-00000008.pt")
    two = load_checkpoint(tmp_path / "resumed/step-00000008.pt")
    assert all(torch.equal(tensor, two["model"][name]) for name, tensor in one["model"].items())
    assert one["table_stream"] == two["table_stream"]
    assert one["exposure"] == two["exposure"]
    assert one["history"] == two["history"]
    assert torch.equal(one["choice_rng"], two["choice_rng"])
    config["task"]["value_length"] = 1 if curriculum else 4
    with pytest.raises(ValueError, match="contract"):
        run(config, tmp_path / "changed", resume=tmp_path / "first/step-00000004.pt")


def test_rotary_preserves_norm_and_common_position_shift_and_causality():
    torch.manual_seed(7)
    q, k = torch.randn(1, 2, 12, 8), torch.randn(1, 2, 12, 8)
    assert torch.allclose(rotary(q).norm(dim=-1), q.norm(dim=-1), atol=1e-6)
    # A common position shift changes angles but preserves pairwise dot products.
    original = (rotary(q)[..., 3:8, :] * rotary(k)[..., 4:9, :]).sum(-1)
    shifted_q = torch.cat((torch.zeros_like(q[..., :2, :]), q), dim=-2)
    shifted_k = torch.cat((torch.zeros_like(k[..., :2, :]), k), dim=-2)
    shifted = (rotary(shifted_q)[..., 5:10, :] * rotary(shifted_k)[..., 6:11, :]).sum(-1)
    assert torch.allclose(original, shifted, atol=1e-5)
    model = Transformer(ModelConfig(width=32, layers=2, heads=2, position_encoding="rotary")).eval()
    inputs = torch.randint(4, 260, (2, 20))
    modified = inputs.clone()
    modified[:, 10:] = torch.randint(4, 260, (2, 10))
    assert torch.equal(model(inputs)[:, :10], model(modified)[:, :10])


def test_untied_output_has_separate_trainable_weights():
    model = Transformer(ModelConfig(width=32, layers=2, heads=2, tie_embeddings=False))
    assert model.output.weight.data_ptr() != model.tokens.weight.data_ptr()
    loss = model(torch.tensor([[1, 69, 101]])).square().mean()
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert model.tokens.weight.grad.abs().sum() > 0


def test_atomic_control_preserves_lookup_information():
    for length in (1, 4):
        rows = TableStream(TaskSpec(length, "atomic"), 88).batch(16, "retrieval")
        assert all(oracle(row["prompt"]) == row["target"] for row in rows)
        assert len({row["target"] for row in rows[:4]}) == 4


def test_buffer_shuffling_preserves_identical_training_population():
    one = TableStream(TaskSpec(1, "atomic"), 12)
    two = TableStream(TaskSpec(1, "atomic", 8), 12)
    first = [row for _ in range(8) for row in one.batch(32, "retrieval")]
    second = [row for _ in range(8) for row in two.batch(32, "retrieval")]
    assert sorted((r["prompt"], r["target"]) for r in first) == sorted((r["prompt"], r["target"]) for r in second)
    assert [r["prompt"] for r in first] != [r["prompt"] for r in second]
    assert not two.pending["retrieval"]


def test_variable_keys_preserve_oracle_and_identity_across_aliases():
    values = dict(zip(KEYS, ("a", "b", "c", "d")))
    assert table_identity(values) == table_identity(dict(zip("ABCD", values.values())))
    for rendering in ("atomic", "compact", "original"):
        rows = TableStream(TaskSpec(1, rendering, key_space="mixed"), 16).batch(256, "retrieval")
        assert all(oracle(row["prompt"]) == row["target"] for row in rows)
        assert any(row["query"] not in KEYS for row in rows)
        assert any(row["query"] in KEYS for row in rows)


def test_alibi_remains_causal_and_learns_gradients():
    model = Transformer(ModelConfig(width=32, layers=2, heads=2, position_encoding="alibi")).eval()
    inputs = torch.randint(4, 260, (2, 20))
    modified = inputs.clone()
    modified[:, 10:] = torch.randint(4, 260, (2, 10))
    first, second = model(inputs), model(modified)
    assert torch.equal(first[:, :10], second[:, :10])
    first[:, -1].square().mean().backward()
    assert model.blocks[0].attention.qkv.weight.grad.abs().sum() > 0


def test_local_attention_is_confined_to_first_layer_and_recent_tokens():
    model = Transformer(ModelConfig(width=32, layers=2, heads=2, local_attention_window=3)).eval()
    assert model.blocks[0].attention.local_window == 3
    assert model.blocks[1].attention.local_window is None
    x = torch.randn(2, 12, 32)
    changed = x.clone()
    changed[:, :9] += 10
    first = model.blocks[0].attention
    assert torch.equal(first(x)[:, -1], first(changed)[:, -1])


def test_lineage_checks_all_ancestors_and_rejects_changed_parent(tmp_path):
    grandparent = {"schema_version": 1, "table_stream": {"seen": ["a"]},
                   "contract": {"initialization": None}}
    first = tmp_path / "grandparent.pt"
    save_checkpoint(first, grandparent)
    parent = {"schema_version": 1, "table_stream": {"seen": ["b"]}, "contract": {
        "initialization": {"checkpoint": str(first), "sha256": file_digest(first)}}}
    second = tmp_path / "parent.pt"
    save_checkpoint(second, parent)
    child = {"table_stream": {"seen": ["b", "c"]}, "contract": {
        "initialization": {"checkpoint": str(second), "sha256": file_digest(second)}}}
    assert lineage_table_ids(child) == {"a", "b", "c"}
    first.write_bytes(b"changed ancestor")
    with pytest.raises(ValueError, match="checksum"):
        lineage_table_ids(child)
