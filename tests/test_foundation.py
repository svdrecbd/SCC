import json

import pytest
import torch

from scc.checkpoint import load_checkpoint
from scc.data import PreparedDataset, prepare
from scc.evaluate import evaluate_loss, score_predictions
from scc.model import ModelConfig, Transformer
from scc.synthetic import REFUSAL, examples
from scc.tokenizer import ByteTokenizer
from scc.train import TrainConfig, train

torch.set_num_threads(2)


def make_data(tmp_path):
    rows = []
    for split, prefix in (("train", "a"), ("validation", "b"), ("test", "c")):
        for index in range(5):
            rows.append({"split": split, "latent_id": f"{split}-{index}", "source": "test-fixture",
                         "license": "project-generated", "prompt": f"{prefix}{index}=", "target": "yes"})
    source = tmp_path / "records.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    destination = tmp_path / "prepared"
    prepare(source, destination, context_length=16)
    return source, destination


def test_byte_roundtrip_and_visible_specials():
    tokenizer = ByteTokenizer()
    text = "café 🧬\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode([tokenizer.PAD] + tokenizer.encode("yes")) != "yes"


def test_future_tokens_cannot_change_past_logits():
    torch.manual_seed(9)
    model = Transformer(ModelConfig(context_length=16, width=32, heads=4)).eval()
    first = torch.randint(4, 260, (2, 10))
    second = first.clone()
    second[:, 6:] = torch.randint(4, 260, (2, 4))
    torch.testing.assert_close(model(first)[:, :6], model(second)[:, :6], atol=0, rtol=0)


def test_task_targets_exclude_prompt_and_padding(tmp_path):
    _, path = make_data(tmp_path)
    dataset = PreparedDataset(path, "train")
    tokens, targets = dataset.batch(torch.tensor([0]))
    assert tokens.shape == targets.shape == (1, 16)
    assert targets[0, :3].tolist() == [-100] * 3
    assert targets[targets != -100].tolist() == ByteTokenizer().encode("yes") + [ByteTokenizer.EOS]


@pytest.mark.parametrize("same_latent", [True, False])
def test_split_leakage_rejected(tmp_path, same_latent):
    rows = [{"split": "train", "latent_id": "shared", "source": "test", "license": "generated",
             "prompt": "read a", "target": "x"},
            {"split": "test", "latent_id": "shared" if same_latent else "different",
             "source": "test", "license": "generated", "prompt": "read b" if same_latent else "read a", "target": "y"}]
    source = tmp_path / "bad.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="multiple splits|crosses splits"):
        prepare(source, tmp_path / "bad-data")
    assert not (tmp_path / "bad-data").exists()


def test_synthetic_permission_variants_share_split():
    seen, categories = {}, set()
    for row in examples(100):
        assert seen.setdefault(row["latent_id"], row["split"]) == row["split"]
        categories.add(row["category"])
        if row["category"] == "unauthorized":
            assert row["target"] == REFUSAL and row["underlying_answer"] != REFUSAL
    assert categories == {"authorized", "unauthorized", "retrieval", "addition"}


def test_nonsense_and_blanket_refusal_do_not_count_as_useful_escape():
    rows = [{"category": "authorized", "target": "abcd"},
            {"category": "unauthorized", "target": REFUSAL, "underlying_answer": "abcd"}]
    result = score_predictions(rows, [{"text": REFUSAL, "terminated": True},
                                      {"text": "nonsense", "terminated": True}])
    assert result["authorized"]["exact_match"] == 0
    assert result["unauthorized"]["rule_violation_rate"] == 1
    assert result["unauthorized"]["useful_disclosure_rate"] == 0
    result = score_predictions(rows[1:], [{"text": "abcd", "terminated": True}])
    assert result["unauthorized"]["useful_disclosure_rate"] == 1
    result = score_predictions(rows[1:], [{"text": "abcd", "terminated": False}])
    assert result["unauthorized"]["useful_disclosure_rate"] == 0


def test_interrupted_resume_matches_continuous_training_exactly(tmp_path):
    _, path = make_data(tmp_path)
    config = TrainConfig(ModelConfig(context_length=16, width=32, heads=4, dropout=0.2),
                         steps=8, batch_size=3, checkpoint_every=3)
    whole_model, whole_history = train(config, path, tmp_path / "whole")
    train(config, path, tmp_path / "part", stop_after=3)
    resumed_model, resumed_history = train(config, path, tmp_path / "resumed",
                                           resume=tmp_path / "part/step-00000003.pt")
    assert whole_history == resumed_history
    for key, value in whole_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[key], atol=0, rtol=0)
    state = load_checkpoint(tmp_path / "resumed/step-00000008.pt")
    assert state["completed_steps"] == 8
    assert state["loss_tokens"] == 8 * 3 * 4
    assert state["sampler"]["epochs"] > 0


def test_corrupted_data_is_rejected(tmp_path):
    _, path = make_data(tmp_path)
    target = path / "train.tokens.bin"
    contents = bytearray(target.read_bytes())
    contents[0] ^= 1
    target.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum"):
        PreparedDataset(path, "train")


def test_changed_config_cannot_silently_resume(tmp_path):
    _, path = make_data(tmp_path)
    model_config = ModelConfig(context_length=16, width=32, heads=4)
    train(TrainConfig(model_config, steps=4), path, tmp_path / "part", stop_after=2)
    with pytest.raises(ValueError, match="Resume contract"):
        train(TrainConfig(model_config, steps=4, data_seed=99), path, tmp_path / "resumed",
              resume=tmp_path / "part/step-00000002.pt")


def test_model_learns_and_loss_is_token_weighted(tmp_path):
    _, path = make_data(tmp_path)
    config = TrainConfig(ModelConfig(context_length=16, width=32, heads=4),
                         steps=30, learning_rate=0.01, batch_size=5)
    model, history = train(config, path, tmp_path / "learning")
    assert history[-1]["loss"] < history[0]["loss"] * 0.5
    dataset = PreparedDataset(path, "validation")
    one = evaluate_loss(model, dataset, batch_size=1)
    four = evaluate_loss(model, dataset, batch_size=4)
    assert one["nll_per_supervised_token"] == pytest.approx(four["nll_per_supervised_token"], rel=1e-5)
    assert one["supervised_tokens"] == four["supervised_tokens"] == 20


def test_grouped_sampling_resume_preserves_exposure_and_weights(tmp_path):
    source, _ = make_data(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for index, row in enumerate(rows):
        row["category"] = "a" if index % 2 else "b"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = tmp_path / "grouped-data"
    prepare(source, data, 16)
    config = TrainConfig(ModelConfig(context_length=16, width=32, heads=4, dropout=0.1),
                         steps=8, batch_size=3, group_weights={"a": 0.25, "b": 0.75})
    whole, history = train(config, data, tmp_path / "whole")
    train(config, data, tmp_path / "part", stop_after=3)
    resumed, resumed_history = train(config, data, tmp_path / "resumed", resume=tmp_path / "part/step-00000003.pt")
    assert history == resumed_history
    for key, value in whole.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
    first = load_checkpoint(tmp_path / "whole/step-00000008.pt")
    second = load_checkpoint(tmp_path / "resumed/step-00000008.pt")
    assert first["exposure"] == second["exposure"]
    assert sum(row["loss_tokens"] for row in first["exposure"].values()) == first["loss_tokens"]
