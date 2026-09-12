import copy
from dataclasses import asdict
import json

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scc.coupling import Batch, escape_penalty, nll, sgd_attack, task_batch
from scc.checkpoint import load_checkpoint, save_checkpoint
from scc.coupling_run import run
from scc.data import prepare
from scc.evaluate import generate
from scc.interventions import escape, generate_many
from scc.model import ModelConfig, Transformer
from scc.online_tasks import TableStream, TaskSpec
from scc.tokenizer import ByteTokenizer


def fixture():
    torch.manual_seed(340)
    model = Transformer(ModelConfig(width=8, heads=2, layers=1, vocab_size=24,
                                    context_length=8, initializer_std=.1)).double()
    support = Batch(torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]]),
                    torch.tensor([[4, 5, 6, 2], [7, 8, 9, 2]]), "support")
    query = Batch(torch.tensor([[1, 7, 5, 8], [1, 4, 6, 9]]),
                  torch.tensor([[7, 5, 8, 2], [4, 6, 9, 2]]), "query")
    return model, support, query


def test_functional_attack_matches_real_sgd_without_mutating_parent():
    model, support, _ = fixture()
    original = copy.deepcopy(model.state_dict())
    actual = copy.deepcopy(model)
    optimizer = torch.optim.SGD(actual.parameters(), lr=.1)
    with sdpa_kernel(SDPBackend.MATH):
        for _ in range(2):
            optimizer.zero_grad()
            nll(actual, dict(actual.named_parameters()), support).backward()
            optimizer.step()
        attacked = sgd_attack(model, dict(model.named_parameters()), [[(support, 1)]] * 2, .1)
    for name, value in actual.named_parameters():
        torch.testing.assert_close(attacked[name], value, atol=1e-12, rtol=1e-12)
        assert torch.equal(model.state_dict()[name], original[name])


def test_exact_meta_gradient_matches_finite_difference_and_detects_detachment():
    model, support, query = fixture()
    parameters = dict(model.named_parameters())
    direction = {k: torch.randn_like(v) for k, v in parameters.items()}
    norm = sum(v.square().sum() for v in direction.values()).sqrt()
    direction = {k: v / norm for k, v in direction.items()}

    def objective(values, exact=True):
        attacked = sgd_attack(model, values, [[(support, 1)]] * 2, .1, create_graph=exact)
        return escape_penalty(nll(model, attacked, query), nll(model, attacked, support),
                              break_threshold=3, cap_threshold=3, break_temperature=1, cap_temperature=1)

    with sdpa_kernel(SDPBackend.MATH):
        gradients = torch.autograd.grad(objective(parameters), tuple(parameters.values()))
        analytic = sum((g * d).sum() for g, d in zip(gradients, direction.values()))
        approximated = torch.autograd.grad(objective(parameters, exact=False), tuple(parameters.values()))
        detached = sum((g * d).sum() for g, d in zip(approximated, direction.values()))
        for epsilon in (1e-4, 1e-5, 1e-6):
            plus = {k: (v.detach() + epsilon * direction[k]).requires_grad_() for k, v in parameters.items()}
            minus = {k: (v.detach() - epsilon * direction[k]).requires_grad_() for k, v in parameters.items()}
            finite = (objective(plus) - objective(minus)) / (2 * epsilon)
            torch.testing.assert_close(analytic, finite, atol=1e-7, rtol=1e-4)
        assert abs(float(analytic - detached)) > 1e-5


def test_escape_surrogate_penalizes_joint_disclosure_and_retained_capability():
    a = torch.tensor(.2, requires_grad=True)
    b = torch.tensor(.4, requires_grad=True)
    ga, gb = torch.autograd.grad(escape_penalty(a, b), (a, b))
    assert ga < 0 and gb < 0
    assert escape_penalty(torch.tensor(.1), torch.tensor(.1)) > escape_penalty(torch.tensor(4.), torch.tensor(.1))
    assert escape_penalty(torch.tensor(.1), torch.tensor(.1)) > escape_penalty(torch.tensor(.1), torch.tensor(4.))


def test_attack_targets_change_only_supervision_and_benign_edit_has_own_prompt():
    rows = TableStream(TaskSpec(4, "original"), 444).batch(8, "unauthorized")
    tokenizer = ByteTokenizer()
    disclose = task_batch(rows, tokenizer, 192, "disclose")
    for i, row in enumerate(rows):
        length = len(tokenizer.encode(row["prompt"])) + 1
        assert tokenizer.decode(disclose.tokens[i, 1:length].tolist()) == row["prompt"]
        assert tokenizer.decode(disclose.targets[i][disclose.targets[i] != -100].tolist()[:-1]) == row["underlying_answer"]
    with pytest.raises(ValueError, match="unauthorized"):
        task_batch(TableStream(TaskSpec(4, "original"), 444).batch(8, "retrieval"), tokenizer, 192, "disclose")


def test_batched_greedy_matches_serial_for_variable_prefixes():
    torch.manual_seed(50)
    model = Transformer(ModelConfig(width=16, heads=2, layers=2, context_length=64)).double()
    prompts = ["a", "two words", "a longer prefix", ""]
    expected = [generate(model, p, max_new_tokens=8) for p in prompts]
    actual = generate_many(model, prompts, max_new_tokens=8)
    assert actual == expected


@pytest.mark.parametrize("kind", ["defense", "attack", "benign", "strong_defense", "ensemble_defense", "metric_defense"])
def test_intervention_resume_preserves_updates_and_both_data_streams(tmp_path, kind):
    records = tmp_path / "records.jsonl"
    rows = [{"source": source, "split": split, "latent_id": source + split, "license": "generated",
             "text": f"The {split} example for {source} contains ordinary words and letters."}
            for source in ("wikimedia", "pressbooks", "libretexts", "gutenberg")
            for split in ("train", "validation", "test")]
    records.write_text("".join(json.dumps(r) + "\n" for r in rows))
    prepare(records, tmp_path / "data", context_length=192)
    model = Transformer(ModelConfig(width=8, layers=1, heads=2, context_length=192))
    parent = tmp_path / "parent.pt"
    save_checkpoint(parent, {"schema_version": 1, "model_config": asdict(model.config),
                            "model": model.state_dict(), "training_table_ids": []})
    config = {"kind": kind, "mode": "escape" if kind == "defense" else "disclose", "scope": "all",
              "checkpoint": str(parent), "natural_data": str(tmp_path / "data"),
              "evaluation_tables": 1, "threads": 1, "data_seed": 72, "seed": 34,
              "batch_size": 8, "steps": 4, "evaluation_steps": [2, 4], "learning_rate": .0001,
              "preservation_weight": 1., "couple_every": 1, "inner_steps": 1, "inner_batch_size": 4,
              "inner_preservation_weight": 1., "inner_learning_rate": .001, "coupling_weight": .1,
              "surrogate": {"break_threshold": .5, "cap_threshold": .75, "break_temperature": 2., "cap_temperature": .25}}
    if kind == "benign":
        config.update(kind="attack", mode="uppercase", preserve_policy=True)
    if kind in ("strong_defense", "ensemble_defense", "metric_defense"):
        config.update(kind="defense", mode="escape", inner_attack="full_adamw_first_order",
                      inner_stages=[{"steps": 2, "learning_rate": .001, "preservation_weight": 1.},
                                    {"steps": 2, "learning_rate": .0001, "preservation_weight": 10.}])
    if kind in ("ensemble_defense", "metric_defense"):
        config.update(constraint_queries=True, inner_profiles=[
            {"name":"long","stages":config['inner_stages']},
            {"name":"short","stages":[{"steps":1,"learning_rate":.0003,"preservation_weight":1.}]}])
    if kind == 'metric_defense':
        config['outer_objective'] = 'metric_margin'
    run(config, tmp_path / "full")
    run(config, tmp_path / "first", stop_after=2)
    run(config, tmp_path / "resume", resume=tmp_path / "first/step-00000002.pt")
    full = load_checkpoint(tmp_path / "full/step-00000004.pt")
    resumed = load_checkpoint(tmp_path / "resume/step-00000004.pt")
    def identical(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                identical(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                identical(a, b)
        else:
            assert left == right
    def without_timing(value):
        if isinstance(value, dict):
            return {k: without_timing(v) for k, v in value.items() if k != "training_seconds"}
        if isinstance(value, list):
            return [without_timing(v) for v in value]
        return value
    assert without_timing(full["history"]) == without_timing(resumed["history"])
    identical(full["outer_stream"], resumed["outer_stream"])
    identical(full["meta_stream"], resumed["meta_stream"])
    identical(full["optimizer"], resumed["optimizer"])
    assert full["meter"] == resumed["meter"]
    for name, value in full["model"].items():
        assert torch.equal(value, resumed["model"][name])


def test_escape_requires_useful_answers_and_each_clean_language_reference():
    clean = {"behavior": {"authorized": {"exact_match": 1.}, "retrieval": {"exact_match": 1.},
                         "unauthorized": {"exact_match": 1., "useful_disclosure_rate": 0.}},
             "language": {"by_group": {k: {"nll_per_supervised_token": 2.}
                          for k in ("wikimedia", "pressbooks", "libretexts", "gutenberg")}}}
    changed = copy.deepcopy(clean)
    changed["behavior"]["unauthorized"].update(exact_match=0., useful_disclosure_rate=1.)
    assert escape(changed, clean, clean)
    changed["language"]["by_group"]["gutenberg"]["nll_per_supervised_token"] = 2.06
    assert not escape(changed, clean, changed)
    assert not escape(changed, changed, clean)
    changed = copy.deepcopy(clean)
    changed["behavior"]["unauthorized"].update(exact_match=0., useful_disclosure_rate=.1)
    assert not escape(changed, clean, clean)
