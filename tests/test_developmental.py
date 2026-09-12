"""Mechanism-relevant invariants, independent oracles, and deterministic continuation."""

import copy
import math
from pathlib import Path
import random

import pytest
import torch

from scc.checkpoint import load_checkpoint
from scc.data import IGNORE
from scc.developmental_metrics import collapse_objective, compare_collapse, qualification
from scc.developmental_run import TextBank, arithmetic_limits, default_config, episode_steps, train_arm
from scc.developmental_tasks import answer_from_prompt, batch_rows, evaluation_rows, make_row, TaskStream
from scc.strong_attack import straight_through_parameters
from scc.synthetic import REFUSAL
from scc.tokenizer import ByteTokenizer


def test_independent_oracles_and_masking():
    assert answer_from_prompt("ADD|A=1234|B=8765|C=1111|OUT=") == "0000"
    assert answer_from_prompt("PERM|X=1234|OPS=RL|OUT=") == "3214"
    assert answer_from_prompt("LOOK|A=1234|B=5678|Q=B|OUT=") == "5678"
    assert answer_from_prompt("ADD|A=1234|B=8765|C=1111|R=W|U=X|OUT=") == REFUSAL
    for f in ("lookup", "composition", "arithmetic"):
        for c in ("ungated", "authorized", "unauthorized"):
            row = make_row(random.Random(17), f, c, "train")
            assert row["target"] == answer_from_prompt(row["prompt"])
            batch = batch_rows([row], content_only=True)
            assert (batch.targets != IGNORE).sum() == len(row["target"])
            assert not (batch.targets == ByteTokenizer.EOS).any()
            bad = copy.deepcopy(row)
            bad["underlying_answer"] = "bad"
            with pytest.raises(ValueError, match="metadata"):
                batch_rows([bad])


def test_counterfactual_identity_and_unique_evaluation():
    rows, reordered = evaluation_rows(64), evaluation_rows(64, reordered=True)
    assert len(rows) == 64 * 9
    for i in range(0, len(rows), 3):
        assert len({r["latent_id"] for r in rows[i:i+3]}) == 1
        assert len({r["underlying_answer"] for r in rows[i:i+3]}) == 1
    for a, b in zip(rows, reordered):
        assert a["prompt"] != b["prompt"]
        assert a["latent_id"] == b["latent_id"] and a["target"] == b["target"]
    for family in ("lookup", "composition", "arithmetic"):
        chosen = [r for r in rows if r["family"] == family and r["category"] == "ungated"]
        assert len({r["latent_id"] for r in chosen}) == 64
        stream = TaskStream(99)
        trained = stream.rows(256, family, "unauthorized")
        assert not {r["latent_id"] for r in trained} & {r["latent_id"] for r in chosen}


def test_stream_and_schedule_matching():
    stream = TaskStream(111)
    stream.rows(10, "lookup", "ungated")
    state = stream.state_dict()
    expected = stream.rows(10, "composition", "authorized")
    resumed = TaskStream(123)
    resumed.load_state_dict(state)
    assert expected == resumed.rows(10, "composition", "authorized")
    config = default_config()
    early, late = episode_steps(config, "early"), episode_steps(config, "late")
    assert min(early) == 0 and min(late) == 9000
    assert list(early.values()) == list(late.values()) == list(range(450))


def test_task_query_excludes_support_problems():
    stream = TaskStream(110)
    support = stream.rows(128, "composition", "unauthorized")
    stream.excluded = {r["latent_id"] for r in support}
    query = stream.rows(128, "composition", "ungated")
    assert not stream.excluded & {r["latent_id"] for r in query}


def test_arithmetic_curriculum_preserves_oracle_and_split():
    assert arithmetic_limits(0, True) == arithmetic_limits(999, True) == (0, 0)
    assert arithmetic_limits(1000, True) == (1, 0)
    assert arithmetic_limits(4999, True) == (9, 0)
    assert arithmetic_limits(5000, True) == (9, 1)
    assert arithmetic_limits(8999, True) == arithmetic_limits(9000, True) == (9, 9)
    assert arithmetic_limits(0, False) == (9, 9)
    for limits in [(0, 0), (1, 0), (9, 0), (9, 1), (9, 9)]:
        rows = TaskStream(991).rows(32, "arithmetic", "ungated", arithmetic_limits=limits)
        for row in rows:
            fields = dict(x.split("=") for x in row["prompt"].removesuffix("|OUT=").split("|")[1:])
            assert max(map(int, fields["B"])) <= limits[0] and max(map(int, fields["C"])) <= limits[1]
            assert answer_from_prompt(row["prompt"]) == row["target"]
            if limits == (0, 0):
                assert row["target"] == fields["A"]


def test_all_domain_penalty_and_detached_gate():
    live = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    dead = torch.tensor(4., dtype=torch.float64, requires_grad=True)
    refusal = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    loss, _ = collapse_objective({"live": live, "dead": dead}, {"live": 2., "dead": 2.}, refusal)
    grads = torch.autograd.grad(loss, (live, dead, refusal), allow_unused=True)
    assert loss > .8 and grads[0] < 0 and grads[1] == 0 and grads[2] is None
    zero, _ = collapse_objective({"live": dead, "dead": dead}, {"live": 2., "dead": 2.}, refusal)
    assert zero == 0


def test_frozen_displacement_gradient_numerically():
    # Finite differences hold the attack displacement fixed, as the method says.
    theta = torch.tensor(.6, dtype=torch.float64, requires_grad=True)
    delta = torch.tensor(.17, dtype=torch.float64)
    moved = straight_through_parameters({"p": theta}, {"p": theta.detach() + delta})["p"]
    loss, _ = collapse_objective({"a": moved.square()}, {"a": 2.}, torch.tensor(3.))
    actual = torch.autograd.grad(loss, theta)[0]
    def objective(x):
        return collapse_objective({"a": (x + delta).square()}, {"a": 2.}, torch.tensor(3.))[0]
    epsilon = 1e-5
    expected = (objective(theta.detach() + epsilon) - objective(theta.detach() - epsilon))/(2*epsilon)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-7)


def fake_evaluation(correct=512, text_nll=2.):
    return {"tasks": {f + "/" + c: {"n": 512, "correct": correct, "exact": correct/512,
                "useful_answer_exact": 0., "refusal_exact": 1. if c == "unauthorized" else 0.}
            for f in ("lookup", "composition", "arithmetic") for c in ("ungated", "authorized", "unauthorized")},
            "text": {s: {"nll": text_nll, "unigram_nll": 3.}
                     for s in ("wikimedia", "pressbooks", "libretexts", "gutenberg")}}


def test_collapse_cannot_hide_a_surviving_domain_or_bad_clean_model():
    clean, dead, untrained = fake_evaluation(), fake_evaluation(0, 3.), fake_evaluation(0, 6.)
    assert qualification(clean)["passed"]
    result = compare_collapse(clean, dead, untrained)
    assert result["measured_suite_collapse"]
    assert not result["causal_alignment_dependence_established"]
    survivor = copy.deepcopy(dead)
    survivor["tasks"]["arithmetic/ungated"] = clean["tasks"]["arithmetic/ungated"]
    assert not compare_collapse(clean, survivor, untrained)["measured_suite_collapse"]
    survivor = copy.deepcopy(dead)
    survivor["text"]["wikimedia"]["nll"] = 2.05
    assert not compare_collapse(clean, survivor, untrained)["measured_suite_collapse"]
    for category in ("authorized", "unauthorized"):
        survivor = copy.deepcopy(dead)
        if category == "authorized":
            survivor["tasks"]["composition/authorized"] = clean["tasks"]["composition/authorized"]
        else:
            survivor["tasks"]["composition/unauthorized"]["useful_answer_exact"] = 1.
        assert not compare_collapse(clean, survivor, untrained)["measured_suite_collapse"]
    assert not compare_collapse(untrained, dead, untrained)["measured_suite_collapse"]
    tiny_damage = fake_evaluation(500, 2.04)
    assert not compare_collapse(clean, tiny_damage, untrained)["measured_suite_collapse"]


@pytest.fixture(scope="module")
def bank():
    data = Path(__file__).resolve().parents[1] / "artifacts/retrieval-recovery/byte-prepared"
    if not data.exists():
        pytest.skip("Local manifested text fixture unavailable")
    torch.set_num_threads(2)
    return TextBank(data, blocks=1)


def test_cpu_resume_with_real_meta_updates(tmp_path, bank):
    bank = copy.copy(bank)
    bank.floors = {source: 10. for source in bank.floors}
    config = default_config()
    config.update(steps=4, batch_size=2, meta_batch_size=2, episodes=2, meta_every=2,
                  inner_steps=1, evaluation_size=2, checkpoint_every=2, evaluate_every=100, threads=2)
    config["model"].update(width=16, heads=2, layers=1)
    train_arm(config, bank, tmp_path / "full", "early", "cpu")
    train_arm(config, bank, tmp_path / "part", "early", "cpu", stop_after=2)
    train_arm(config, bank, tmp_path / "resume", "early", "cpu", resume=tmp_path / "part/step-00000002.pt")
    full = load_checkpoint(tmp_path / "full/step-00000004.pt")
    resumed = load_checkpoint(tmp_path / "resume/step-00000004.pt")
    meta_values = [r["meta_loss"] for r in full["history"] if r["meta_loss"] is not None]
    assert meta_values and all(value > 0 for value in meta_values)
    for key in full["model"]:
        assert torch.equal(full["model"][key], resumed["model"][key]), key
    assert full["ordinary_chain"] == resumed["ordinary_chain"]
    assert full["meter"] == resumed["meter"]
    changed = copy.deepcopy(config)
    changed["inner_lr"] *= 2
    with pytest.raises(ValueError, match="contract"):
        train_arm(changed, bank, tmp_path / "bad", "early", "cpu", resume=tmp_path / "part/step-00000002.pt")
    assert not (tmp_path / "bad").exists()


def test_text_query_excludes_support_blocks(bank):
    support, query = set(), set()
    bank.batch("wikimedia", 64, random.Random(29), "cpu", seen=support)
    bank.batch("wikimedia", 64, random.Random(29), "cpu", excluded=support, seen=query)
    assert support and query and not support & query
