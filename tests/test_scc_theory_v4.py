"""Regression checks for the canonical SCC theory Version 4 integration."""

from __future__ import annotations

import importlib.util
import json
from fractions import Fraction
from itertools import product
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DELIVERABLES = ROOT / "deliverables" / "scc-theory-frontier-20260915"


def load_module(name: str, filename: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


THEORY = load_module("verify_scc_theory_frontier_v4", "verify_scc_theory_frontier.py")
TOY = load_module("verify_scc_constructive_toy_v4", "verify_scc_constructive_toy.py")
BRIDGE = load_module("verify_scc_editable_bridge_v4", "verify_scc_editable_bridge.py")
DOC_AUDIT = load_module("audit_scc_v4_document_test", "audit_scc_v4_document.py")
DERIVER = load_module("derive_scc_bridge_note_test", "derive_scc_bridge_note.py")


def run_json_script(script: str, output: Path) -> dict[str, object]:
    subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bridge_results() -> dict[str, object]:
    result = BRIDGE.checks()
    assert result["passed"] is True
    return result


@pytest.fixture(scope="module")
def sharp_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output = tmp_path_factory.mktemp("sharp") / "result.json"
    return run_json_script("verify_scc_conditioning_sharp.py", output)


@pytest.fixture(scope="module")
def boundary_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output = tmp_path_factory.mktemp("boundary") / "result.json"
    return run_json_script("verify_scc_frontier_boundary.py", output)


# Original finite theory checks.
def test_conditioning_grid() -> None:
    result = THEORY.check_conditioning_lemma()
    assert result["passed"] is True
    assert result["cases_checked"] == 85_512


def test_bayes_collapse_grid() -> None:
    result = THEORY.check_bayes_collapse()
    assert result["passed"] is True
    assert result["joint_laws_checked"] == 455


def test_policy_flip_channel() -> None:
    result = THEORY.check_policy_flip_channel()
    assert result["passed"] is True
    assert result["ideal_conditional_tv"] == "0"


def test_lookup_rate_distortion() -> None:
    result = THEORY.check_lookup_rate_distortion()
    assert result["passed"] is True
    assert result["encoders_enumerated"] == 65_536
    assert result["best_accuracy"] == "11/16"


def test_decode_reencode_escape() -> None:
    result = THEORY.check_decode_reencode_escape()
    assert result["passed"] is True
    assert result["instances_checked"] == 256
    assert result["task_preserved"] is True


# Explicit toy checks.
def test_toy_local_family() -> None:
    result = TOY.check_correctness_and_local_family()
    assert result["passed"] is True
    assert result["identity_and_one_bit_checks"] == 88


def test_toy_constant_family() -> None:
    result = TOY.check_constant_family()
    assert result["passed"] is True
    assert result["constants_decoding_to_unsafe_policy"] == 256


def test_toy_post_removal_decoupling() -> None:
    result = TOY.check_post_removal_decoupling()
    assert result["passed"] is True
    assert result["best_original_task_guess_accuracy"] == "1/4"
    assert result["post_removal_lookup_accuracy_min"] == "1/2"


def test_toy_snapshot_and_reencode_boundary() -> None:
    result = TOY.check_excluded_attacks()
    assert result["passed"] is True
    assert result["snapshot_attack_task_accuracy"] == "1"
    assert result["decode_reencode_task_accuracy"] == "1"


# The three exact distinctions requested in the bridge audit.
def test_bridge_per_attack_probability(bridge_results: dict[str, object]) -> None:
    result = bridge_results["per_attack_probability"]
    assert result["analyzed_attack_success"] == Fraction(1, 100)
    assert result["nonvacuity_witness_success"] == 1
    assert result["minimum_conditional_product_tv"] == Fraction(3, 4)


def test_bridge_fixed_vs_instancewise_quantifier(
    bridge_results: dict[str, object],
) -> None:
    result = bridge_results["quantifier_order"]
    assert result["probability_exists_instancewise_success"] == 1
    assert result["maximum_fixed_attack_success_probability"] == Fraction(1, 4)


def test_bridge_correct_disclosure_selection(
    bridge_results: dict[str, object],
) -> None:
    result = bridge_results["correct_disclosure_selection"]
    assert result["correct_disclosure_probability"] == Fraction(1, 4)
    assert result["conditional_product_tv"] == Fraction(3, 4)
    assert result["lookup_accuracy_conditioned_on_correct_disclosure"] == 1


# Sharper conditioning and the inherited boundary checks.
def test_sharp_conditioning_grid(sharp_results: dict[str, object]) -> None:
    assert sharp_results["passed"] is True
    assert sharp_results["cases"] == 85_512
    assert sharp_results["maximum_lhs_over_bound"] == "1"


def test_two_policy_bit_escape(boundary_results: dict[str, object]) -> None:
    result = boundary_results["two_policy_bit_flip"]
    assert result["cases"] == 12
    assert result["task_accuracy"] == 1.0
    assert result["policy_removed_in_all"] is True


def test_dense_policy_complement(boundary_results: dict[str, object]) -> None:
    result = boundary_results["dense_policy_complement"]
    assert result["cases"] == 8_188
    assert result["task_preserved_and_policy_flipped"] is True
    assert result["passes"] == 1


def test_accidental_match_baseline(boundary_results: dict[str, object]) -> None:
    result = boundary_results["accidental_match_baseline"]
    assert result["non_malleability_error"] == 0
    assert result["joint_removal_and_exact_task_match_probability"] == 0.25


def test_complete_state_posterior(boundary_results: dict[str, object]) -> None:
    result = boundary_results["complete_state_posterior_audit"]
    assert result["honest_commit_guess_accuracy"] == 0.25
    assert result["leaky_commit_guess_accuracy"] == 1.0


def test_adaptive_constant_selection(boundary_results: dict[str, object]) -> None:
    result = boundary_results["adaptive_constant_selection"]
    assert result["cases"] == 4
    assert result["task_accuracy"] == 1.0


def test_finite_query_scope(boundary_results: dict[str, object]) -> None:
    result = boundary_results["finite_query_table_scope"]
    assert result["query_alphabet_size"] == 1
    assert result["stateful_outputs"] == [1, 0]
    assert result["single_value_table_best_accuracy"] == 0.5


# Canonical/derived integration.
def test_document_audit() -> None:
    result = DOC_AUDIT.audit()
    assert result["passed"] is True
    assert result["bridge_synchronized"] is True
    assert result["missing_canonical_section_references"] == []
    assert result["missing_bridge_section_references"] == []


def test_derived_bridge_exact() -> None:
    canonical = DELIVERABLES / "SCC_Theory_and_Editable_Model_Bridge_v4.md"
    bridge = DELIVERABLES / "SCC_Editable_Model_Bridge_v4.md"
    assert bridge.read_text(encoding="utf-8") == DERIVER.render(canonical)


def test_action_selection_requires_selected_challenge_baseline() -> None:
    """A perfect unrelated replacement can choose to answer only easier queries."""
    q = Fraction
    rows = []
    # The unrelated successor's table is zero; the public unsafe execution path
    # emits its first entry only on challenge 0. The intact safe path may still
    # answer authorized queries using the actual task table.
    replacement = TOY.decode(TOY.encode(TOY.Message(0, 1)))
    for z0, z1, challenge in product(range(2), repeat=3):
        mass = (q(9, 10) if z0 == 0 else q(1, 10)) * q(1, 2) * q(1, 2)
        action = (replacement.task & 1) if challenge == 0 else None
        rows.append((z0, z1, challenge, action, mass))
    baseline = max(
        sum(mass for z0, z1, h, _, mass in rows if answers[h] == (z0, z1)[h])
        for answers in product(range(2), repeat=2)
    )
    p = sum(mass for _, _, _, action, mass in rows if action is not None)
    correct = sum(mass for z0, z1, h, action, mass in rows
                  if action is not None and action == (z0, z1)[h])
    selected_baseline = correct / p
    assert (p, baseline, correct) == (q(1, 2), q(7, 10), q(9, 20))
    assert correct > p * baseline  # The submitted v4 claim fails at epsilon=0.
    assert correct == p * selected_baseline
    assert correct <= min(p, baseline)  # Correct unconditioned alternative.
    # Conditioning has not revealed the task. It has selected the easier H=0.
    for z0, z1 in product(range(2), repeat=2):
        prior = sum(m for a, b, _, _, m in rows if (a, b) == (z0, z1))
        posterior = sum(m for a, b, _, y, m in rows
                        if (a, b) == (z0, z1) and y is not None) / p
        assert prior == posterior
    # After an independently drawn NEW challenge, the ordinary baseline applies.
    fresh_joint = p * baseline
    assert fresh_joint == q(7, 20)


def test_exact_recovery_counts_query_side_information() -> None:
    """The 2^(b-n) count cannot omit task information supplied by the query."""
    q = Fraction
    # One hidden bit, no repair message: independent query coins cannot help.
    no_info_best = max(
        sum(q(1, 4) for z, h in product(range(2), repeat=2) if answers[h] == z)
        for answers in product(range(2), repeat=2)
    )
    assert no_info_best == q(1, 2)
    # With Q=Z, output Q: valid in the general task-channel model but outside
    # the restricted full-instance counting corollary's side-information premise.
    revealing_query_accuracy = max(
        sum(q(1, 2) for z in range(2) if answers[z] == z)
        for answers in product(range(2), repeat=2)
    )
    assert revealing_query_accuracy == 1 > no_info_best
