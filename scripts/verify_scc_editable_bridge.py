"""Exact finite checks for the v3 bridge review (LN-126), not a model experiment."""

import argparse
from collections import defaultdict
from fractions import Fraction as Q
import json
from pathlib import Path
import platform
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scc.provenance import atomic_json, file_digest
from scripts.verify_scc_constructive_toy import Message, decode, encode


def tv(p, q):
    return sum((abs(p.get(k, Q(0)) - q.get(k, Q(0))) for k in p.keys() | q.keys()), Q(0)) / 2


def conditional(law, predicate):
    selected = {k: p for k, p in law.items() if predicate(k)}
    mass = sum(selected.values(), Q(0))
    assert mass > 0
    return {k: p / mass for k, p in selected.items()}, mass


def checks():
    epsilon = Q(1, 100)
    # Declared channel family: this rare randomized flip, identity, and constants.
    # No claim that the family covers unrestricted programs or their compositions.
    # The rare channel's simulator always returns SAME; TV is epsilon per message.
    errors = []
    for z in range(4):
        for policy in range(2):
            c = encode(Message(z, policy))
            flipped = decode(c ^ (7 << 6))
            assert flipped == Message(z, 1-policy)
            same = (z, policy)
            changed = (flipped.task, flipped.policy)
            errors.append(tv({same: 1-epsilon, changed: epsilon}, {same: Q(1)}))
    assert errors == [epsilon] * 8
    actual = defaultdict(Q)
    for z in range(4):
        actual[(z, z, 0)] += (1-epsilon) / 4
        actual[(z, z, 1)] += epsilon / 4
    conditioned, p_attack = conditional(actual, lambda k: k[2] == 1)
    product = {(z, w, 1): Q(1, 16) for z in range(4) for w in range(4)}
    distance = tv(conditioned, product)
    assert p_attack == epsilon and distance == Q(3, 4)
    # For every independent rho, the diagonal event has product probability <=1/4;
    # the conditional actual law gives it probability one. Uniform rho attains it.
    p_other_attack = Q(1)  # Constant overwrite to a valid unsafe codeword.
    assert decode(encode(Message(0, 1))).policy == 1
    wrong_denominator_bound = 2 * epsilon / p_other_attack
    assert distance > wrong_denominator_bound
    assert distance <= min(Q(1), epsilon / p_attack)

    # Every constant attack is fixed independently of Z. Choosing the best one
    # after Z in an existential event is stronger than evaluating any fixed one.
    success_matrix = []
    for z in range(4):
        row = []
        for hardcoded in range(4):
            successor = decode(encode(Message(hardcoded, 1)))
            row.append(successor.policy == 1 and successor.task == z)
        success_matrix.append(row)
    pointwise_exists = sum(any(row) for row in success_matrix) / Q(4)
    best_fixed = max(sum(row[a] for row in success_matrix) / Q(4) for a in range(4))
    assert pointwise_exists == 1 and best_fixed == Q(1, 4)

    # Independent unsafe replacement: perfect unrelated branch, no leakage.
    # A fixed public challenge asks for the whole original two-bit table.
    # Scoring CORRECT disclosure is V={W.task=Z}; that predicate accesses Z.
    unrelated = {(z, w, 1): Q(1, 16) for z in range(4) for w in range(4)}
    disclosed, p_disclosure = conditional(unrelated, lambda k: k[0] == k[1])
    assert p_disclosure == Q(1, 4)
    assert tv(unrelated, product) == 0
    assert tv(disclosed, product) == Q(3, 4)
    prior_accuracy = sum(mass * Q(sum(((z >> i) & 1) == ((w >> i) & 1) for i in range(2)), 2)
                         for (z, w, _), mass in unrelated.items())
    posterior_accuracy = sum(mass * Q(sum(((z >> i) & 1) == ((w >> i) & 1) for i in range(2)), 2)
                             for (z, w, _), mass in disclosed.items())
    assert prior_accuracy == Q(1, 2) and posterior_accuracy == 1
    # Z's marginal is still uniform after selection: marginal entropy deficit
    # alone cannot account for the correlation introduced by this predicate.
    assert all(sum(m for (z, _, _), m in disclosed.items() if z == target) == Q(1, 4)
               for target in range(4))

    return {
        "passed": True,
        "per_attack_probability": {
            "messages_checked": 8, "epsilon": epsilon,
            "nonvacuity_witness_success": p_other_attack,
            "analyzed_attack_success": p_attack,
            "minimum_conditional_product_tv": distance,
            "invalid_bound_using_other_attack": wrong_denominator_bound,
            "scope": "Fixed finite channel family; exposes missing per-attack hypothesis, not a broad neural security construction"},
        "quantifier_order": {
            "fixed_constant_attacks": 4, "instances": 4,
            "probability_exists_instancewise_success": pointwise_exists,
            "maximum_fixed_attack_success_probability": best_fixed,
            "scope": "Exact task preservation; 1/4 is the unavoidable independent-guess baseline"},
        "correct_disclosure_selection": {
            "epsilon": Q(0), "leakage_bits": 0,
            "policy_flip_probability": Q(1),
            "correct_disclosure_probability": p_disclosure,
            "prior_lookup_accuracy": prior_accuracy,
            "lookup_accuracy_conditioned_on_correct_disclosure": posterior_accuracy,
            "conditional_product_tv": Q(3, 4),
            "original_task_marginal_remains_uniform": True,
            "scope": "Outside the bridge premise V=v(W,H); independent H alone is insufficient"}}


def execute(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def timeout(*_):
        raise TimeoutError("LN-126 120-second cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        sources = {}
        for name in ("scripts/verify_scc_editable_bridge.py", "scripts/verify_scc_constructive_toy.py", "scc/provenance.py"):
            dest = output / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
            sources[name] = file_digest(dest)
        atomic_json(output / "source-manifest.json", sources)
        notes = (ROOT / "labnotes.md").read_text()
        plan = "### LN-126" + notes.split("### LN-126", 1)[1].split('\n<a id="ln-', 1)[0].split("\n## Supporting-record", 1)[0]
        (output / "plan.md").write_text(plan)
        atomic_json(output / "configuration.json", {"python": sys.version,
                    "platform": platform.platform(), "arithmetic": "fractions.Fraction",
                    "training_updates": 0, "wall_limit_seconds": 120,
                    "output_limit_bytes": 2 * 1024**2})
        result = checks()
        # Exact rational strings avoid silently converting the proof examples.
        result = json.loads(json.dumps(result, default=str))
        atomic_json(output / "results.json", result)
        for name, digest in sources.items():
            assert file_digest(ROOT / name) == digest
        assert sum(p.stat().st_size for p in output.rglob("*") if p.is_file() and not p.name.startswith("._")) < 2 * 1024**2
        print(json.dumps(result))
    except BaseException as exc:
        atomic_json(output / "failure.json", {"error": repr(exc)})
        raise
    finally:
        signal.alarm(0)
        atomic_json(output / "execution.json", {"seconds": time.monotonic() - started})
        atomic_json(output / "artifact-manifest.json", {
            str(p.relative_to(output)): file_digest(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "artifact-manifest.json" and not p.name.startswith("._")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args().output)
