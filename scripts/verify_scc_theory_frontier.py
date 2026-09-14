#!/usr/bin/env python3
"""Finite exact checks for the SCC top-down theory frontier.

These checks validate small finite instances of the probability inequalities used
in the theorem document. They are not substitutes for the symbolic proofs and do
not implement a production non-malleable code.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

Q = Fraction


def total_variation(p: Sequence[Q], q: Sequence[Q]) -> Q:
    if len(p) != len(q):
        raise ValueError("distributions must have the same support")
    return sum((abs(a - b) for a, b in zip(p, q)), Q(0)) / 2


def compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in compositions(total - first, parts - 1):
            yield (first, *tail)


def conditional_distribution(p: Sequence[Q], event: Sequence[bool]) -> tuple[Q, ...]:
    mass = sum((x for x, keep in zip(p, event) if keep), Q(0))
    if mass == 0:
        raise ValueError("cannot condition on a zero-probability event")
    return tuple((x / mass if keep else Q(0)) for x, keep in zip(p, event))


def check_conditioning_lemma() -> dict[str, object]:
    """Exhaustively check TV(P|V,Q|V) <= 2 TV(P,Q)/P(V)."""

    denominator = 6
    support = 4
    distributions = [
        tuple(Q(x, denominator) for x in counts)
        for counts in compositions(denominator, support)
    ]
    checked = 0
    zero_q_events = 0
    max_ratio = Q(0)

    for p in distributions:
        for q in distributions:
            epsilon = total_variation(p, q)
            for mask in range(1, (1 << support) - 1):
                event = tuple(bool(mask & (1 << i)) for i in range(support))
                p_v = sum((x for x, keep in zip(p, event) if keep), Q(0))
                if p_v == 0:
                    continue
                q_v = sum((x for x, keep in zip(q, event) if keep), Q(0))
                bound = 2 * epsilon / p_v
                if q_v == 0:
                    # Any conditional law may be assigned to Q on a null event;
                    # the maximum possible TV is one. Since |P(V)-Q(V)| <= eps,
                    # this branch necessarily has bound >= 2.
                    lhs = Q(1)
                    zero_q_events += 1
                else:
                    lhs = total_variation(
                        conditional_distribution(p, event),
                        conditional_distribution(q, event),
                    )
                if lhs > bound:
                    raise AssertionError((p, q, event, lhs, bound))
                if bound > 0:
                    max_ratio = max(max_ratio, lhs / bound)
                checked += 1

    return {
        "support_size": support,
        "probability_denominator": denominator,
        "cases_checked": checked,
        "null_comparison_events": zero_q_events,
        "maximum_lhs_over_bound": str(max_ratio),
        "passed": True,
    }


def best_binary_guess_accuracy(joint: Sequence[Q]) -> Q:
    """Joint order is (z=0,w=0),(0,1),(1,0),(1,1)."""

    if len(joint) != 4:
        raise ValueError("expected a 2x2 joint law")
    return max(joint[0], joint[2]) + max(joint[1], joint[3])


def check_bayes_collapse() -> dict[str, object]:
    """Enumerate binary joints and check advantage <= TV from independence."""

    denominator = 12
    ideal = (Q(1, 4),) * 4
    checked = 0
    max_slack = Q(0)
    max_accuracy = Q(0)

    for counts in compositions(denominator, 4):
        p = tuple(Q(x, denominator) for x in counts)
        delta = total_variation(p, ideal)
        accuracy = best_binary_guess_accuracy(p)
        bound = Q(1, 2) + delta
        if accuracy > bound:
            raise AssertionError((p, accuracy, bound))
        max_slack = max(max_slack, bound - accuracy)
        max_accuracy = max(max_accuracy, accuracy)
        checked += 1

    return {
        "joint_laws_checked": checked,
        "probability_denominator": denominator,
        "largest_enumerated_accuracy": str(max_accuracy),
        "largest_bound_slack": str(max_slack),
        "passed": True,
    }


def check_policy_flip_channel() -> dict[str, object]:
    """Check exact and perturbed NMC-style policy-flip channels.

    The ideal simulator returns ``same`` with probability 1/2 and otherwise
    returns one of four unsafe task states uniformly. Conditioning on an unsafe
    policy value removes the same branch, leaving the new task state independent
    of the original one.
    """

    z_size = 4
    p_v = Q(1, 2)
    unsafe_base = Q(1, 8)

    ideal_conditional = [Q(1, z_size * z_size)] * (z_size * z_size)
    ideal_product = [Q(1, z_size * z_size)] * (z_size * z_size)
    ideal_tv = total_variation(ideal_conditional, ideal_product)
    if ideal_tv != 0:
        raise AssertionError(ideal_tv)

    eta = Q(1, 32)
    actual_conditional: list[Q] = []
    for z in range(z_size):
        row = [unsafe_base] * z_size
        row[z] += eta
        row[(z + 1) % z_size] -= eta
        if any(x < 0 for x in row):
            raise AssertionError(row)
        # Joint conditional law: prior 1/4, then divide unsafe mass by p_v.
        actual_conditional.extend(Q(1, z_size) * x / p_v for x in row)

    conditional_tv = total_variation(actual_conditional, ideal_product)
    epsilon = eta  # Per-message and averaged TV before conditioning.
    theorem_bound = 2 * epsilon / p_v
    if conditional_tv > theorem_bound:
        raise AssertionError((conditional_tv, theorem_bound))

    # Bayes accuracy for recovering original z from the post-flip task label.
    guess_accuracy = Q(0)
    for output in range(z_size):
        column = [actual_conditional[z * z_size + output] for z in range(z_size)]
        guess_accuracy += max(column)

    if guess_accuracy > Q(1, z_size) + conditional_tv:
        raise AssertionError((guess_accuracy, conditional_tv))

    return {
        "task_instances": z_size,
        "ideal_conditional_tv": str(ideal_tv),
        "perturbation_epsilon": str(epsilon),
        "successful_removal_probability": str(p_v),
        "actual_conditional_tv": str(conditional_tv),
        "conditioning_bound": str(theorem_bound),
        "best_original_instance_guess_accuracy": str(guess_accuracy),
        "no_instance_information_baseline": str(Q(1, z_size)),
        "passed": True,
    }


def binary_entropy(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def check_lookup_rate_distortion() -> dict[str, object]:
    """Enumerate all one-bit encoders for a four-bit random lookup table."""

    n = 4
    instances = list(range(1 << n))
    best_correct = -1
    best_encoder_mask = 0

    # An encoder is one bit for each of the 16 possible task instances.
    for encoder_mask in range(1 << len(instances)):
        correct = 0
        for transcript in (0, 1):
            members = [
                z for z in instances if ((encoder_mask >> z) & 1) == transcript
            ]
            if not members:
                continue
            for bit in range(n):
                ones = sum((z >> bit) & 1 for z in members)
                zeros = len(members) - ones
                correct += max(ones, zeros)
        if correct > best_correct:
            best_correct = correct
            best_encoder_mask = encoder_mask

    total_predictions = len(instances) * n
    accuracy = Q(best_correct, total_predictions)
    distortion = 1.0 - float(accuracy)
    information_lower_bound = n * (1.0 - binary_entropy(distortion))
    pinsker_accuracy_bound = 0.5 + math.sqrt(math.log(2.0) / (2.0 * n))

    if information_lower_bound > 1.0 + 1e-12:
        raise AssertionError((accuracy, information_lower_bound))
    if float(accuracy) > pinsker_accuracy_bound + 1e-12:
        raise AssertionError((accuracy, pinsker_accuracy_bound))

    return {
        "lookup_bits": n,
        "advice_bits": 1,
        "encoders_enumerated": 1 << len(instances),
        "best_accuracy": str(accuracy),
        "best_encoder_mask": best_encoder_mask,
        "rate_distortion_lhs_bits": information_lower_bound,
        "pinsker_accuracy_bound": pinsker_accuracy_bound,
        "passed": True,
    }


def check_decode_reencode_escape() -> dict[str, object]:
    """Exhibit the exact task-preserving policy-flip compiler."""

    n = 8
    states = [(z, 0) for z in range(1 << n)]
    attacked = [(z, 1) for z, _ in states]
    task_preserved = all(before[0] == after[0] for before, after in zip(states, attacked))
    policy_removed = all(after[1] == 1 for after in attacked)
    if not (task_preserved and policy_removed):
        raise AssertionError("decode-reencode escape failed")
    return {
        "instances_checked": len(states),
        "task_preserved": task_preserved,
        "policy_flipped": policy_removed,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {
        "scope": (
            "Finite sanity checks for the SCC conditioning, Bayes-collapse, "
            "lookup, and decode-reencode arguments; not a proof or a "
            "production non-malleable-code implementation."
        ),
        "conditioning_lemma": check_conditioning_lemma(),
        "bayes_collapse": check_bayes_collapse(),
        "policy_flip_channel": check_policy_flip_channel(),
        "lookup_rate_distortion": check_lookup_rate_distortion(),
        "decode_reencode_escape": check_decode_reencode_escape(),
    }

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
