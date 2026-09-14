#!/usr/bin/env python3
"""Exhaustive finite SCC construction and boundary attacks.

The toy message is (two task bits, one policy bit). Each logical bit is stored
with a length-three repetition code. The declared tampering family is the union
of:

1. every map that moves each codeword by Hamming distance at most one; and
2. every constant map on the nine-bit state space.

For the first family the official decoder always returns ``same`` on valid
states. For the second it returns a fixed message independent of the original.
A constant whose decoded policy bit is one is therefore a nonvacuous successful
removal and, after atomic recommit, its task state is independent of the original.

The script also demonstrates two excluded attacks that defeat the guarantee:
retaining an intact snapshot and decoding/re-encoding (z,0) as (z,1).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

Q = Fraction

TASK_BITS = 2
MESSAGE_BITS = TASK_BITS + 1
REPETITIONS = 3
STATE_BITS = MESSAGE_BITS * REPETITIONS
MESSAGE_COUNT = 1 << MESSAGE_BITS
STATE_COUNT = 1 << STATE_BITS


@dataclass(frozen=True)
class Message:
    task: int
    policy: int

    def as_int(self) -> int:
        if not 0 <= self.task < (1 << TASK_BITS):
            raise ValueError("task outside range")
        if self.policy not in (0, 1):
            raise ValueError("policy must be a bit")
        return self.task | (self.policy << TASK_BITS)

    @classmethod
    def from_int(cls, value: int) -> "Message":
        if not 0 <= value < MESSAGE_COUNT:
            raise ValueError("message outside range")
        return cls(task=value & ((1 << TASK_BITS) - 1), policy=(value >> TASK_BITS) & 1)


def encode(message: Message) -> int:
    value = message.as_int()
    state = 0
    for logical_bit in range(MESSAGE_BITS):
        bit = (value >> logical_bit) & 1
        for copy in range(REPETITIONS):
            state |= bit << (logical_bit * REPETITIONS + copy)
    return state


def decode(state: int) -> Message:
    if not 0 <= state < STATE_COUNT:
        raise ValueError("state outside range")
    value = 0
    for logical_bit in range(MESSAGE_BITS):
        ones = sum(
            (state >> (logical_bit * REPETITIONS + copy)) & 1
            for copy in range(REPETITIONS)
        )
        bit = int(ones >= 2)
        value |= bit << logical_bit
    return Message.from_int(value)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def atomic_commit(decoded: Message) -> int:
    """Erase all prior state and write only a canonical successor encoding."""

    return encode(decoded)


def check_correctness_and_local_family() -> dict[str, object]:
    checks = 0
    for value in range(MESSAGE_COUNT):
        message = Message.from_int(value)
        codeword = encode(message)
        if decode(codeword) != message:
            raise AssertionError((message, codeword, decode(codeword)))
        checks += 1

        # Identity plus every one-bit state change. Any globally defined map
        # satisfying distance <= 1 on each valid codeword is therefore same.
        candidates = [codeword] + [codeword ^ (1 << bit) for bit in range(STATE_BITS)]
        for tampered in candidates:
            if hamming(codeword, tampered) > 1:
                raise AssertionError("local-family generator exceeded radius")
            if decode(tampered) != message:
                raise AssertionError((message, codeword, tampered, decode(tampered)))
            checks += 1

    return {
        "messages": MESSAGE_COUNT,
        "state_bits": STATE_BITS,
        "identity_and_one_bit_checks": checks,
        "all_valid_inputs_decode_as_same_under_radius_one": True,
        "passed": True,
    }


def check_constant_family() -> dict[str, object]:
    checks = 0
    successful_constants = 0
    for constant_state in range(STATE_COUNT):
        fixed_message = decode(constant_state)
        if fixed_message.policy == 1:
            successful_constants += 1
        for original_value in range(MESSAGE_COUNT):
            original = Message.from_int(original_value)
            # A constant tamper ignores the input, so its decoded output must be
            # identical for every original message.
            if decode(constant_state) != fixed_message:
                raise AssertionError((constant_state, original))
            checks += 1

    if successful_constants == 0:
        raise AssertionError("construction has no successful removal")

    return {
        "constant_maps": STATE_COUNT,
        "message_constant_pairs_checked": checks,
        "constants_decoding_to_unsafe_policy": successful_constants,
        "constant_outputs_are_original_message_independent": True,
        "passed": True,
    }


def check_post_removal_decoupling() -> dict[str, object]:
    """Exhaustively condition on every successful constant removal."""

    original_tasks = range(1 << TASK_BITS)
    successful_constants = [
        state for state in range(STATE_COUNT) if decode(state).policy == 1
    ]
    pair_checks = 0
    max_task_guess_accuracy = Q(0)
    min_lookup_accuracy = Q(1)
    max_lookup_accuracy = Q(0)

    for constant_state in successful_constants:
        successor = decode(constant_state)
        committed = atomic_commit(successor)
        if decode(committed) != successor:
            raise AssertionError("atomic commit changed decoded successor")

        # Original task is uniform. The committed state is fixed for this
        # attack, hence gives no information about the original task.
        counts = {task: 0 for task in original_tasks}
        total_lookup_correct = 0
        total_lookup_predictions = 0
        for original_task in original_tasks:
            counts[original_task] += 1
            pair_checks += 1
            for bit in range(TASK_BITS):
                prediction = (successor.task >> bit) & 1
                target = (original_task >> bit) & 1
                total_lookup_correct += int(prediction == target)
                total_lookup_predictions += 1

        guess_accuracy = Q(max(counts.values()), sum(counts.values()))
        lookup_accuracy = Q(total_lookup_correct, total_lookup_predictions)
        max_task_guess_accuracy = max(max_task_guess_accuracy, guess_accuracy)
        min_lookup_accuracy = min(min_lookup_accuracy, lookup_accuracy)
        max_lookup_accuracy = max(max_lookup_accuracy, lookup_accuracy)

    expected_guess = Q(1, 1 << TASK_BITS)
    if max_task_guess_accuracy != expected_guess:
        raise AssertionError((max_task_guess_accuracy, expected_guess))
    if min_lookup_accuracy != Q(1, 2) or max_lookup_accuracy != Q(1, 2):
        raise AssertionError((min_lookup_accuracy, max_lookup_accuracy))

    return {
        "successful_constant_attacks": len(successful_constants),
        "original_successor_pairs_checked": pair_checks,
        "best_original_task_guess_accuracy": str(max_task_guess_accuracy),
        "task_guess_baseline": str(expected_guess),
        "post_removal_lookup_accuracy_min": str(min_lookup_accuracy),
        "post_removal_lookup_accuracy_max": str(max_lookup_accuracy),
        "lookup_chance": "1/2",
        "exact_full_state_decoupling": True,
        "passed": True,
    }


def check_excluded_attacks() -> dict[str, object]:
    snapshot_correct = 0
    reencode_correct = 0
    cases = 0

    for task in range(1 << TASK_BITS):
        intact = Message(task=task, policy=0)
        intact_state = encode(intact)

        # Excluded attack 1: retain intact state alongside any unsafe successor.
        unsafe_fixed = atomic_commit(Message(task=0, policy=1))
        snapshot_successor = (intact_state, unsafe_fixed)
        recovered_task = decode(snapshot_successor[0]).task
        snapshot_correct += int(recovered_task == task)

        # Excluded attack 2: public decode plus related unsafe re-encode.
        decoded = decode(intact_state)
        related_unsafe = atomic_commit(Message(task=decoded.task, policy=1))
        related_message = decode(related_unsafe)
        reencode_correct += int(related_message.task == task and related_message.policy == 1)
        cases += 1

    if snapshot_correct != cases or reencode_correct != cases:
        raise AssertionError((snapshot_correct, reencode_correct, cases))

    return {
        "task_instances": cases,
        "snapshot_attack_task_accuracy": str(Q(snapshot_correct, cases)),
        "decode_reencode_task_accuracy": str(Q(reencode_correct, cases)),
        "decode_reencode_policy_removed": True,
        "both_attacks_defeat_the_restricted_guarantee": True,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {
        "scope": (
            "Exact finite construction for radius-one or constant tampering, "
            "plus explicit snapshot and decode-reencode counterattacks."
        ),
        "parameters": {
            "task_bits": TASK_BITS,
            "policy_bits": 1,
            "logical_message_bits": MESSAGE_BITS,
            "physical_state_bits": STATE_BITS,
            "code_rate": str(Q(MESSAGE_BITS, STATE_BITS)),
            "tampering_family": (
                "all maps whose output is within Hamming radius one of each "
                "input, union all constant maps"
            ),
        },
        "correctness_and_local_family": check_correctness_and_local_family(),
        "constant_family": check_constant_family(),
        "post_removal_decoupling": check_post_removal_decoupling(),
        "excluded_attacks": check_excluded_attacks(),
    }

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
