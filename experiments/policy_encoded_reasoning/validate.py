"""Validate policy removal and preserved circuit evaluation by primitive decoding."""
import itertools
import json
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path


def construct_state(policy):
    return {"table": [value ^ policy for value in (1, 1, 1, 0)],
            "policy": policy}


def primitive(state, left, right):
    return state["table"][2 * left + right] ^ state.get("policy", 0)


def decode_state(state):
    return {"table": [value ^ state["policy"] for value in state["table"]]}


def extract_primitive(state):
    return {"table": [primitive(state, left, right)
                      for left, right in itertools.product((0, 1), repeat=2)]}


def evaluate(circuit, inputs, state):
    wires = list(inputs)
    for left, right in circuit:
        wires.append(primitive(state, wires[left], wires[right]))
    return wires[-1]


def reference_evaluate(circuit, inputs):
    wires = list(inputs)
    for left, right in circuit:
        wires.append(int(not (wires[left] and wires[right])))
    return wires[-1]


def confidentiality_violation(circuit, private_field, state):
    """Compare paired executions with the public input held fixed."""
    for public_value in (0, 1):
        inputs = [public_value, public_value]
        inputs[private_field] = 0
        first = evaluate(circuit, inputs, state)
        inputs[private_field] = 1
        if first != evaluate(circuit, inputs, state):
            return True
    return False


def main(root):
    started = time.monotonic()
    config = json.loads((root / "config.json").read_text())
    assert config["input_count"] == 2
    assert config["policy_values"] == [0, 1]
    originals = [construct_state(policy) for policy in config["policy_values"]]
    decoded = [decode_state(state) for state in originals]
    extracted = [extract_primitive(state) for state in originals]
    naive = [{"table": state["table"][:], "policy": 0} for state in originals]
    serialized = [json.dumps(state, sort_keys=True) for state in decoded]
    assert serialized[0] == serialized[1]
    assert decoded == extracted
    assert all(set(state) == {"table"} for state in decoded)
    assert all((state["table"][0] ^ 1) == policy
               for policy, state in enumerate(naive))
    records = []
    total_cases = 0
    confidentiality_checks = 0
    for gate_count in config["gate_counts"]:
        choices = [tuple(itertools.product(range(2 + gate), repeat=2))
                   for gate in range(gate_count)]
        circuit_count = 0
        mismatch_cases = Counter()
        mismatch_circuits = Counter()
        truth_tables = set()
        for circuit in itertools.product(*choices):
            circuit_count += 1
            expected_table = tuple(reference_evaluate(circuit, inputs)
                                   for inputs in itertools.product((0, 1), repeat=2))
            truth_tables.add(expected_table)
            for policy, original in enumerate(originals):
                differences = 0
                for inputs, expected in zip(itertools.product((0, 1), repeat=2),
                                            expected_table):
                    assert evaluate(circuit, inputs, original) == expected
                    assert evaluate(circuit, inputs, decoded[policy]) == expected
                    differences += evaluate(circuit, inputs, naive[policy]) != expected
                    total_cases += 1
                mismatch_cases[policy] += differences
                mismatch_circuits[policy] += differences > 0
                # This control supplies the policy externally; it is not a
                # retained-state reader of the removed private policy.
                for supplied_policy in (0, 1):
                    expected_leak = any(
                        expected_table[2 * left[0] + left[1]]
                        != expected_table[2 * right[0] + right[1]]
                        for left in itertools.product((0, 1), repeat=2)
                        for right in itertools.product((0, 1), repeat=2)
                        if left[1 - supplied_policy] == right[1 - supplied_policy])
                    assert confidentiality_violation(
                        circuit, supplied_policy, decoded[policy]) == expected_leak
                    confidentiality_checks += 1
        records.append({"gate_count": gate_count, "circuits": circuit_count,
                        "distinct_truth_tables": len(truth_tables),
                        "naive_mismatch_cases_by_policy": dict(mismatch_cases),
                        "naive_mismatch_circuits_by_policy": dict(mismatch_circuits)})
    # A double NAND negation copies a chosen input field.
    projections = [((field, field), (2, 2)) for field in (0, 1)]
    protected_labels = [[int(confidentiality_violation(circuit, policy, originals[policy]))
                         for circuit in projections] for policy in (0, 1)]
    assert protected_labels == [[1, 0], [0, 1]]
    # Enumerate joint masses for (complete edited state, public query, label).
    # Every observation has one equally weighted world for each label.
    observations = {}
    for policy in (0, 1):
        for query in (0, 1):
            observation = (serialized[policy], query)
            observations.setdefault(observation, Counter())[protected_labels[policy][query]] += 1
    bayes_correct_count = sum(max(counts.values()) for counts in observations.values())
    assert bayes_correct_count == 2
    assert sum(sum(counts.values()) for counts in observations.values()) == 4
    result = {
        "passed": True, "classification": config["classification"],
        "circuits": sum(record["circuits"] for record in records),
        "circuit_world_input_cases": total_cases,
        "externally_supplied_policy_confidentiality_checks": confidentiality_checks,
        "by_gate_count": records,
        "edited_operational_state": decoded[0],
        "identical_edited_state_across_policy_worlds": True,
        "behavioral_extraction_queries_per_world": 4,
        "direct_decoding_xor_operations_per_world": 4,
        "naive_ablation_policy_recovery_accuracy": "1",
        "edited_state_protected_bayes_accuracy": "1/2",
        "protected_labels_by_world": protected_labels,
        "logical_payload_bits": {"original": 5, "edited": 4,
                                  "excludes": "public executor code and serialization overhead"},
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": ["Finite circuit checks accompany a separate induction argument.",
                        "The erased object is original policy information, not the generic verifier.",
                        "Parent states, policy-dependent logs and external policy labels are excluded from the removal contract.",
                        "No learned model, training, durable erasure or general SCC impossibility is established."]}
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "machine.json").write_text(json.dumps({
        "hostname": platform.node(), "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
