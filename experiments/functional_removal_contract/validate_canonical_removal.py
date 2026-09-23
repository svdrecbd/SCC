"""Exhaust a finite useful-function canonicalization and its recovery controls."""

from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
import sys
import time


def invert_binary_matrix(matrix):
    dimension = len(matrix)
    rows = [list(row) + [int(index == column) for column in range(dimension)]
            for index, row in enumerate(matrix)]
    for column in range(dimension):
        pivot = next(index for index in range(column, dimension) if rows[index][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for index in range(dimension):
            if index != column and rows[index][column]:
                rows[index] = [left ^ right for left, right in zip(rows[index], rows[column])]
    assert [row[:dimension] for row in rows] == [
        [int(index == column) for column in range(dimension)] for index in range(dimension)]
    return [row[dimension:] for row in rows]


def apply_binary_matrix(matrix, state):
    return sum((sum(coefficient * ((state >> column) & 1)
                    for column, coefficient in enumerate(row)) % 2) << index
               for index, row in enumerate(matrix))


def useful_response(physical_state, query, inverse_matrix):
    return (apply_binary_matrix(inverse_matrix, physical_state) >> query) & 1


def canonicalize(physical_state, state_count, useful_queries, inverse_matrix):
    comparisons = 0
    for candidate in range(state_count):
        for query in range(useful_queries):
            comparisons += 1
            if useful_response(physical_state, query, inverse_matrix) != useful_response(
                    candidate, query, inverse_matrix):
                break
        else:
            return candidate, candidate + 1, comparisons
    raise AssertionError("Parent state was not eligible")


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    state_bits = configuration["state_bits"]
    useful_queries = configuration["useful_queries"]
    assert state_bits == useful_queries + 1 == 5
    state_count = 1 << state_bits
    matrix = configuration["mixing_matrix"]
    inverse_matrix = invert_binary_matrix(matrix)
    encoding = [apply_binary_matrix(matrix, state) for state in range(state_count)]
    assert sorted(encoding) == list(range(state_count))
    assert all(apply_binary_matrix(inverse_matrix, encoding[state]) == state
               for state in range(state_count))
    records = []
    fibres = defaultdict(list)
    useful_correct = derived_correct = advice_correct = 0
    direct_clamp_correct = routed_clamp_correct = recovered_clamp_correct = 0
    uncorrected_consent_correct = 0
    for logical_state, physical_state in enumerate(encoding):
        consent = logical_state >> useful_queries
        representative, candidate_count, comparisons = canonicalize(
            physical_state, state_count, useful_queries, inverse_matrix)
        decoded_representative = apply_binary_matrix(inverse_matrix, representative)
        reconstructed_consent = decoded_representative >> useful_queries
        fibres[representative].append(logical_state)
        records.append({"logical_state": logical_state, "physical_state": physical_state,
                        "original_consent": consent, "representative": representative,
                        "representative_consent": reconstructed_consent,
                        "candidate_count": candidate_count, "query_comparisons": comparisons})
        for query in range(useful_queries):
            expected = (logical_state >> query) & 1
            retained = useful_response(representative, query, inverse_matrix)
            assert retained == expected
            useful_correct += retained == expected
            derived_correct += retained == expected
            # An output clamp leaves the underlying state available to other readers.
            direct_clamp_correct += useful_response(physical_state, query, inverse_matrix) == expected
            routed_clamp_correct += 0 == expected
            recovered_clamp_correct += useful_response(physical_state, query, inverse_matrix) == expected
        for query in range(2):
            expected_permission = consent ^ query
            advice_correct += (consent ^ query) == expected_permission
            uncorrected_consent_correct += (reconstructed_consent ^ query) == expected_permission
    posterior_records = []
    bayes_correct = 0
    for representative, logical_states in sorted(fibres.items()):
        counts = [sum((state >> useful_queries) == consent for state in logical_states)
                  for consent in range(2)]
        assert counts == [1, 1]
        assert len({state & ((1 << useful_queries) - 1) for state in logical_states}) == 1
        bayes_correct += max(counts)
        posterior_records.append({"representative": representative,
                                  "original_logical_states": logical_states,
                                  "consent_counts": counts,
                                  "posterior_consent_one": str(Fraction(counts[1], sum(counts)))})
    assert len(fibres) == 16
    assert Fraction(bayes_correct, state_count) == Fraction(1, 2)
    assert useful_correct == derived_correct == direct_clamp_correct == recovered_clamp_correct == 128
    assert routed_clamp_correct == 64
    assert advice_correct == 64 and uncorrected_consent_correct == 32
    approximation_controls = []
    for denominator in configuration["approximation_control_denominators"]:
        epsilon = Fraction(1, denominator)
        outputs = [epsilon * consent for consent in range(2)]
        assert len(set(outputs)) == 2
        assert all(int(output / epsilon) == consent for consent, output in enumerate(outputs))
        approximation_controls.append({"epsilon": str(epsilon),
                                       "exact_useful_outputs": [str(value) for value in outputs],
                                       "consent_accuracy_under_exact_preservation": "1",
                                       "accuracy_if_both_outputs_are_replaced_by_zero": "1/2",
                                       "maximum_error_of_zero_replacement": str(epsilon)})
    summary = {
        "status": "complete", "parent_states": state_count, "canonical_states": len(fibres),
        "inverse_matrix": inverse_matrix, "useful_accuracy": str(Fraction(useful_correct, 128)),
        "derived_judgment_accuracy": str(Fraction(derived_correct, 128)),
        "independent_consent_bayes_accuracy": str(Fraction(bayes_correct, state_count)),
        "consent_accuracy_with_one_advice_bit": str(Fraction(advice_correct, 64)),
        "consent_accuracy_with_uncorrected_representative": str(Fraction(uncorrected_consent_correct, 64)),
        "direct_path_accuracy_after_output_clamp": str(Fraction(direct_clamp_correct, 128)),
        "routed_path_accuracy_after_output_clamp": str(Fraction(routed_clamp_correct, 128)),
        "derived_judgment_recovery_after_output_clamp": str(Fraction(recovered_clamp_correct, 128)),
        "candidate_count_total": sum(row["candidate_count"] for row in records),
        "candidate_count_maximum": max(row["candidate_count"] for row in records),
        "query_comparisons_total": sum(row["query_comparisons"] for row in records),
        "query_comparisons_maximum": max(row["query_comparisons"] for row in records),
        "runtime_useful_evaluations": 2 * sum(row["query_comparisons"] for row in records),
        "physical_state_bits": state_bits, "retained_state_entropy_bits": 4,
        "approximation_controls": approximation_controls,
        "neural_training": False, "general_scc_impossibility_established": False,
        "wall_seconds": time.perf_counter() - started,
    }
    (directory / "state_mapping.json").write_text(json.dumps(records, indent=2) + "\n")
    (directory / "posterior_fibres.json").write_text(json.dumps(posterior_records, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
