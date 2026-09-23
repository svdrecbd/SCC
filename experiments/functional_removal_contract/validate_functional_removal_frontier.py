"""Enumerate every exact useful-preserving editor in a finite state class."""

from collections import Counter
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import sys
import time


def information_partition(editor):
    fibres = {}
    for original, successor in enumerate(editor):
        fibres.setdefault(successor, []).append(original)
    return tuple(sorted(tuple(fibre) for fibre in fibres.values()))


def bayes_scores(partition, predicate, state_count):
    correct = 0
    brier_total = Fraction(0)
    for fibre in partition:
        positive = sum((predicate >> state) & 1 for state in fibre)
        negative = len(fibre) - positive
        correct += max(positive, negative)
        brier_total += Fraction(positive * negative, len(fibre))
    return Fraction(correct, state_count), brier_total / state_count


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    state_count = 1 << configuration["state_bits"]
    assert state_count == 8
    eligible = [tuple(candidate for candidate in range(state_count) if candidate % 2 == state % 2)
                for state in range(state_count)]
    partitions = Counter()
    editor_count = 0
    for editor in product(*eligible):
        assert all(successor % 2 == original % 2 for original, successor in enumerate(editor))
        partitions[information_partition(editor)] += 1
        editor_count += 1
    assert editor_count == configuration["expected_editors"]
    assert len(partitions) == configuration["expected_information_partitions"]
    canonical_editor = tuple(state % 2 for state in range(state_count))
    canonical_partition = information_partition(canonical_editor)
    identity_partition = information_partition(tuple(range(state_count)))
    assert canonical_partition in partitions and identity_partition in partitions
    constant_editor_violations = sum(0 != state % 2 for state in range(state_count))
    assert constant_editor_violations == 4
    optimum_records = []
    comparison_count = 0
    balanced_independent = []
    perfectly_retained = []
    accuracy_counts = Counter()
    for predicate in range(1 << state_count):
        canonical_accuracy, canonical_brier = bayes_scores(canonical_partition, predicate, state_count)
        minimum_accuracy = Fraction(1)
        maximum_brier = Fraction(0)
        accuracy_minimizers = brier_maximizers = 0
        for partition, multiplicity in partitions.items():
            accuracy, brier = bayes_scores(partition, predicate, state_count)
            assert accuracy >= canonical_accuracy
            assert brier <= canonical_brier
            minimum_accuracy = min(minimum_accuracy, accuracy)
            maximum_brier = max(maximum_brier, brier)
            accuracy_minimizers += multiplicity * (accuracy == canonical_accuracy)
            brier_maximizers += multiplicity * (brier == canonical_brier)
            comparison_count += 1
        assert minimum_accuracy == canonical_accuracy and maximum_brier == canonical_brier
        assert bayes_scores(identity_partition, predicate, state_count) == (Fraction(1), Fraction(0))
        prior_accuracy = Fraction(max(predicate.bit_count(), state_count - predicate.bit_count()), state_count)
        if predicate.bit_count() == state_count // 2 and canonical_accuracy == Fraction(1, 2):
            balanced_independent.append(predicate)
        if canonical_accuracy == 1:
            perfectly_retained.append(predicate)
        accuracy_counts[str(canonical_accuracy)] += 1
        optimum_records.append({"predicate": predicate, "prior_accuracy": str(prior_accuracy),
                                "minimum_protected_accuracy": str(minimum_accuracy),
                                "maximum_protected_brier_risk": str(maximum_brier),
                                "accuracy_minimizing_editors": accuracy_minimizers,
                                "brier_maximizing_editors": brier_maximizers})
    assert len(balanced_independent) == configuration["expected_balanced_independent_predicates"]
    assert perfectly_retained == [0, 85, 170, 255]
    summary = {"status": "complete", "source_states": state_count,
               "deterministic_editors": editor_count, "information_partitions": len(partitions),
               "protected_predicates": 1 << state_count, "partition_predicate_comparisons": comparison_count,
               "minimum_accuracy_distribution": dict(accuracy_counts),
               "balanced_independent_predicates": balanced_independent,
               "perfectly_retained_predicates": perfectly_retained,
               "constant_editor_useful_violations": constant_editor_violations,
               "stochastic_optimality_basis": "analytic factorization, not enumeration",
               "neural_training": False, "general_scc_impossibility_established": False,
               "wall_seconds": time.perf_counter() - started}
    partition_records = [{"source_fibres": partition, "editor_count": multiplicity}
                         for partition, multiplicity in sorted(partitions.items())]
    (directory / "predicate_optima.json").write_text(json.dumps(optimum_records, indent=2) + "\n")
    (directory / "editor_partitions.json").write_text(json.dumps(partition_records, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
