"""Validate fixed-query minimax coverage and adaptive disclosure witnesses."""

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import json
import sys
import time


def parity(value):
    return value.bit_count() % 2


def encode(basis, source):
    return sum(parity(row & source) << index for index, row in enumerate(basis))


def enumerate_subspaces(dimension, rank):
    records = []
    for pivots in combinations(range(dimension), rank):
        free_positions = [(row, column) for row, pivot in enumerate(pivots)
                          for column in range(pivot + 1, dimension) if column not in pivots]
        for assignment in range(1 << len(free_positions)):
            basis = [1 << pivot for pivot in pivots]
            for index, (row, column) in enumerate(free_positions):
                basis[row] |= ((assignment >> index) & 1) << column
            span = {0}
            for row in basis:
                span |= {value ^ row for value in tuple(span)}
            assert len(span) == 1 << rank
            records.append({"basis": basis, "nonzero_span": sorted(span - {0})})
    assert len({tuple(row["nonzero_span"]) for row in records}) == len(records)
    return records


def query_laws(dimension):
    queries = range(1, 1 << dimension)
    weights = {
        "uniform": {query: 1 for query in queries},
        "coordinate": {query: int(query.bit_count() == 1) for query in queries},
        "odd_weight": {query: query.bit_count() % 2 for query in queries},
        "sparse_weighted": {query: 1 << (dimension - query.bit_count()) for query in queries},
        "concentrated": {query: int(query == (1 << dimension) - 1) for query in queries},
    }
    return {name: {query: Fraction(weight, sum(mapping.values())) for query, weight in mapping.items()}
            for name, mapping in weights.items()}


def disclosure_output(source, query, secret):
    register = 0
    observations = 0
    for coordinate in range(query.bit_length()):
        if (query >> coordinate) & 1:
            register ^= (source >> coordinate) & 1
            observations += 1
    return secret if register else 0, observations


def validate_vector_case(configuration):
    dimension = configuration["dimension"]
    rank = configuration["rank"]
    source_count = 1 << dimension
    subspaces = enumerate_subspaces(dimension, rank)
    assert len(subspaces) == configuration["expected_subspaces"]
    incidence = Counter(query for record in subspaces for query in record["nonzero_span"])
    fraction = Fraction((1 << rank) - 1, (1 << dimension) - 1)
    expected_incidence = fraction * len(subspaces)
    assert expected_incidence.denominator == 1
    assert set(incidence) == set(range(1, source_count))
    assert all(count == expected_incidence for count in incidence.values())
    posterior_checks = policy_checks = reversed_criterion_rejections = 0
    policy_observation_counts = Counter()
    for record in subspaces:
        basis = record["basis"]
        span = set(record["nonzero_span"])
        fibres = defaultdict(list)
        for source in range(source_count):
            fibres[encode(basis, source)].append(source)
        assert len(fibres) == 1 << rank
        for query in range(1, source_count):
            correct = 0
            for sources in fibres.values():
                positive = sum(parity(source & query) for source in sources)
                negative = len(sources) - positive
                if query in span:
                    assert positive == 0 or negative == 0
                else:
                    assert positive == negative
                correct += max(positive, negative)
                posterior_checks += 1
            actual = Fraction(correct, source_count)
            expected = Fraction(1) if query in span else Fraction(1, 2)
            assert actual == expected
            reversed_criterion_rejections += actual != (Fraction(1, 2) if query in span else Fraction(1))
        for index, query in enumerate(basis):
            for source in range(source_count):
                output_zero, observations = disclosure_output(source, query, 0)
                output_one, other_observations = disclosure_output(source, query, 1)
                recovered = (encode(basis, source) >> index) & 1
                assert (output_zero != output_one) == recovered
                assert observations == other_observations == query.bit_count()
                policy_observation_counts[observations] += 1
                policy_checks += 1
    law_records = []
    for name, law in query_laws(dimension).items():
        masses = [sum(law[query] for query in record["nonzero_span"]) for record in subspaces]
        average = sum(masses) / len(masses)
        minimum = min(masses)
        assert average == fraction and minimum <= fraction
        if name == "uniform":
            assert all(mass == fraction for mass in masses)
        minimizing = masses.index(minimum)
        law_records.append({"law": name, "mean_subspace_mass": str(average),
                            "minimum_subspace_mass": str(minimum),
                            "minimum_protected_accuracy": str(Fraction(1, 2) + minimum / 2),
                            "minimizing_basis": subspaces[minimizing]["basis"],
                            "law_probabilities": {str(query): str(probability) for query, probability in law.items()}})
    return {"dimension": dimension, "rank": rank, "subspaces": subspaces,
            "incidence_per_nonzero_query": int(expected_incidence),
            "fixed_law_minimax_accuracy": str(Fraction(1, 2) + fraction / 2),
            "posterior_fibres_checked": posterior_checks, "adaptive_policy_cases": policy_checks,
            "adaptive_policy_observation_counts": dict(policy_observation_counts),
            "reversed_criterion_rejections": reversed_criterion_rejections,
            "source_dependent_retained_bits": rank, "explicit_public_matrix_bits": rank * dimension,
            "query_laws": law_records}


def apply_source_matrix(matrix_rows, input_value):
    return sum(parity(row & input_value) << index for index, row in enumerate(matrix_rows))


def retain_operator(basis, matrix_rows, input_dimension):
    return tuple(sum(parity(row & apply_source_matrix(matrix_rows, 1 << column)) << column
                     for column in range(input_dimension)) for row in basis)


def validate_operator_case(configuration):
    dimension = configuration["dimension"]
    rank = configuration["rank"]
    input_dimension = configuration["input_dimension"]
    subspaces = enumerate_subspaces(dimension, rank)
    source_matrices = list(product(range(1 << input_dimension), repeat=dimension))
    posterior_checks = useful_checks = adaptive_checks = 0
    policy_observation_counts = Counter()
    for record in subspaces:
        basis = record["basis"]
        span = set(record["nonzero_span"])
        fibres = defaultdict(list)
        for matrix_rows in source_matrices:
            retained = retain_operator(basis, matrix_rows, input_dimension)
            source_bits = sum(row << (index * input_dimension) for index, row in enumerate(matrix_rows))
            fibres[retained].append(matrix_rows)
            for input_value in range(1, 1 << input_dimension):
                actual = apply_source_matrix(retained, input_value)
                expected = encode(basis, apply_source_matrix(matrix_rows, input_value))
                assert actual == expected
                useful_checks += 1
                for index, query in enumerate(basis):
                    matrix_query = sum(input_value << (row * input_dimension)
                                       for row in range(dimension) if (query >> row) & 1)
                    public_zero, observations = disclosure_output(source_bits, matrix_query, 0)
                    public_one, other_observations = disclosure_output(source_bits, matrix_query, 1)
                    assert (public_zero != public_one) == ((actual >> index) & 1)
                    assert observations == other_observations == query.bit_count() * input_value.bit_count()
                    policy_observation_counts[observations] += 1
                    adaptive_checks += 1
        assert len(fibres) == 1 << (rank * input_dimension)
        for query in range(1, 1 << dimension):
            for input_value in range(1, 1 << input_dimension):
                for matrices in fibres.values():
                    positive = sum(parity(query & apply_source_matrix(matrix_rows, input_value)) for matrix_rows in matrices)
                    negative = len(matrices) - positive
                    if query in span:
                        assert positive == 0 or negative == 0
                    else:
                        assert positive == negative
                    posterior_checks += 1
    return {"dimension": dimension, "rank": rank, "input_dimension": input_dimension,
            "subspaces": len(subspaces), "source_matrices": len(source_matrices),
            "posterior_fibres_checked": posterior_checks, "new_input_operator_checks": useful_checks,
            "adaptive_policy_cases": adaptive_checks,
            "adaptive_policy_observation_counts": dict(policy_observation_counts),
            "source_dependent_retained_bits": rank * input_dimension,
            "explicit_public_matrix_bits": rank * dimension}


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    vector_records = [validate_vector_case(case) for case in configuration["subspace_cases"]]
    operator_record = validate_operator_case(configuration["operator_case"])
    large_records = []
    for case in configuration["large_cases"]:
        dimension = case["dimension"]
        rank = dimension - case["erased_directions"]
        advantage = Fraction((1 << rank) - 1, 2 * ((1 << dimension) - 1))
        large_records.append({"dimension": dimension, "rank": rank,
                              "retained_entropy_fraction": str(Fraction(rank, dimension)),
                              "fixed_law_minimax_advantage": str(advantage),
                              "fixed_law_minimax_advantage_decimal": float(advantage),
                              "adaptive_row_query_accuracy": "1",
                              "source_dependent_retained_bits": rank,
                              "explicit_public_matrix_bits": rank * dimension})
    summary = {"status": "complete", "vector_cases": [
        {key: value for key, value in record.items() if key not in {"subspaces", "query_laws"}}
        for record in vector_records], "operator_case": operator_record, "large_cases": large_records,
        "neural_training": False, "general_scc_impossibility_established": False,
        "wall_seconds": time.perf_counter() - started}
    (directory / "vector_cases.json").write_text(json.dumps(vector_records, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
