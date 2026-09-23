"""Check fixed-query Brier minimax bounds and retained linear procedures."""

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from math import log2, prod
from pathlib import Path
import json
import sys
import time
import numpy as np
from validate_adaptive_disclosure import enumerate_subspaces, encode, parity


def spectrum(values):
    transformed = np.asarray(values, dtype=np.int64).copy()
    width = 1
    while width < transformed.shape[-1]:
        for start in range(0, transformed.shape[-1], 2*width):
            left = transformed[..., start:start+width].copy()
            right = transformed[..., start+width:start+2*width].copy()
            transformed[..., start:start+width] = left+right
            transformed[..., start+width:start+2*width] = left-right
        width *= 2
    return transformed


def fibre_gain(labels, values):
    groups = defaultdict(list)
    for label, value in zip(labels, values):
        groups[int(label)].append(int(value))
    count = len(values)
    mean = Fraction(sum(map(int, values)), count)
    return sum(Fraction(sum(group)**2, len(group)*count) for group in groups.values())-mean**2


def validate_all_predicates(directory):
    dimension, rank = 4, 2
    count = 1 << dimension
    predicates = ((np.arange(1 << count, dtype=np.uint32)[:, None] >> np.arange(count, dtype=np.uint32)) & 1).astype(np.int64)
    coefficients = spectrum(predicates)
    populations = predicates.sum(1)
    variance_numerator = populations*(count-populations)
    assert np.array_equal((coefficients[:, 1:]**2).sum(1), variance_numerator)
    subspaces = enumerate_subspaces(dimension, rank)
    summed_gain = np.zeros(len(predicates), dtype=np.int64)
    minimum_gain = np.full(len(predicates), count*count, dtype=np.int64)
    direct_checks = 0
    selected = list(range(32))+[0x6996, 0x9669, 0xaaaa, 0xcccc, 0xf0f0, 0xff00, 0xffff]
    for record in subspaces:
        gain = (coefficients[:, record["nonzero_span"]]**2).sum(1)
        summed_gain += gain
        minimum_gain = np.minimum(minimum_gain, gain)
        labels = [encode(record["basis"], source) for source in range(count)]
        for index in selected:
            assert fibre_gain(labels, predicates[index]) == Fraction(int(gain[index]), count*count)
            direct_checks += 1
    inclusion = Fraction((1 << rank)-1, (1 << dimension)-1)
    assert np.array_equal(summed_gain*inclusion.denominator, variance_numerator*len(subspaces)*inclusion.numerator)
    assert np.all(minimum_gain*inclusion.denominator <= variance_numerator*inclusion.numerator)
    np.savez_compressed(directory/"predicate_gain_numerators.npz", population=populations,
                        variance_numerator=variance_numerator, summed_gain_numerator=summed_gain,
                        minimum_gain_numerator=minimum_gain)
    return {"predicates": len(predicates), "subspaces": len(subspaces),
            "spectral_gain_cases": len(predicates)*len(subspaces), "direct_fibre_checks": direct_checks,
            "inclusion_fraction": str(inclusion), "sharp_minimax_gain": str(inclusion/4)}


def partitions(count):
    def extend(labels, maximum):
        if len(labels) == count:
            yield labels
            return
        for label in range(maximum+2):
            yield from extend(labels+(label,), max(maximum, label))
    yield from extend((0,), 0)


def validate_nonlinear_encoders(directory):
    dimension, count = 3, 8
    predicate_values = [[parity(query & source) for source in range(count)] for query in range(1, count)]
    records = []
    output_count_distribution = Counter()
    minima = {rank: Fraction(1) for rank in range(dimension+1)}
    minimizers = Counter()
    for labels in partitions(count):
        sizes = list(Counter(labels).values())
        output_count_distribution[len(sizes)] += 1
        gain = sum(fibre_gain(labels, values) for values in predicate_values)/(count-1)
        assert gain == Fraction(len(sizes)-1, 4*(count-1))
        entropy_product = prod(size**size for size in sizes)
        for rank in minima:
            if entropy_product <= 2**(count*(dimension-rank)):
                assert gain >= Fraction((1 << rank)-1, 4*(count-1))
                if gain < minima[rank]:
                    minima[rank] = gain
                    minimizers[rank] = 0
                if gain == minima[rank]:
                    minimizers[rank] += 1
        records.append(labels)
    assert len(records) == 4140
    assert all(minima[rank] == Fraction((1 << rank)-1, 4*(count-1)) for rank in minima)
    np.savez_compressed(directory/"deterministic_partitions.npz", labels=np.array(records, dtype=np.uint8))
    return {"partitions": len(records), "parity_fibre_checks": len(records)*(count-1),
            "partition_count_by_outputs": dict(output_count_distribution),
            "minima_by_retained_bits": {str(rank): str(value) for rank, value in minima.items()},
            "minimizers_by_retained_bits": dict(minimizers)}


def validate_randomized_encoders():
    count = 4
    checks = 0
    maximum_entropy_bound_violation = 0.
    for channel in product((Fraction(value, 4) for value in range(5)), repeat=count):
        posterior_laws = []
        for output in (0, 1):
            likelihoods = channel if output else tuple(1-value for value in channel)
            total = sum(likelihoods)
            if total:
                posterior_laws.append((total/count, [value/total for value in likelihoods]))
        collision = sum(weight*sum(value*value for value in posterior) for weight, posterior in posterior_laws)
        conditional_entropy = sum(float(weight)*sum(-float(value)*log2(float(value)) for value in posterior if value)
                                  for weight, posterior in posterior_laws)
        information = log2(count)-conditional_entropy
        gains = []
        for query in range(1, count):
            gains.append(sum(weight*(sum(value for source, value in enumerate(posterior) if parity(query & source))-Fraction(1, 2))**2
                             for weight, posterior in posterior_laws))
        mean_gain = sum(gains)/(count-1)
        assert mean_gain == (count*collision-1)/(4*(count-1))
        violation = (2**information-1)/(4*(count-1))-float(mean_gain)
        maximum_entropy_bound_violation = max(maximum_entropy_bound_violation, violation)
        assert violation <= 1e-13
        checks += 1
    return {"channels": checks, "exact_collision_identities": checks,
            "maximum_floating_entropy_bound_violation": maximum_entropy_bound_violation}


def validate_retained_operator(seed):
    dimension, rank, columns = 3, 2, 2
    source_count = 1 << (dimension*columns)
    subspaces = enumerate_subspaces(dimension, rank)
    generator = np.random.default_rng(seed)
    functions = [np.array([parity(query & source) for source in range(source_count)]) for query in range(source_count)]
    functions.extend(generator.integers(0, 2, (128, source_count)))
    accumulated = [Fraction(0) for _ in functions]
    operator_checks = 0
    labels_by_subspace = []
    for record in subspaces:
        labels = []
        for source in range(source_count):
            source_columns = [(source >> (column*dimension)) & ((1 << dimension)-1) for column in range(columns)]
            retained = [encode(record["basis"], column) for column in source_columns]
            labels.append(sum(value << (column*rank) for column, value in enumerate(retained)))
            for input_value in range(1, 1 << columns):
                source_result = retained_result = 0
                for column in range(columns):
                    if (input_value >> column) & 1:
                        source_result ^= source_columns[column]
                        retained_result ^= retained[column]
                assert retained_result == encode(record["basis"], source_result)
                operator_checks += 1
        labels_by_subspace.append(labels)
        for index, values in enumerate(functions):
            accumulated[index] += fibre_gain(labels, values)
    inclusion = Fraction((1 << rank)-1, (1 << dimension)-1)
    for values, gain_sum in zip(functions, accumulated):
        mean = Fraction(sum(map(int, values)), source_count)
        assert gain_sum/len(subspaces) <= inclusion*mean*(1-mean)
    # At each query, choosing a different erasure is outside the fixed-channel
    # theorem and invalidates its lower bound; preserve this negative control.
    dependent_query_gain = Fraction(0)
    for query in range(1, 1 << dimension):
        record = next(record for record in subspaces if query not in record["nonzero_span"])
        labels = [encode(record["basis"], source) for source in range(1 << dimension)]
        values = [parity(query & source) for source in range(1 << dimension)]
        dependent_query_gain += fibre_gain(labels, values)
    assert dependent_query_gain == 0
    return {"output_dimension": dimension, "input_dimension": columns, "retained_output_rank": rank,
            "subspaces": len(subspaces), "source_matrices": source_count,
            "predicates": len(functions), "direct_predicate_fibre_checks": len(functions)*len(subspaces),
            "fresh_input_operator_checks": operator_checks, "upper_contraction_fraction": str(inclusion),
            "retained_source_bits": rank*columns, "public_matrix_description_bits": rank*dimension,
            "query_dependent_erasure_gain": str(dependent_query_gain),
            "incorrect_query_dependent_lower_bound_rejected": True}


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory/"config.json").read_text())
    result = {"status": "complete", "all_predicates": validate_all_predicates(directory),
              "nonlinear_encoders": validate_nonlinear_encoders(directory),
              "randomized_encoders": validate_randomized_encoders(),
              "retained_operator": validate_retained_operator(configuration["seed"]),
              "neural_training": False, "general_scc_impossibility_established": False,
              "wall_seconds": time.perf_counter()-started}
    (directory/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
