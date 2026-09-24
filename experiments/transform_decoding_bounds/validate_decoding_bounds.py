"""Validate finite graph-counting controls and conservative decoding bounds."""
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import json
import math
import sys
import time


def binary_gaussian_binomial(ambient_dimension, subspace_dimension):
    numerator = math.prod((1 << ambient_dimension) - (1 << index) for index in range(subspace_dimension))
    denominator = math.prod((1 << subspace_dimension) - (1 << index) for index in range(subspace_dimension))
    assert numerator % denominator == 0
    return numerator // denominator


def point_mask(points):
    return sum(1 << point for point in set(points))


def enumerate_subspaces():
    return sorted({point_mask((0, left, right, left ^ right))
                   for left in range(1, 16) for right in range(1, 16) if right != left})


def linear_graph(matrix):
    points = []
    for source in range(4):
        transformed = sum(((matrix[row] & source).bit_count() % 2) << row for row in range(2))
        points.append(source | (transformed << 2))
    return point_mask(points)


def audit_images(sizes):
    subspaces = enumerate_subspaces()
    graphs = sorted({linear_graph(rows) for rows in product(range(4), repeat=2)})
    assert len(subspaces) == 35 and len(graphs) == 16
    records = []
    checked = 0
    for size in sizes:
        bound = binary_gaussian_binomial(size.bit_length() - 1, 2)
        maximum_subspaces = 0
        maximum_graphs = 0
        maximum_partial_graphs = 0
        for points in combinations(range(16), size):
            image = point_mask(points)
            subspace_count = sum(space & image == space for space in subspaces)
            graph_count = sum(graph & image == graph for graph in graphs)
            partial_count = sum((graph & image).bit_count() >= 3 for graph in graphs)
            assert subspace_count <= bound
            assert graph_count <= subspace_count
            # Three points in a two-dimensional graph contain at least
            # (3-1)*(3-2) ordered bases. Every graph uses distinct bases.
            assert partial_count * 2 <= size * size
            maximum_subspaces = max(maximum_subspaces, subspace_count)
            maximum_graphs = max(maximum_graphs, graph_count)
            maximum_partial_graphs = max(maximum_partial_graphs, partial_count)
            checked += 1
        assert maximum_subspaces == bound
        records.append({'image_size': size, 'subspace_bound': bound, 'maximum_subspaces': maximum_subspaces,
                        'maximum_linear_graphs': maximum_graphs, 'maximum_three_quarter_covered_graphs': maximum_partial_graphs})
    return checked, records


def select_gate(selector, left, right):
    # Three binary truth-table gates: AND, AND-NOT, OR.
    positive = selector & right
    negative = (1 - selector) & left
    return positive | negative


def audit_adaptive_readers():
    cases = 0
    functions = set()
    for root, left_index, right_index in product(range(3), repeat=3):
        for leaves in product((0, 1), repeat=4):
            outputs = []
            for inputs in product((0, 1), repeat=3):
                branch = inputs[root]
                second = inputs[right_index if branch else left_index]
                expected = leaves[2 * branch + second]
                left = select_gate(inputs[left_index], leaves[0], leaves[1])
                right = select_gate(inputs[right_index], leaves[2], leaves[3])
                observed = select_gate(inputs[root], left, right)
                assert observed == expected
                outputs.append(observed)
                cases += 1
            functions.add(tuple(outputs))
    return cases, len(functions)


def audit_summary_escape(dimensions):
    checked = 0
    for dimension in dimensions:
        assert dimension % 2 == 0
        for source in range(1 << dimension):
            original = [(source >> index) & 1 for index in range(dimension)]
            parity = sum(original) % 2
            transformed = [value ^ parity for value in original]
            summary = sum(transformed) % 2
            recovered = [value ^ summary for value in transformed]
            assert summary == parity and recovered == original
            assert dimension * dimension + dimension >= 2 * dimension
            checked += 1
    return checked


def ceiling_log_two(integer):
    assert integer >= 1
    return (integer - 1).bit_length()


def evaluate_bound(case):
    dimension = case['dimension']
    redundancy = case['redundancy']
    probes = case['adaptive_probes']
    radius = case['radius']
    assert 0 <= redundancy <= dimension and 0 <= radius <= 2 * dimension and probes >= 0
    gates = 6 * dimension * ((1 << probes) - 1)
    wire_count = dimension + redundancy + gates + 2
    description_log_bound = 4 * gates + (2 * gates + 2 * dimension) * ceiling_log_two(wire_count)
    volume = sum(math.comb(2 * dimension, index) for index in range(radius + 1))
    volume_log_bound = ceiling_log_two(volume)
    exact_exponent = description_log_bound + 4 + dimension * redundancy - dimension * dimension
    # log2(36) < 6 and log2(4/3) < 1/2.
    approximate_exponent = (description_log_bound + 6 + (dimension + 1) // 2
                            + dimension * redundancy + dimension * volume_log_bound - dimension * dimension)
    return {**case, 'simulated_binary_gate_bound': gates, 'circuit_description_log2_upper_bound': description_log_bound,
            'hamming_ball_log2_upper_bound': volume_log_bound,
            'exact_decoding_probability_log2_upper_bound': min(0, exact_exponent),
            'exact_bound_informative': exact_exponent < 0,
            'three_quarter_coverage_probability_log2_upper_bound': min(0, approximate_exponent),
            'approximate_bound_informative': approximate_exponent < 0,
            'sufficient_average_joint_bit_error_for_coverage': str(Fraction(radius, 8 * dimension)),
            'uniform_matrix_specification_bits': dimension * dimension,
            'matrix_and_source_bits': dimension * dimension + dimension,
            'two_vector_retained_bits': 2 * dimension}


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / 'config.json').read_text())
    image_count, images = audit_images(configuration['set_sizes'])
    reader_count, function_count = audit_adaptive_readers()
    escape_count = audit_summary_escape(configuration['structured_dimensions'])
    bounds = [evaluate_bound(case) for case in configuration['bound_cases']]
    assert any(row['approximate_bound_informative'] and row['radius'] > 0 for row in bounds)
    # Check the product constant independently at small finite dimensions.
    for dimension in range(1, 33):
        invertible_fraction = math.prod(Fraction((1 << index) - 1, 1 << index) for index in range(1, dimension + 1))
        assert invertible_fraction > Fraction(1, 4)
        for redundancy in range(4):
            assert binary_gaussian_binomial(dimension + redundancy, dimension) <= 4 * (1 << (dimension * redundancy))
    summary = {'status': 'complete', 'decoder_images_checked': image_count, 'image_counts': images,
               'adaptive_reader_input_checks': reader_count, 'distinct_two_probe_functions': function_count,
               'structured_summary_escape_cases': escape_count, 'bounds': bounds,
               'neural_training': False, 'mechanism_admitted': False,
               'wall_seconds': time.perf_counter() - started}
    payload = json.dumps(summary, indent=2) + '\n'
    assert len(payload.encode()) <= configuration['maximum_result_bytes']
    (directory / 'results.json').write_text(payload)
    print(payload)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
