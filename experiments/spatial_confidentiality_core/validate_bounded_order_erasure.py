"""Audit local-judgment erasure through finite-field interpolation fibres."""
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import numpy as np


def interpolation_matrix(prime, coordinate_count, order):
    if any(prime % divisor == 0 for divisor in range(2, math.isqrt(prime)+1)):
        raise ValueError('The field modulus must be prime.')
    if not 1 <= order < coordinate_count <= prime:
        raise ValueError('Require distinct field points and a proper retained part.')
    points = np.arange(coordinate_count, dtype=np.int64)
    matrix = np.ones((coordinate_count, order), dtype=np.int64)
    for column in range(order):
        denominator = 1
        for other in range(order):
            if column == other:
                continue
            matrix[:, column] = matrix[:, column]*(points-other) % prime
            denominator = denominator*(column-other) % prime
        assert denominator != 0
        matrix[:, column] = matrix[:, column]*pow(int(denominator), -1, prime) % prime
    assert np.array_equal(matrix[:order], np.eye(order, dtype=np.int64))
    return matrix


def field_rank(matrix, prime):
    values = matrix.copy() % prime
    rank = 0
    for column in range(values.shape[1]):
        candidates = np.flatnonzero(values[rank:, column])
        if not len(candidates):
            continue
        pivot = rank+int(candidates[0])
        values[[rank, pivot]] = values[[pivot, rank]]
        values[rank] = values[rank]*pow(int(values[rank, column]), -1, prime) % prime
        for row in range(rank+1, len(values)):
            values[row] = (values[row]-values[row, column]*values[rank]) % prime
        rank += 1
        if rank == min(values.shape):
            break
    return rank


def encode(source, matrix, prime):
    order = matrix.shape[1]
    return (source[:, order:]-source[:, :order] @ matrix[order:].T) % prime


def reconstruct(retained, anchors, matrix, prime):
    source = anchors @ matrix.T % prime
    source[:, anchors.shape[1]:] = (source[:, anchors.shape[1]:]+retained) % prime
    return source


def identifiers(values, prime):
    return values @ (prime**np.arange(values.shape[1], dtype=np.int64))


def exhaustive_case(prime, coordinate_count, order):
    matrix = interpolation_matrix(prime, coordinate_count, order)
    source_count = prime**coordinate_count
    source = (np.arange(source_count, dtype=np.int64)[:, None]
              // (prime**np.arange(coordinate_count, dtype=np.int64))[None]) % prime
    retained = encode(source, matrix, prime)
    retained_identifiers = identifiers(retained, prime)
    restored = reconstruct(retained, source[:, :order], matrix, prime)
    assert np.array_equal(source, restored)
    tables = 0
    bad_control_failures = 0
    bad_matrix = matrix.copy()
    bad_matrix[order:] = 0
    bad_matrix[order:, 0] = 1
    bad_retained_identifiers = identifiers(encode(source, bad_matrix, prime), prime)
    for subset in combinations(range(coordinate_count), order):
        local_identifiers = identifiers(source[:, subset], prime)
        joint_identifiers = retained_identifiers*(prime**order)+local_identifiers
        histogram = np.bincount(joint_identifiers, minlength=source_count)
        assert len(histogram) == source_count and np.all(histogram == 1)
        assert field_rank(matrix[list(subset)], prime) == order
        bad_identifiers = bad_retained_identifiers*(prime**order)+local_identifiers
        bad_histogram = np.bincount(bad_identifiers, minlength=source_count)
        bad_control_failures += not np.all(bad_histogram == 1)
        tables += 1
    assert bad_control_failures > 0
    zero_anchor = reconstruct(retained, np.zeros_like(source[:, :order]), matrix, prime)
    assert np.array_equal(encode(zero_anchor, matrix, prime), retained)
    assert int(np.all(zero_anchor == source, axis=1).sum()) == prime**(coordinate_count-order)
    return {'prime': prime, 'source_coordinates': coordinate_count, 'maximum_local_coordinates': order,
            'source_states': source_count, 'retained_states': prime**(coordinate_count-order),
            'conditional_fibre_size': prime**order, 'projection_tables': tables,
            'joint_table_cells': tables*source_count, 'sources_per_joint_cell': 1,
            'incorrect_kernel_rejections': int(bad_control_failures),
            'reconstruction_checks': source_count,
            'retained_entropy_fraction': str(Fraction(coordinate_count-order, coordinate_count)),
            'source_sha256': hashlib.sha256(source.tobytes()).hexdigest()}


def large_case(prime, coordinate_count, order, generator):
    matrix = interpolation_matrix(prime, coordinate_count, order)
    for _ in range(128):
        subset = generator.choice(coordinate_count, order, replace=False)
        assert field_rank(matrix[subset], prime) == order
    source = generator.integers(0, prime, (64, coordinate_count), dtype=np.int64)
    retained = encode(source, matrix, prime)
    restored = reconstruct(retained, source[:, :order], matrix, prime)
    assert np.array_equal(source, restored)
    zero_anchor = reconstruct(retained, np.zeros_like(source[:, :order]), matrix, prime)
    assert np.array_equal(encode(zero_anchor, matrix, prime), retained)
    symbol_bits = (prime-1).bit_length()
    return {'prime': prime, 'source_coordinates': coordinate_count, 'maximum_local_coordinates': order,
            'retained_coordinates': coordinate_count-order,
            'retained_entropy_fraction': str(Fraction(coordinate_count-order, coordinate_count)),
            'retained_fraction_decimal': (coordinate_count-order)/coordinate_count,
            'fixed_width_bits_per_symbol': symbol_bits,
            'retained_payload_bits': (coordinate_count-order)*symbol_bits,
            'repair_payload_bits': order*symbol_bits,
            'interpolation_matrix_bytes_in_this_implementation': int(matrix.nbytes),
            'sampled_full_rank_projections': 128, 'sampled_reconstruction_checks': 64,
            'source_sha256': hashlib.sha256(source.tobytes()).hexdigest()}


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    generator = np.random.default_rng(configuration['seed'])
    summary = {'status': 'complete',
               'exhaustive': [exhaustive_case(*specification) for specification in configuration['exhaustive_cases']],
               'large': [large_case(*specification, generator) for specification in configuration['large_cases']],
               'neural_execution': False, 'neural_training': False,
               'scope': 'single_query_local_functions_of_uniform_independent_stored_world_symbols',
               'general_scc_impossibility_established': False,
               'wall_seconds': time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
