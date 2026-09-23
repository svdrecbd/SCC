"""Exact coverage audit of local privacy judgments on a stored depth field."""
from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np


def coordinate_sets(dimension):
    count = 2**dimension-1
    pivots = [2**index for index in range(dimension)]
    triples = []
    for index in range(1, count+1):
        highest = 1 << (index.bit_length()-1)
        if index != highest:
            triples.append((index, highest, index-highest))
    return count, pivots, triples


def encode(source, triples):
    return np.stack([source[:, index-1] ^ source[:, highest-1] ^ source[:, remainder-1]
                     for index, highest, remainder in triples], axis=1)


def reconstruct(retained, pivot_values, dimension):
    count, pivots, triples = coordinate_sets(dimension)
    result = np.zeros((len(retained), count), dtype=np.uint8)
    result[:, np.array(pivots)-1] = pivot_values
    for column, (index, highest, remainder) in enumerate(triples):
        result[:, index-1] = retained[:, column] ^ result[:, highest-1] ^ result[:, remainder-1]
    return result


def exhaustive_validation(dimension):
    count, pivots, triples = coordinate_sets(dimension)
    identifiers = np.arange(2**count, dtype=np.uint64)
    source = ((identifiers[:, None] >> np.arange(count, dtype=np.uint64)) & 1).astype(np.uint8)
    retained = encode(source, triples)
    retained_identifiers = (retained.astype(np.uint64) << np.arange(len(triples), dtype=np.uint64)).sum(axis=1).astype(np.int64)
    groups = 2**len(triples)
    fibres = np.bincount(retained_identifiers, minlength=groups)
    assert np.array_equal(fibres, np.full(groups, 2**dimension))
    singleton_tables = 0
    pair_tables = 0
    order_tables = 0
    for coordinate in range(count):
        sums = np.bincount(retained_identifiers, weights=source[:, coordinate], minlength=groups)
        assert np.array_equal(sums*2, fibres)
        singleton_tables += groups
    depths = 2*count+np.arange(1, count+1)[None, :]+4*count*source.astype(np.int64)
    for first, second in combinations(range(count), 2):
        pair_code = source[:, first].astype(np.int64)*2+source[:, second]
        for value in range(4):
            sums = np.bincount(retained_identifiers, weights=(pair_code == value), minlength=groups)
            assert np.array_equal(sums*4, fibres)
            pair_tables += groups
        labels = depths[:, first] < depths[:, second]
        sums = np.bincount(retained_identifiers, weights=labels, minlength=groups)
        assert np.array_equal(sums*4, fibres*3)
        reverse = np.bincount(retained_identifiers, weights=(depths[:, second] < depths[:, first]), minlength=groups)
        assert np.array_equal(reverse*4, fibres)
        order_tables += 2*groups
    recovered = reconstruct(retained, source[:, np.array(pivots)-1], dimension)
    assert np.array_equal(recovered, source)
    zero_pivot = reconstruct(retained, np.zeros((len(source), dimension), dtype=np.uint8), dimension)
    assert np.array_equal(encode(zero_pivot, triples), retained)
    failed_zero_pivot_sources = int(np.any(source != zero_pivot, axis=1).sum())
    assert failed_zero_pivot_sources == len(source)-groups
    joint_tables = 0
    for column, (index, highest, remainder) in enumerate(triples):
        joint = source[:, index-1] & source[:, highest-1] & source[:, remainder-1]
        sums = np.bincount(retained_identifiers, weights=joint, minlength=groups)
        group_parity = (np.arange(groups) >> column) & 1
        assert np.array_equal(sums*4, fibres*group_parity)
        posterior_counts = Counter(int(value) for value in sums)
        baseline_probability = Fraction(int(joint.sum()), len(joint))
        conditional_loss = sum(Fraction(number, groups)*Fraction(value, 2**dimension)*(1-Fraction(value, 2**dimension))
                               for value, number in posterior_counts.items())
        baseline_loss = baseline_probability*(1-baseline_probability)
        assert baseline_probability == Fraction(1, 8)
        assert baseline_loss == Fraction(7, 64)
        assert conditional_loss == Fraction(3, 32)
        assert baseline_loss-conditional_loss == Fraction(1, 64)
        assert np.array_equal(retained[:, column], source[:, index-1] ^ source[:, highest-1] ^ source[:, remainder-1])
        joint_tables += groups
    return {'dimension': dimension, 'source_bits': count, 'source_states': len(source),
            'retained_bits': len(triples), 'retained_entropy_fraction': str(Fraction(len(triples), count)),
            'retained_states': groups, 'fibre_size': 2**dimension,
            'conditional_singleton_tables': singleton_tables, 'conditional_pair_cells': pair_tables,
            'conditional_order_tables': order_tables, 'conditional_joint_tables': joint_tables,
            'all_source_reconstructions_passed': len(source),
            'zero_pivot_reconstruction_failures': failed_zero_pivot_sources,
            'all_local_judgment_information_gains': '0',
            'relative_order_public_accuracy': '3/4', 'relative_order_retained_accuracy': '3/4',
            'retained_relation_accuracy': '1', 'relation_public_accuracy': '1/2',
            'joint_baseline_brier': '7/64', 'joint_retained_brier': '3/32', 'joint_brier_gain': '1/64'}


def large_validation(dimension, generator):
    count, pivots, triples = coordinate_sets(dimension)
    masks = [(1 << (index-1)) ^ (1 << (highest-1)) ^ (1 << (remainder-1))
             for index, highest, remainder in triples]
    leading_coordinates = [value.bit_length() for value in masks]
    assert leading_coordinates == [triple[0] for triple in triples]
    assert len(set(leading_coordinates)) == count-dimension
    kernel_columns = [tuple((index >> bit) & 1 for bit in range(dimension)) for index in range(1, count+1)]
    assert len(set(kernel_columns)) == count and all(any(column) for column in kernel_columns)
    source = generator.integers(0, 2, (64, count), dtype=np.uint8)
    retained = encode(source, triples)
    recovered = reconstruct(retained, source[:, np.array(pivots)-1], dimension)
    assert np.array_equal(recovered, source)
    assert np.array_equal(encode(reconstruct(retained, np.zeros((64, dimension), dtype=np.uint8), dimension), triples), retained)
    return {'dimension': dimension, 'source_bits': count, 'retained_bits': len(triples),
            'retained_entropy_fraction': str(Fraction(len(triples), count)),
            'retained_fraction_decimal': len(triples)/count, 'repair_bits': dimension,
            'nonzero_distinct_kernel_columns': len(kernel_columns),
            'triangular_independent_rows': len(masks), 'sampled_reconstruction_checks': len(source),
            'sampled_source_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
            'bit_operations_per_encoded_relation': 2}


def sharing_validation():
    records = []
    for visibility in product([0, 1], repeat=3):
        laws = []
        for secret in [0, 1]:
            observations = Counter()
            for first_mask, second_mask in product([0, 1], repeat=2):
                shares = (first_mask, second_mask, secret ^ first_mask ^ second_mask)
                observation = tuple(value if visible else -1 for value, visible in zip(shares, visibility))
                observations[observation] += 1
            laws.append(observations)
        transcript_set = set(laws[0]) | set(laws[1])
        distance = sum(Fraction(abs(laws[0][transcript]-laws[1][transcript]), 8) for transcript in transcript_set)
        assert distance == int(all(visibility))
        records.append({'visible_rays': list(visibility), 'total_variation': str(distance),
                        'violation': distance != 0,
                        'observation_laws': [{str(key): value for key, value in sorted(law.items())} for law in laws]})
    return records


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    generator = np.random.default_rng(configuration['seed'])
    summary = {'status': 'complete',
               'exhaustive': [exhaustive_validation(value) for value in configuration['exhaustive_dimensions']],
               'algebraic_and_sampled': [large_validation(value, generator) for value in configuration['large_dimensions']],
               'sharing_laws': sharing_validation(),
               'reader_scope': 'retained_state_and_independent_public_query_only',
               'neural_model_attack': False, 'neural_training': False,
               'general_scc_impossibility_established': False,
               'wall_seconds': time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    size = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
    assert size < configuration['maximum_output_bytes']
    print(json.dumps({key: value for key, value in summary.items() if key != 'sharing_laws'}, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
