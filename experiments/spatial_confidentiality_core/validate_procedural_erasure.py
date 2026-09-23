"""Lift local-judgment erasure to reusable linear mappings on fresh inputs."""
from itertools import combinations
from pathlib import Path
import json
import sys
import time
import numpy as np
from validate_bounded_order_erasure import interpolation_matrix, identifiers


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    prime = configuration['prime']
    coordinates = configuration['coordinates']
    order = configuration['maximum_local_coordinates']
    input_dimension = configuration['input_dimension']
    interpolation = interpolation_matrix(prime, coordinates, order)
    source_count = prime**(coordinates*input_dimension)
    source = ((np.arange(source_count)[:, None]
               // (prime**np.arange(coordinates*input_dimension))[None]) % prime).reshape(source_count, coordinates, input_dimension)
    projection = np.concatenate([-interpolation[order:], np.eye(coordinates-order, dtype=np.int64)], axis=1) % prime
    retained = np.einsum('ij,bjk->bik', projection, source) % prime
    retained_identifiers = identifiers(retained.reshape(source_count, -1), prime)
    table_count = 0
    for subset in combinations(range(coordinates), order):
        local_identifiers = identifiers(source[:, subset, :].reshape(source_count, -1), prime)
        joint_identifiers = retained_identifiers*(prime**(order*input_dimension))+local_identifiers
        counts = np.bincount(joint_identifiers, minlength=source_count)
        assert np.all(counts == 1)
        table_count += 1
    restored = np.einsum('ij,bjk->bik', interpolation, source[:, :order]) % prime
    restored[:, order:] = (restored[:, order:]+retained) % prime
    assert np.array_equal(restored, source)
    input_identifiers = np.arange(1, prime**input_dimension)
    inputs = (input_identifiers[:, None] // (prime**np.arange(input_dimension))[None]) % prime
    ordinary_outputs = np.einsum('bij,tj->bti', source, inputs) % prime
    retained_outputs = np.einsum('bij,tj->bti', retained, inputs) % prime
    projected_outputs = np.einsum('ij,btj->bti', projection, ordinary_outputs) % prime
    assert np.array_equal(retained_outputs, projected_outputs)
    for index in range(len(inputs)):
        codes = identifiers(retained_outputs[:, index], prime)
        counts = np.bincount(codes, minlength=prime**(coordinates-order))
        assert np.all(counts == source_count//len(counts))
    basis = {tuple(row) for row in np.eye(input_dimension, dtype=np.int64)}
    summary = {'status': 'complete', 'source_matrices': source_count,
               'conditional_row_block_tables': table_count,
               'conditional_row_block_cells': table_count*source_count,
               'sources_per_joint_cell': 1, 'reconstructed_matrices': source_count,
               'nonzero_inputs': len(inputs),
               'inputs_outside_coordinate_basis': sum(tuple(row) not in basis for row in inputs),
               'retained_procedure_checks': source_count*len(inputs),
               'retained_procedure_accuracy': 1,
               'source_independent_full_output_accuracy': 1/(prime**(coordinates-order)),
               'repair_field_symbols': order*input_dimension,
               'matrix_origin': 'provisioned_exhaustive_control',
               'neural_execution': False, 'neural_training': False,
               'scope': 'public_linear_arithmetic_with_uniform_hidden_matrix_and_no_correlated_repair_data',
               'wall_seconds': time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
