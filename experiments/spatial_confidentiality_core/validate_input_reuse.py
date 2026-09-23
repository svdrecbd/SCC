"""Verify recovery from repeated, related observations of one source."""
from collections import defaultdict
from fractions import Fraction
from itertools import permutations
from pathlib import Path
import json
import sys
import time
from validate_adaptive_disclosure import enumerate_subspaces, parity


def transform(row, permutation):
    return sum(((row >> index) & 1) << position for index, position in enumerate(permutation))


def span(rows):
    values = {0}
    for row in rows:
        values |= {value ^ row for value in tuple(values)}
    return values


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / 'config.json').read_text())
    dimension = configuration['dimension']
    assert dimension == 4
    reflection = (1, 0, 3, 2)
    groups = {'identity': [tuple(range(4))],
              'reflection': [tuple(range(4)), reflection],
              'permutations': list(permutations(range(4)))}
    records = []
    checks = 0
    for rank in range(dimension + 1):
        for subspace in enumerate_subspaces(dimension, rank):
            basis = subspace['basis']
            for name, group in groups.items():
                rows = [transform(row, permutation) for permutation in group for row in basis]
                row_space = span(rows)
                fibres = defaultdict(list)
                for source in range(1 << dimension):
                    observation = tuple(parity(row & source) for row in rows)
                    fibres[observation].append(source)
                kernel = [source for source in range(1 << dimension)
                          if all(parity(row & source) == 0 for row in rows)]
                assert len(fibres) == len(row_space)
                assert len(kernel) * len(row_space) == 1 << dimension
                for query in range(1 << dimension):
                    fibre_constant = all(len({parity(query & source) for source in fibre}) == 1
                                         for fibre in fibres.values())
                    kernel_annihilated = all(parity(query & source) == 0 for source in kernel)
                    assert fibre_constant == kernel_annihilated == (query in row_space)
                    checks += 1
                # Each group-closed kernel must be invariant under every allowed action.
                assert all(transform(source, permutation) in kernel
                           for source in kernel for permutation in group)
                records.append({'basis': basis, 'group': name,
                                'observable_dimension': (len(row_space)).bit_length()-1,
                                'unobservable_sources': kernel})
    original_rows = [1, 4]
    reflected_rows = [transform(row, reflection) for row in original_rows]
    assert len(span(original_rows)) == 4
    assert len(span(original_rows + reflected_rows)) == 16
    invariant_rows = [3, 12]
    assert len(span(invariant_rows + [transform(row, reflection) for row in invariant_rows])) == 4
    assert 1 not in span(invariant_rows)
    # For real half-field selection, the reconstruction selects one observation
    # for each coordinate. Squared error equals the sum of selected error squares.
    real_checks = 0
    sources = [(Fraction(a), Fraction(b), Fraction(c), Fraction(d))
               for a in range(2) for b in range(2) for c in range(2) for d in range(2)]
    for source in sources:
        errors = [Fraction(1, 7), Fraction(-2, 9), Fraction(3, 11), Fraction(-4, 13)]
        original = [source[0]+errors[0], source[2]+errors[1]]
        reflected = [source[1]+errors[2], source[3]+errors[3]]
        recovered = [original[0], reflected[0], original[1], reflected[1]]
        assert sum((value-truth)**2 for value, truth in zip(recovered, source)) == sum(e*e for e in errors)
        assert recovered != list(source)  # A zero-error equivariance claim must fail.
        real_checks += 1
    (directory / 'cases.json').write_text(json.dumps(records, indent=2)+'\n')
    summary = {'status': 'complete', 'subspaces': len(records)//len(groups),
               'observation_families': len(records), 'target_recovery_checks': checks,
               'real_error_checks': real_checks,
               'coordinate_half_rank_before_after': [2, 4],
               'reflection_invariant_rank_before_after': [2, 2],
               'neural_training': False, 'wall_seconds': time.perf_counter()-started}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
