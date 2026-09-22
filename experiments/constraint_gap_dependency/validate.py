"""Check clause-score recovery, near-solutions, repairs and missing gap promises."""
import itertools
import json
import platform
import resource
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path


def clause(block, forbidden):
    return tuple(-(3 * block + bit + 1) if forbidden & (1 << bit)
                 else 3 * block + bit + 1 for bit in range(3))


def block_clauses(block, omitted=None, duplicated=None):
    clauses = [clause(block, forbidden) for forbidden in range(8) if forbidden != omitted]
    if duplicated is not None:
        clauses.append(clause(block, duplicated))
    return clauses


def satisfied_count(clauses, assignment):
    return sum(any(bool(assignment & (1 << (abs(literal) - 1))) == (literal > 0)
                   for literal in constraint) for constraint in clauses)


def reference_count(clauses, assignment):
    satisfied = 0
    for constraint in clauses:
        forbidden = sum(1 << (abs(literal) - 1) for literal in constraint if literal < 0)
        mask = sum(1 << (abs(literal) - 1) for literal in constraint)
        satisfied += assignment & mask != forbidden
    return satisfied


def number(value):
    return {"exact": str(value), "decimal": float(value)}


def inspect_formula(clauses, variable_count):
    scores = []
    for assignment in range(1 << variable_count):
        count = satisfied_count(clauses, assignment)
        assert count == reference_count(clauses, assignment)
        scores.append(count)
    mean = Fraction(sum(scores), len(scores) * len(clauses))
    assert mean == Fraction(7, 8)
    return scores, {'variables': variable_count, 'clauses': len(clauses),
                    'assignments': len(scores), 'score_count_histogram': dict(sorted(Counter(scores).items())),
                    'optimum': number(Fraction(max(scores), len(clauses))),
                    'uniform_assignment_mean': number(mean)}


def main(root):
    start = time.monotonic()
    config = json.loads((root / 'config.json').read_text())
    baseline = Fraction(config['baseline'])
    assert baseline == Fraction(7, 8)
    cases = []
    assignment_checks = 0
    local_cases = 0
    for omitted, duplicated in itertools.permutations(range(8), 2):
        clauses = block_clauses(0, omitted, duplicated)
        scores, record = inspect_formula(clauses, 3)
        assert max(scores) == 8 and scores[omitted] == 8
        for count in scores:
            quality = Fraction(count, 8)
            assert quality <= baseline + (1 - baseline) * (quality > baseline)
        assignment_checks += len(scores)
        local_cases += 1
    for blocks in config['block_counts']:
        for family in ('satisfiable', 'gap_negative', 'padded_negative'):
            clauses = []
            for block in range(blocks):
                constrained = family == 'gap_negative' or (family == 'padded_negative' and block == 0)
                clauses.extend(block_clauses(block) if constrained else block_clauses(block, 0, 1))
            scores, record = inspect_formula(clauses, 3 * blocks)
            assignment_checks += len(scores)
            record.update({'family': family, 'blocks': blocks})
            record['above_gap_count'] = sum(Fraction(count, len(clauses)) > baseline for count in scores)
            if family == 'satisfiable':
                assert max(scores) == len(clauses) and scores[0] == len(clauses)
                near_solution_counts = [scores[1 << bit] for bit in range(3 * blocks)]
                assert all(count < len(clauses) for count in near_solution_counts)
                repaired = 0
                for bit in range(3 * blocks):
                    damaged = 1 << bit
                    candidates = [damaged] + [damaged ^ (1 << position) for position in range(3 * blocks)]
                    repaired += max(scores[candidate] for candidate in candidates) == len(clauses)
                assert repaired == 3 * blocks
                record['one_bit_damaged_scores'] = [number(Fraction(count, len(clauses))) for count in near_solution_counts]
                record['damaged_outputs_above_gap'] = sum(Fraction(count, len(clauses)) > baseline for count in near_solution_counts)
                record['radius_one_recovered_count'] = repaired
            elif family == 'gap_negative':
                assert set(scores) == {7 * blocks}
                assert record['above_gap_count'] == 0
            else:
                assert max(scores) == len(clauses) - 1
                assert (record['above_gap_count'] > 0) == (blocks > 1)
                record['gap_promise_rejected'] = blocks > 1
            cases.append(record)
    advantage = Fraction(config['reader_advantage'])
    excess = Fraction(config['gap_excess'])
    soundness = baseline + excess
    conditional_score_cap = soundness + 2 * advantage * (1 - soundness)
    sensitivity = []
    for specification in config['coverage_constants']:
        coverage = Fraction(specification)
        score_cap = 1 - (1 - soundness) * (1 - 2 * advantage) / coverage
        sensitivity.append({'coverage_constant': number(coverage), 'score_cap': number(score_cap),
                            'fraction_of_perfect_over_baseline_advantage_lost': number((1 - score_cap) / (1 - baseline))})
    assert conditional_score_cap == Fraction(10981, 12500)
    assert (1 - conditional_score_cap) / (1 - baseline) == Fraction(3038, 3125)
    result = {'local_positive_formulas': local_cases, 'assignment_checks_with_independent_evaluator': assignment_checks,
              'families': cases, 'conditional_score_cap': number(conditional_score_cap),
              'coverage_sensitivity': sensitivity,
              'maximum_coverage_for_eighty_percent_loss': number((1 - conditional_score_cap) / ((1 - baseline) * Fraction(4, 5))),
              'elapsed_seconds': time.monotonic() - start,
              'peak_memory_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'scope': 'Tiny explicit clause families; all are cheaply decidable. No PCP reduction implementation, average-case hardness, protected-function erasure or learned cognition is demonstrated.'}
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'families'}, indent=2))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    (root / 'machine.json').write_text(json.dumps({'platform': platform.platform(), 'python': sys.version,
        'hostname': platform.node()}, indent=2) + '\n')
    main(root)
