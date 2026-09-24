"""Audit exact complete-state guessing bounds and uniform privacy certificates."""
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import math
import sys
import time


def encode(source, protected_bit, truth_table):
    return source, protected_bit & ((truth_table >> source) & 1)


def audit_table(bit_count, truth_table, enumerate_readers, caps):
    source_count = 1 << bit_count
    joint = defaultdict(lambda: [0, 0])
    satisfying = []
    for source in range(source_count):
        zero = encode(source, 0, truth_table)
        one = encode(source, 1, truth_table)
        predicate = (truth_table >> source) & 1
        assert (zero != one) == bool(predicate)
        if predicate:
            satisfying.append(source)
        for protected_bit in (0, 1):
            joint[encode(source, protected_bit, truth_table)][protected_bit] += 1
    total = 2 * source_count
    population = sum((Fraction(max(counts), total) for counts in joint.values()), Fraction())
    worst = max(Fraction(max(counts), sum(counts)) for counts in joint.values())
    prevalence = Fraction(len(satisfying), source_count)
    assert population == (1 + prevalence) / 2
    assert worst == (1 if satisfying else Fraction(1, 2))
    for cap in caps:
        assert (worst <= cap) == (not satisfying)
    reader_count = 0
    if enumerate_readers:
        best = Fraction()
        states = list(joint)
        for decisions in product((0, 1), repeat=len(states)):
            correct = sum(joint[state][decision] for state, decision in zip(states, decisions))
            best = max(best, Fraction(correct, total))
            reader_count += 1
        assert best == population
    return {'source_bits': bit_count, 'truth_table': truth_table,
            'satisfying_assignments': len(satisfying), 'counterexample': satisfying[0] if satisfying else None,
            'population_bayes_accuracy': str(population), 'maximum_state_posterior_accuracy': str(worst),
            'reachable_states': len(joint), 'uniform_certificate': not satisfying,
            'readers_enumerated': reader_count}


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / 'config.json').read_text())
    caps = [Fraction(value) for value in configuration['uniform_accuracy_caps']]
    assert all(Fraction(1, 2) <= cap < 1 for cap in caps)
    records = []
    for bit_count in configuration['source_bit_counts']:
        for truth_table in range(1 << (1 << bit_count)):
            records.append(audit_table(bit_count, truth_table,
                                      bit_count <= configuration['maximum_reader_enumeration_bits'], caps))
    bit_count = configuration['rare_event_source_bits']
    prevalence = Fraction(1, 1 << bit_count)
    sample_count = configuration['analytical_sample_count']
    # Exact expressions and a stable floating evaluation are reported separately.
    log_miss = sample_count * math.log1p(-float(prevalence))
    miss_probability = math.exp(log_miss)
    rare = {'source_bits': bit_count, 'predicate': 'conjunction_of_all_source_bits',
            'revealing_endpoint_probability': str(prevalence),
            'population_bayes_accuracy': str((1 + prevalence) / 2),
            'population_bayes_advantage': str(prevalence / 2),
            'maximum_state_posterior_accuracy': '1',
            'analytical_sample_count': sample_count,
            'exact_miss_probability_base': str(1 - prevalence),
            'exact_miss_probability_exponent': sample_count,
            'evaluated_miss_probability': miss_probability,
            'simulated_samples': 0}
    assert rare['population_bayes_accuracy'] == '4294967297/8589934592'
    assert Fraction(1, 2) + prevalence / 2 < Fraction(107, 200)
    assert miss_probability > 0.99999
    summary = {'status': 'complete', 'truth_tables': len(records),
               'deterministic_readers_enumerated': sum(record['readers_enumerated'] for record in records),
               'uniform_cap_checks': len(records) * len(caps),
               'structural_source_checks': sum(1 << record['source_bits'] for record in records),
               'uniformly_private_encoders': sum(record['uniform_certificate'] for record in records),
               'rare_event': rare, 'source_information_certificate_only': True,
               'neural_removal_demonstrated': False, 'neural_training': False,
               'wall_seconds': time.perf_counter() - started}
    payload = json.dumps({'summary': summary, 'records': records}, indent=2) + '\n'
    assert len(payload.encode()) <= configuration['maximum_result_bytes']
    (directory / 'results.json').write_text(payload)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
