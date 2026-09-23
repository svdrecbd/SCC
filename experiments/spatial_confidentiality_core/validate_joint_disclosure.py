"""Exact static exposure bounds and a sequential disclosure counter-control."""
from fractions import Fraction
from math import comb
from pathlib import Path
import json
import sys
import time


def xor_policy(world, count, secret):
    register = 0
    for coordinate in range(count):
        register ^= (world >> coordinate) & 1
    return secret if register else 0


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    monotone_records = []
    for count in range(1, 5):
        worlds = 1 << count
        edges = [(world, world | (1 << coordinate)) for world in range(worlds)
                 for coordinate in range(count) if not (world >> coordinate) & 1]
        signs = [1 if world.bit_count() % 2 == 0 else -1 for world in range(worlds)]
        accepted = 0
        maximum_numerator = 0
        for function in range(1 << worlds):
            if any(((function >> lower) & 1) > ((function >> upper) & 1) for lower, upper in edges):
                continue
            accepted += 1
            numerator = sum(((function >> world) & 1)*signs[world] for world in range(worlds))
            maximum_numerator = max(maximum_numerator, abs(numerator))
        bound_numerator = comb(count-1, (count-1)//2)
        assert maximum_numerator == bound_numerator
        threshold_maximum = max(abs(sum(signs[world]*int(world.bit_count() >= threshold) for world in range(worlds)))
                                for threshold in range(1, count+1))
        assert threshold_maximum == bound_numerator
        assert accepted == [3, 6, 20, 168][count-1]
        monotone_records.append({'coordinates': count, 'enumerated_functions': 1 << worlds,
                                 'monotone_functions': accepted,
                                 'maximum_parity_coefficient': str(Fraction(maximum_numerator, worlds)),
                                 'maximum_brier_gain': str(Fraction(maximum_numerator, worlds)**2)})
    conjunction_checks = 0
    for count in range(1, 9):
        worlds = 1 << count
        for selected in range(worlds):
            numerator = sum(int((world & selected) == selected)*(1 if world.bit_count() % 2 == 0 else -1)
                            for world in range(worlds))
            expected = (-1)**count if selected == worlds-1 else 0
            assert numerator == expected
            conjunction_checks += 1
    sequential_checks = 0
    monotonicity_failures = 0
    for count in range(1, 11):
        worlds = 1 << count
        for world in range(worlds):
            observed_zero = xor_policy(world, count, 0)
            observed_one = xor_policy(world, count, 1)
            violation = observed_zero != observed_one
            assert violation == world.bit_count() % 2
            sequential_checks += 1
        if count >= 2:
            assert xor_policy(1, count, 1) > xor_policy(3, count, 1)
            monotonicity_failures += 1
    asymptotic_records = []
    for count in configuration['reported_sizes']:
        coefficient = Fraction(comb(count-1, (count-1)//2), 2**count)
        asymptotic_records.append({'coordinates': count,
                                   'monotone_maximum_brier_gain_fraction': str(coefficient**2),
                                   'monotone_maximum_brier_gain_decimal': float(coefficient**2),
                                   'conjunction_maximum_brier_gain_fraction': str(Fraction(1, 4**count)),
                                   'sequential_xor_brier_gain': '1/4',
                                   'sequential_observations': count,
                                   'sequential_register_bits': 1})
    summary = {'status': 'complete', 'monotone_exhaustion': monotone_records,
               'conjunction_functions_checked': conjunction_checks,
               'sequential_worlds_checked': sequential_checks,
               'sequential_monotonicity_counterexamples': monotonicity_failures,
               'dimension_comparisons': asymptotic_records,
               'neural_training': False, 'general_scc_impossibility_established': False,
               'wall_seconds': time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'dimension_comparisons'}, indent=2))
    print(json.dumps([{'coordinates': record['coordinates'], 'monotone_maximum_brier_gain': record['monotone_maximum_brier_gain_decimal']}
                      for record in asymptotic_records]))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
