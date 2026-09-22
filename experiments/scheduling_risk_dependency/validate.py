"""Exact checks of hazard-dependent scheduling and a surviving generic scheduler."""
import itertools
import json
import math
import platform
import resource
import sys
import time
from fractions import Fraction
from pathlib import Path


def schedule_value(jobs, order, survival):
    elapsed = 0
    result = Fraction(0)
    for index in order:
        duration, reward = jobs[index]
        elapsed += duration
        result += reward * survival ** elapsed
    return result


def schedule_order(jobs, survival):
    priorities = [reward * survival ** duration / (1 - survival ** duration)
                  for duration, reward in jobs]
    return tuple(sorted(range(len(jobs)), key=lambda index: (-priorities[index], index)))


def number(value):
    return {"exact": str(value), "decimal": float(value)}


def main(root):
    start = time.monotonic()
    config = json.loads((root / 'config.json').read_text())
    survival_values = tuple(map(Fraction, config['survival_probabilities']))
    assert survival_values == (Fraction(1, 4), Fraction(3, 4))
    assert config['job_count'] == 3
    job_types = list(itertools.product(config['durations'], config['rewards']))
    permutations = tuple(itertools.permutations(range(config['job_count'])))
    replacement_survival = Fraction(config['replacement_survival_probability'])
    intact_total = Fraction(0)
    replacement_total = Fraction(0)
    baseline_total = Fraction(0)
    instance_count = 0
    exhaustive_values = 0
    priority_checks = 0
    optimal_in_both_worlds = 0
    for jobs in itertools.product(job_types, repeat=config['job_count']):
        world_values = []
        optimal_orders = []
        for survival in survival_values:
            values = {order: schedule_value(jobs, order, survival) for order in permutations}
            optimum = max(values.values())
            assert values[schedule_order(jobs, survival)] == optimum
            world_values.append(values)
            optimal_orders.append({order for order, value in values.items() if value == optimum})
            intact_total += optimum / 2
            priority_checks += 1
            exhaustive_values += len(values)
        common_order = schedule_order(jobs, replacement_survival)
        replacement_total += sum(values[common_order] for values in world_values) / 2
        baseline_total += max(sum(values[order] for values in world_values) / 2
                              for order in permutations)
        optimal_in_both_worlds += bool(optimal_orders[0] & optimal_orders[1])
        instance_count += 1
    jobs = config['reference_jobs']
    reference_orders = ((0, 1), (1, 0))
    reference_values = [[schedule_value(jobs, order, survival) for order in reference_orders]
                        for survival in survival_values]
    assert reference_values == [[Fraction(19, 64), Fraction(13, 64)],
                                [Fraction(129, 64), Fraction(135, 64)]]
    reference_intact = sum(max(values) for values in reference_values) / 2
    reference_baseline = max(sum(values[index] for values in reference_values) / 2
                             for index in range(2))
    assert reference_intact == Fraction(77, 64)
    assert reference_baseline == Fraction(74, 64)
    # Returning the optimal first job discloses which side of hazard 1/2 applies.
    for survival in survival_values:
        order = schedule_order(jobs, survival)
        assert int(order[0] == 0) == int(1 - survival >= Fraction(1, 2))
    repair = []
    for count in config['observation_counts']:
        assert count > 0 and count % 2 == 1
        error = sum(Fraction(math.comb(count, failures) * 3 ** (count - failures), 4 ** count)
                    for failures in range(count // 2 + 1, count + 1))
        repair.append({'independent_observations': count, 'judgment_error': number(error),
                       'reference_regret': number(Fraction(3, 32) * error)})
    result = {
        'instances': instance_count, 'permutation_values': exhaustive_values,
        'priority_optimality_checks': priority_checks,
        'instances_with_a_common_optimal_order': optimal_in_both_worlds,
        'mean_intact_value': number(intact_total / instance_count),
        'mean_optimal_no_state_value': number(baseline_total / instance_count),
        'mean_fixed_parameter_value': number(replacement_total / instance_count),
        'fixed_parameter_fraction_of_intact_value': number(replacement_total / intact_total),
        'no_state_fraction_of_intact_value': number(baseline_total / intact_total),
        'reference_values_low_then_high_survival': [[number(value) for value in values] for values in reference_values],
        'reference_intact_value': number(reference_intact),
        'reference_no_state_value': number(reference_baseline),
        'reference_fraction_of_intact_value_retained': number(reference_baseline / reference_intact),
        'repair_from_new_evidence': repair,
        'elapsed_seconds': time.monotonic() - start,
        'peak_memory_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope': 'Exact finite validation of a derived scheduling rule and conditional coupling; the generic scheduler remains intact. Observation repairs are analytic, not executed environmental trials.'
    }
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    (root / 'machine.json').write_text(json.dumps({'platform': platform.platform(),
        'python': sys.version, 'hostname': platform.node()}, indent=2) + '\n')
    main(root)
