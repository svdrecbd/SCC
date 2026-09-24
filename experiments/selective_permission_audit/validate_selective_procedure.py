"""Check complete cyclic procedures under exact and approximate phase removal."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
import random
import sys
import time


def response_profiles(count, group_size):
    if count < 2 or count % 2 or group_size < 1 or count % group_size or count // group_size < 2:
        raise ValueError('require an even source count and at least two equal groups')
    return [[((world + query) % count) // group_size for query in range(count)] for world in range(count)]


def partitions(count):
    assignment = [0] * count

    def extend(position, maximum):
        if position == count:
            yield tuple(assignment)
            return
        for state in range(maximum + 2):
            assignment[position] = state
            yield from extend(position + 1, max(maximum, state))

    yield from extend(1, 0)


def measure_channel(masses, profiles):
    """Rows are source worlds; equal row sums give the uniform source law."""
    count = len(profiles)
    row_mass = sum(masses[0])
    assert row_mass > 0 and all(sum(row) == row_mass for row in masses)
    total_mass = count * row_mass
    protected_numerator = 0
    useful_numerator = 0
    for state in range(len(masses[0])):
        phases = [0, 0]
        for world in range(count):
            phases[world % 2] += masses[world][state]
        protected_numerator += max(phases)
        for query in range(count):
            labels = defaultdict(int)
            for world in range(count):
                labels[profiles[world][query]] += masses[world][state]
            useful_numerator += max(labels.values())
    return Fraction(protected_numerator, total_mass), Fraction(useful_numerator, count * total_mass)


def validate_distances(profiles, group_size):
    count = len(profiles)
    assert len({tuple(row) for row in profiles}) == count
    minimum = count
    for left in range(count):
        for right in range(count):
            observed = sum(a != b for a, b in zip(profiles[left], profiles[right]))
            separation = min((left - right) % count, (right - left) % count)
            expected = Fraction(min(separation, group_size), group_size)
            assert Fraction(observed, count) == expected
            if left % 2 != right % 2:
                minimum = min(minimum, observed)
    assert Fraction(minimum, count) == Fraction(1, group_size)
    return count * count


def frontier_channel(count, cap):
    revelation = 2 * cap - 1
    denominator = revelation.denominator
    numerator = revelation.numerator
    masses = [[0] * (count // 2 + count) for _ in range(count)]
    for world in range(count):
        masses[world][world // 2] = denominator - numerator
        masses[world][count // 2 + world] = numerator
    return masses


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / 'config.json').read_text())
    count = configuration['partition_source_count']
    assignments = list(partitions(count))
    assert count == 8 and len(assignments) == 4140
    generator = random.Random(configuration['seed'])
    partition_checks = 0
    stochastic_checks = 0
    distance_checks = 0
    summaries = []
    for group_size in configuration['partition_group_sizes']:
        profiles = response_profiles(count, group_size)
        distance_checks += validate_distances(profiles, group_size)
        private_maximum = Fraction()
        for assignment in assignments:
            masses = [[int(state == assignment[world]) for state in range(max(assignment) + 1)] for world in range(count)]
            protected, useful = measure_channel(masses, profiles)
            assert useful <= 1 - (1 - protected) / group_size
            if protected == Fraction(1, 2):
                private_maximum = max(private_maximum, useful)
            partition_checks += 1
        assert private_maximum == 1 - Fraction(1, 2 * group_size)
        for _ in range(configuration['stochastic_channels_per_group_size']):
            masses = [[0] * 5 for _ in range(count)]
            for row in masses:
                for _ in range(17):
                    row[generator.randrange(5)] += 1
            protected, useful = measure_channel(masses, profiles)
            assert useful <= 1 - (1 - protected) / group_size
            stochastic_checks += 1
        summaries.append({'source_count': count, 'group_size': group_size, 'maximum_private_useful_accuracy': str(private_maximum)})
    count = configuration['large_source_count']
    group_size = configuration['large_group_size']
    profiles = response_profiles(count, group_size)
    distance_checks += validate_distances(profiles, group_size)
    # The same stored pair generates a complete reusable procedure in both worlds.
    for world in range(count):
        partner = world ^ 1
        assert world // 2 == partner // 2 and world % 2 != partner % 2
        success = sum(Fraction(1, 2) + Fraction(profiles[partner][query] == profiles[world][query], 2) for query in range(count)) / count
        assert success == 1 - Fraction(1, 2 * group_size)
        for query in range(count):
            assert ((world + query) % count) % 2 != ((partner + query) % count) % 2
    frontier = []
    for cap_text in configuration['accuracy_caps']:
        cap = Fraction(cap_text)
        protected, useful = measure_channel(frontier_channel(count, cap), profiles)
        assert protected == cap
        assert useful == 1 - (1 - cap) / group_size
        frontier.append({'protected_accuracy': str(protected), 'useful_accuracy': str(useful)})
    for dimensions in [(1, 1), (3, 1), (8, 0), (8, 3), (8, 8)]:
        try:
            response_profiles(*dimensions)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid dimensions accepted')
    result = {'status': 'complete', 'partition_checks': partition_checks, 'stochastic_channel_checks': stochastic_checks,
              'ordered_profile_distance_checks': distance_checks, 'private_maxima': summaries, 'frontier': frontier,
              'large_source_count': count, 'group_size': group_size, 'complete_protected_query_family': True,
              'source_posterior_certificate': True, 'neural_removal_demonstrated': False, 'neural_training': False,
              'wall_seconds': time.perf_counter() - started}
    payload = json.dumps(result, indent=2) + '\n'
    assert len(payload.encode()) <= configuration['maximum_result_bytes']
    (directory / 'results.json').write_text(payload)
    print(payload)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
