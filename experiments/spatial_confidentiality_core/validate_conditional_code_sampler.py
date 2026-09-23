"""Check exact conditional sampling and adaptive short-history query limits."""
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import json
import math
import platform
import sys
import time
import numpy as np
from finite_field_recovery import BinaryExtensionField
from conditional_code_sampler import ConditionalCodeSampler, draw_noise


def vectors(order, dimension):
    return (np.arange(order**dimension)[:, None] // order**np.arange(dimension)[None, :]) % order


def identifier(values, order):
    return np.asarray(values) @ (order**np.arange(np.asarray(values).shape[-1]))


def entropy(probabilities):
    return -sum(float(value)*math.log2(float(value)) for value in probabilities if value)


def finite_validation(configuration):
    field = BinaryExtensionField(configuration['field_degree'], configuration['modulus'])
    count, dimension, order = configuration['coordinates'], configuration['dimension'], field.order
    assert configuration['correlation'] == [1, 2]
    matrix = field.interpolation(count, dimension)
    states = vectors(order, count)
    syndromes = states[:, dimension:] ^ np.stack([field.linear_image(matrix[dimension:], state[:dimension]) for state in states])
    encoded = identifier(syndromes, order)
    retained_count = order**(count-dimension)
    posterior = np.zeros((retained_count, len(states)), dtype=np.int64)
    for state, retained in zip(states, encoded):
        posterior[retained] += (order+1)**np.count_nonzero(states == state, axis=1)
    denominator = order**dimension*(2*order)**count
    assert np.all(posterior.sum(axis=1) == denominator)
    assert np.all(posterior.sum(axis=0) == retained_count*denominator//len(states))
    coordinate_checks = block_checks = 0
    for history_size in range(dimension):
        for observed in combinations(range(count), history_size):
            for values in product(range(order), repeat=history_size):
                selected = np.all(states[:, observed] == values, axis=1)
                mass = posterior[:, selected].sum(axis=1)
                assert np.all(mass == denominator//order**history_size)
                for block_size in range(1, dimension-history_size+1):
                    for targets in combinations([position for position in range(count) if position not in observed], block_size):
                        for target_values in product(range(order), repeat=block_size):
                            target = selected & np.all(states[:, targets] == target_values, axis=1)
                            assert np.array_equal(posterior[:, target].sum(axis=1)*order**block_size, mass)
                            block_checks += retained_count
                            coordinate_checks += retained_count if block_size == 1 else 0
    adaptive_checks = 0
    for retained, syndrome in enumerate(vectors(order, count-dimension)):
        first = int(syndrome[0])
        remaining = [position for position in range(count) if position != first]
        for value in range(order):
            second = remaining[(int(syndrome[1])+value) % len(remaining)]
            selected = states[:, first] == value
            for answer in range(order):
                assert int(posterior[retained, selected & (states[:, second] == answer)].sum())*order**2 == denominator
                adaptive_checks += 1
    # Compare independent enumeration of the conditional sampler with the exact
    # posterior obtained by summing every original latent source and noise law.
    contexts = sampling_paths = overwrite_failures = 0
    for retained, syndrome in enumerate(vectors(order, count-dimension)):
        for observed in ([], [0], [2], [0, 2]):
            sampler = ConditionalCodeSampler(field, count, dimension, syndrome, observed)
            remaining_noise = vectors(order, count-len(observed))
            remaining_weights = (order+1)**np.count_nonzero(remaining_noise == 0, axis=1)
            for values in vectors(order, len(observed)):
                tally = np.zeros(len(states), dtype=np.int64)
                for observed_noise in vectors(order, len(observed)):
                    weight = (order+1)**int(np.count_nonzero(observed_noise == 0))
                    for free_values in vectors(order, dimension-len(observed)):
                        source, sample = sampler.construct(values, observed_noise, free_values, np.zeros(len(sampler.unobserved), dtype=np.int64))
                        assert np.array_equal(source[dimension:] ^ field.linear_image(matrix[dimension:], source[:dimension]), syndrome)
                        samples = np.broadcast_to(sample, (len(remaining_noise), count)).copy()
                        samples[:, sampler.unobserved] ^= remaining_noise
                        sample_ids = identifier(samples, order)
                        np.add.at(tally, sample_ids, weight*remaining_weights)
                        sampling_paths += len(remaining_noise)
                selected = np.all(states[:, observed] == values, axis=1)
                expected = posterior[retained]*selected
                assert np.array_equal(tally, expected)
                assert int(tally.sum()) == denominator//order**len(observed)
                if observed:
                    overwritten = states.copy()
                    overwritten[:, observed] = values
                    overwrite_tally = np.zeros(len(states), dtype=np.int64)
                    np.add.at(overwrite_tally, identifier(overwritten, order), posterior[retained])
                    assert not np.array_equal(overwrite_tally, expected*order**len(observed))
                    overwrite_failures += 1
                contexts += 1
    # At exactly dimension observations the next-coordinate guarantee stops.
    gain = Fraction(0)
    for retained in range(retained_count):
        for values in product(range(order), repeat=dimension):
            selected = np.all(states[:, :dimension] == values, axis=1)
            mass = int(posterior[retained, selected].sum())
            positive = int(posterior[retained, selected & ((states[:, dimension] & 1) == 1)].sum())
            probability = Fraction(positive, mass)
            gain += Fraction(mass, retained_count*denominator)*(probability-Fraction(1, 2))**2
    assert gain > 0
    information = math.log2(len(states))-sum(entropy(row/denominator) for row in posterior)/retained_count
    conditional_information = []
    for observed in ([0], [0, 2]):
        conditional_entropy = 0.0
        for retained in range(retained_count):
            for values in vectors(order, len(observed)):
                selected = np.all(states[:, observed] == values, axis=1)
                mass = int(posterior[retained, selected].sum())
                conditional_entropy += mass/(retained_count*denominator)*entropy(posterior[retained, selected]/mass)
        result = (count-len(observed))*math.log2(order)-conditional_entropy
        assert abs(result-information) < 1e-12
        conditional_information.append(result)
    return {'coordinate_checks': coordinate_checks, 'block_checks': block_checks,
            'adaptive_branch_checks': adaptive_checks, 'conditional_sampler_contexts': contexts,
            'enumerated_sampling_paths': sampling_paths, 'incorrect_overwrite_controls_rejected': overwrite_failures,
            'retained_predictive_information_bits': information,
            'conditional_predictive_information_bits': conditional_information,
            'next_bit_brier_gain_at_dimension_observations': str(gain)}


def large_validation(configuration, seed):
    field = BinaryExtensionField(configuration['field_degree'], configuration['modulus'])
    count, dimension, observed_count = configuration['coordinates'], configuration['dimension'], configuration['observations']
    correlation = float(Fraction(*configuration['correlation']))
    generator = np.random.default_rng(seed)
    matrix = field.interpolation(count, dimension)
    construction_times, sampling_times = [], []
    maximum_array_bytes = 0
    for trial in range(configuration['trials']):
        source = generator.integers(field.order, size=count, dtype=np.int64)
        syndrome = source[dimension:] ^ field.linear_image(matrix[dimension:], source[:dimension])
        actual = source ^ draw_noise(generator, field.order, count, correlation)
        # Every new index depends only on the retained state and prior reveals.
        observed, available = [], list(range(count))
        cursor = int(syndrome[0])
        for step in range(observed_count):
            coordinate = available.pop(cursor % len(available))
            observed.append(coordinate)
            cursor = cursor+int(actual[coordinate])+int(syndrome[(step+1) % len(syndrome)])
        started = time.perf_counter()
        sampler = ConditionalCodeSampler(field, count, dimension, syndrome, observed)
        construction_times.append(time.perf_counter()-started)
        started = time.perf_counter()
        sampled_source, sampled_observation = sampler.sample(actual[observed], correlation, generator)
        sampling_times.append(time.perf_counter()-started)
        assert np.array_equal(sampled_source[dimension:] ^ field.linear_image(matrix[dimension:], sampled_source[:dimension]), syndrome)
        assert np.array_equal(sampled_observation[observed], actual[observed])
        maximum_array_bytes = max(maximum_array_bytes, sampler.matrix.nbytes+sampler.offset.nbytes+sampler.anchors.nbytes+sampler.observed.nbytes+sampler.unobserved.nbytes)
    noise = [correlation+(1-correlation)/field.order]+[(1-correlation)/field.order]*(field.order-1)
    intact_information = count*(math.log2(field.order)-entropy(noise))
    retained_lower = intact_information-dimension*math.log2(field.order)
    return {'trials': configuration['trials'], 'intact_predictive_information_bits': intact_information,
            'retained_predictive_information_lower_bound_bits': retained_lower,
            'retained_fraction_lower_bound': retained_lower/intact_information,
            'stored_syndrome_bits': (count-dimension)*field.degree,
            'source_bits': count*field.degree, 'history_symbols': observed_count,
            'maximum_precomputed_array_bytes': maximum_array_bytes,
            'interpolation_construction_seconds': construction_times, 'sampling_seconds': sampling_times,
            'sampling_law_empirically_estimated': False}


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory/'config.json').read_text())
    started = time.perf_counter()
    results = {'finite': finite_validation(configuration['finite_case']),
               'large': large_validation(configuration['large_case'], configuration['seed']),
               'python': platform.python_version(), 'numpy': np.__version__, 'neural_training': False}
    results['seconds'] = time.perf_counter()-started
    (directory/'validation.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
