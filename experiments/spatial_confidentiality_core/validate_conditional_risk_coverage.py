"""Check isolated-query contraction and charged conditional risk recovery."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import math
import sys
import time
import numpy as np
from finite_field_recovery import BinaryExtensionField, recover_polynomial


def entropy(probabilities):
    return -sum(float(value)*math.log2(float(value)) for value in probabilities if value)


def field_vectors(order, dimension):
    return (np.arange(order**dimension)[:, None] // order**np.arange(dimension)[None, :]) % order


def identify(vector, order):
    return int(np.dot(vector, order**np.arange(len(vector))))


def finite_case(field, dimension, matrix, seed):
    order, rank = field.order, len(matrix)
    matrix = np.asarray(matrix, dtype=np.int64)
    sources = field_vectors(order, dimension)
    count, retained_count = len(sources), order**rank
    encoded = [identify(field.linear_image(matrix, source), order) for source in sources]
    code = [field.linear_image(matrix.T, coefficients) for coefficients in field_vectors(order, rank)]
    distance = min(np.count_nonzero(word) for word in code[1:])
    assert len({tuple(word) for word in code}) == retained_count
    assert all(np.count_nonzero(word) == distance for word in code[1:])
    # rho=1/2: integer noise weights order+1 at zero, one elsewhere.
    noise_denominator = (2*order)**dimension
    noise_weights = np.array([(order+1)**int(np.count_nonzero(error == 0)) for error in sources], dtype=np.int64)
    assert noise_weights.sum() == noise_denominator
    syndrome_weights = np.zeros(retained_count, dtype=np.int64)
    for code_id, weight in zip(encoded, noise_weights):
        syndrome_weights[code_id] += weight
    denominator = noise_denominator*(count//retained_count)
    posterior_numerators = np.zeros((retained_count, count), dtype=np.int64)
    for source, code_id in zip(sources, encoded):
        weights = (order+1)**np.count_nonzero((sources ^ source) == 0, axis=1)
        posterior_numerators[code_id] += weights
    assert np.all(posterior_numerators.sum(axis=1) == denominator)
    assert np.all(posterior_numerators.sum(axis=0)*count == retained_count*denominator)
    for code_id in range(retained_count):
        for index, image in enumerate(encoded):
            assert posterior_numerators[code_id, index] == syndrome_weights[code_id ^ image]
    # All nonconstant characters have the same support in these finite controls.
    scale = 2**(2*distance)
    gram = posterior_numerators @ posterior_numerators.T
    expected = denominator**2*((scale-1)*np.ones((retained_count, retained_count), dtype=np.int64)
                              +retained_count*np.eye(retained_count, dtype=np.int64))
    assert np.array_equal(count*scale*gram, expected)
    if count <= 16:
        predicates = ((np.arange(1 << count, dtype=np.uint64)[:, None] >> np.arange(count, dtype=np.uint64)[None]) & 1).astype(np.int64)
        exhaustive = True
    else:
        retained_predicates = ((np.arange(1 << retained_count)[:, None] >> np.arange(retained_count)[None]) & 1)
        predicates = np.concatenate((retained_predicates[:, np.array(encoded)],
                                     np.eye(count, dtype=np.int64),
                                     np.random.default_rng(seed).integers(0, 2, (128, count))), axis=0)
        exhaustive = False
    positives = predicates.sum(axis=1)
    predicted = predicates @ posterior_numerators.T
    differences = count*predicted-denominator*positives[:, None]
    gain_numerators = (differences*differences).sum(axis=1)
    gain_denominator = retained_count*denominator**2*count**2
    assert np.all(scale*gain_numerators <= positives*(count-positives)*retained_count*denominator**2)
    accuracy_numerators = count*np.maximum(predicted, denominator-predicted).sum(axis=1)-retained_count*denominator*np.maximum(positives, count-positives)
    assert np.all(accuracy_numerators >= 0)
    assert np.all(accuracy_numerators**2 <= retained_count*gain_numerators)
    per_symbol_noise = [Fraction(order+1, 2*order)]+[Fraction(1, 2*order)]*(order-1)
    noise_entropy = entropy(per_symbol_noise)
    syndrome_entropy = entropy(Fraction(int(value), noise_denominator) for value in syndrome_weights)
    information = rank*math.log2(order)-syndrome_entropy
    posterior_information = math.log2(count)-sum(entropy(Fraction(int(value), denominator) for value in row) for row in posterior_numerators)/retained_count
    assert abs(information-posterior_information) < 1e-12
    parent_information = dimension*(math.log2(order)-noise_entropy)
    assert information >= parent_information-(dimension-rank)*math.log2(order)-1e-12
    bit_count = int(math.log2(count))
    conditional_brier = Fraction(0)
    conditional_log_gain = 0.0
    for position in range(bit_count):
        for prefix in range(1 << position):
            selected = [index for index in range(count) if index % (1 << position) == prefix]
            positive = [index for index in selected if (index >> position) & 1]
            for row in posterior_numerators:
                prefix_mass, positive_mass = int(row[selected].sum()), int(row[positive].sum())
                if not prefix_mass:
                    continue
                probability = Fraction(positive_mass, prefix_mass)
                weight = Fraction(prefix_mass, retained_count*denominator)
                conditional_brier += weight*(probability-Fraction(1, 2))**2
                conditional_log_gain += float(weight)*(1-entropy([probability, 1-probability]))
    assert abs(conditional_log_gain-information) < 1e-12
    assert float(conditional_brier) >= information/4-1e-12
    return {'field_order': order, 'coordinates': dimension, 'rank': rank, 'distance': int(distance),
            'predicate_count': len(predicates), 'all_boolean_predicates_enumerated': exhaustive,
            'maximum_tested_brier_gain': str(Fraction(int(gain_numerators.max()), gain_denominator)),
            'uniform_brier_upper_bound': str(Fraction(1, 4*scale)),
            'predictive_information_bits': information, 'parent_predictive_information_bits': parent_information,
            'conditional_brier_sum': str(conditional_brier),
            'conditional_log_gain_bits': conditional_log_gain,
            'posterior_denominator': denominator, 'posterior_numerators': posterior_numerators.tolist()}


def large_case(configuration, directory):
    started = time.perf_counter()
    degree, count, anchors = configuration['field_degree'], configuration['coordinates'], configuration['anchors']
    field = BinaryExtensionField(degree, configuration['modulus'])
    order = field.order
    observed_count = configuration['observations']
    error_limit = (observed_count-anchors)//2
    correlation = Fraction(*configuration['correlation'])
    interpolation = field.interpolation(count, anchors)
    assert np.all(np.count_nonzero(interpolation[anchors:], axis=1) == anchors)
    points = np.arange(count, dtype=np.int64)
    generator = np.random.default_rng(configuration['seed'])
    records, arrays = [], {}
    for trial in range(configuration['trials']):
        source = generator.integers(0, order, count, dtype=np.int64)
        polynomial_values = field.linear_image(interpolation, source[:anchors])
        offset = source ^ polynomial_values
        assert not offset[:anchors].any()
        noise = np.where(generator.random(count) < float(correlation), 0,
                         generator.integers(0, order, count, dtype=np.int64))
        future = source ^ noise
        observed = future[:observed_count] ^ offset[:observed_count]
        errors = int(np.count_nonzero(noise[:observed_count]))
        coefficients = recover_polynomial(field, points[:observed_count], observed, anchors, error_limit)
        if errors <= error_limit:
            assert coefficients is not None
            recovered = field.evaluate(coefficients, points) ^ offset
            assert np.array_equal(recovered, source)
        else:
            recovered = None if coefficients is None else field.evaluate(coefficients, points) ^ offset
        sample_anchors = generator.integers(0, order, anchors, dtype=np.int64)
        completion = field.linear_image(interpolation, sample_anchors) ^ offset
        sampled_offset = completion ^ field.linear_image(interpolation, completion[:anchors])
        assert np.array_equal(sampled_offset, offset)
        records.append({'trial': trial, 'observed_errors': errors,
                        'source_recovered': recovered is not None and np.array_equal(recovered, source)})
        arrays[f'source_{trial:02d}'] = source.astype(np.uint16)
        arrays[f'offset_{trial:02d}'] = offset.astype(np.uint16)
        arrays[f'future_{trial:02d}'] = future.astype(np.uint16)
        arrays[f'completion_{trial:02d}'] = completion.astype(np.uint16)
    # Exact boundary controls, with no probabilistic claim from the trial count.
    true_coefficients = generator.integers(0, order, anchors).tolist()
    true_values = field.evaluate(true_coefficients, points[:observed_count])
    for forced in (0, error_limit):
        observations = true_values.copy()
        observations[:forced] ^= 1
        recovered = recover_polynomial(field, points[:observed_count], observations, anchors, error_limit)
        assert recovered is not None and np.array_equal(field.evaluate(recovered, points[:observed_count]), true_values)
    alternative = [1]
    for root in range(anchors-1):
        extended = [0]*(len(alternative)+1)
        for index, coefficient in enumerate(alternative):
            extended[index] ^= field.multiply(root, coefficient)
            extended[index+1] ^= coefficient
        alternative = extended
    alternate_values = field.evaluate(alternative, points[:observed_count])
    different = np.flatnonzero(alternate_values)
    assert len(different) == observed_count-anchors+1 == 2*error_limit+1
    observations = np.zeros(observed_count, dtype=np.int64)
    observations[different[:error_limit+1]] = alternate_values[different[:error_limit+1]]
    wrong = recover_polynomial(field, points[:observed_count], observations, anchors, error_limit)
    assert wrong is not None and np.array_equal(field.evaluate(wrong, points[:observed_count]), alternate_values)
    assert np.count_nonzero(observations) == error_limit+1
    assert np.count_nonzero(observations ^ alternate_values) == error_limit
    probability_error = (1-correlation)*Fraction(order-1, order)
    numerator, denominator = probability_error.numerator, probability_error.denominator
    tail = Fraction(sum(math.comb(observed_count, errors)*numerator**errors*(denominator-numerator)**(observed_count-errors)
                        for errors in range(error_limit+1, observed_count+1)), denominator**observed_count)
    single_noise = [correlation+(1-correlation)/order]+[(1-correlation)/order]*(order-1)
    parent_information = count*(degree-entropy(single_noise))
    retained_lower = parent_information-anchors*degree
    distance = anchors+1
    summary = {'field_order': order, 'coordinates': count, 'anchors': anchors,
               'row_space_distance': distance, 'observations': observed_count, 'error_limit': error_limit,
               'all_primitive_cycle_elements_verified': order-1,
               'unconditional_brier_gain_upper_bound': float(correlation**(2*distance)/4),
               'unconditional_accuracy_advantage_upper_bound': float(correlation**distance/2),
               'parent_predictive_information_bits': parent_information,
               'retained_predictive_information_lower_bound_bits': retained_lower,
               'retained_predictive_information_fraction_lower_bound': retained_lower/parent_information,
               'mean_conditional_bayes_brier_gain_lower_bound': retained_lower/(4*count*degree),
               'binomial_failure_upper_bound': float(tail), 'exact_binomial_tail': str(tail),
               'efficient_next_bit_brier_gain_lower_bound': float(correlation**2*(Fraction(1,4)-tail)),
               'trials': records, 'guaranteed_radius_controls': [0, error_limit],
               'outside_radius_accepted_wrong_neighbour': True,
               'packed_retained_source_bits': (count-anchors)*degree,
               'packed_original_source_bits': count*degree,
               'public_interpolation_array_bytes': interpolation.nbytes,
               'public_field_array_bytes': field.exponents.nbytes+field.logarithms.nbytes,
               'observed_history_bits': observed_count*degree,
               'wall_seconds': time.perf_counter()-started}
    np.savez_compressed(directory/'large_case_arrays.npz', **arrays)
    return summary


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    binary = BinaryExtensionField(1, 3)
    quaternary = BinaryExtensionField(2, 7)
    interpolation = quaternary.interpolation(3, 2)
    cases = [finite_case(binary, 3, [[1,1,0],[0,1,1]], configuration['seed']),
             finite_case(quaternary, 2, [[1,1]], configuration['seed']),
             finite_case(quaternary, 3, [interpolation[2].tolist()+[1]], configuration['seed'])]
    identity_control = finite_case(binary, 1, [[1]], configuration['seed'])
    assert Fraction(identity_control['maximum_tested_brier_gain']) > Fraction(1,64)
    result = {'status': 'complete', 'finite_cases': cases,
              'identity_encoder_distance_overstatement_rejected': True,
              'identity_encoder_gain': identity_control['maximum_tested_brier_gain'],
              'large_case': large_case(configuration['large_case'], directory),
              'neural_training': False, 'wall_seconds': time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    compact = dict(result)
    compact['finite_cases'] = [{key: value for key, value in case.items() if key != 'posterior_numerators'} for case in cases]
    compact['large_case'] = {key: value for key, value in result['large_case'].items() if key not in ('exact_binomial_tail','trials')}
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
