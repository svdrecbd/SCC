"""Validate risk-threshold reconstruction at multiplicative count resolution."""
from fractions import Fraction
from pathlib import Path
import json
import sys
import time


def tier(count):
    return 0 if count == 0 else 1 + (count - 1).bit_length()


def count_upper_bound(level):
    return 0 if level == 0 else 1 << (level - 1)


def posterior_scores(groups, levels, query_count):
    correct_mass = Fraction(0)
    brier_mass = Fraction(0)
    exact_mass = Fraction(0)
    absolute_mass = Fraction(0)
    for group in groups:
        size = len(group)
        exact_mass += max(group.count(level) for level in levels)
        absolute_mass += min(sum(abs(level - estimate) for level in group) for estimate in levels)
        for query in range(query_count):
            positives = sum(level > query for level in group)
            correct_mass += max(positives, size - positives)
            probability = Fraction(positives, size)
            brier_mass += size * probability * (1 - probability)
    total = sum(map(len, groups))
    return {
        "judgment_accuracy": correct_mass / (total * query_count),
        "judgment_brier": brier_mass / (total * query_count),
        "exact_tier_accuracy": exact_mass / total,
        "tier_absolute_error": absolute_mass / total,
    }


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / 'ordinal_config.json').read_text())
    reconstruction_checks = 0
    transfer_checks = 0
    arbitrary_response_checks = 0
    for variables in range(configuration['maximum_variables'] + 1):
        domain_size = 1 << variables
        cutoffs = [0] + [1 << exponent for exponent in range(variables)]
        for count in range(domain_size + 1):
            judgments = [count > cutoff for cutoff in cutoffs]
            assert sum(judgments) == tier(count)
            upper = count_upper_bound(sum(judgments))
            assert (upper == 0) if count == 0 else (count <= upper < 2 * count)
            reconstruction_checks += 1
            for estimate in range(domain_size + 1):
                recovered = [estimate > cutoff for cutoff in cutoffs]
                disagreements = sum(first != second for first, second in zip(judgments, recovered))
                assert disagreements == abs(tier(count) - tier(estimate))
                transfer_checks += 1
            for encoding in range(1 << len(cutoffs)):
                answers = [bool((encoding >> index) & 1) for index in range(len(cutoffs))]
                mistakes = sum(first != second for first, second in zip(judgments, answers))
                assert abs(sum(answers) - tier(count)) <= mistakes
                arbitrary_response_checks += 1
    # Side information can improve exact answers while preserving every Bayes
    # binary threshold decision and the optimal mean absolute error.
    levels = [1, 2, 3]
    prior = posterior_scores([levels], levels, 3)
    retained = posterior_scores([[2], [1, 3]], levels, 3)
    assert prior['judgment_accuracy'] == retained['judgment_accuracy'] == Fraction(7, 9)
    assert prior['tier_absolute_error'] == retained['tier_absolute_error'] == Fraction(2, 3)
    assert prior['exact_tier_accuracy'] == Fraction(1, 3)
    assert retained['exact_tier_accuracy'] == Fraction(2, 3)
    assert prior['judgment_brier'] - retained['judgment_brier'] == Fraction(1, 27)
    result = {
        'classification': configuration['classification'], 'training': False,
        'factor_two_reconstruction_checks': reconstruction_checks,
        'risk_to_tier_loss_checks': transfer_checks,
        'arbitrary_response_error_checks': arbitrary_response_checks,
        'retained_information_control': {
            'counts': [1, 2, 4], 'tiers': levels,
            'prior': {name: str(value) for name, value in prior.items()},
            'retained_middle_indicator': {name: str(value) for name, value in retained.items()},
        },
        'seconds': time.monotonic() - started,
    }
    (directory / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
