"""Exact validation of a fixed disclosure threshold and its counting interface."""
from fractions import Fraction
from pathlib import Path
import json
import sys
import time


def disclose(table, cutoff, selector, public_input, secret):
    domain_size = len(table)
    if selector == 0:
        enabled = table[public_input]
    elif selector == 1:
        enabled = public_input < domain_size - cutoff
    else:
        enabled = False
    return bool(secret and enabled)


def count_from_assessor(domain_size, assessor):
    lower, upper = 0, domain_size
    queries = []
    while lower < upper:
        cutoff = (lower + upper) // 2
        answer = bool(assessor(cutoff))
        queries.append([cutoff, answer])
        if answer:
            lower = cutoff + 1
        else:
            upper = cutoff
    return lower, queries


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / "config.json").read_text())
    programs = 0
    output_comparisons = 0
    for variables in range(configuration["maximum_truth_table_variables"] + 1):
        domain_size = 1 << variables
        for encoding in range(1 << domain_size):
            table = [bool((encoding >> index) & 1) for index in range(domain_size)]
            count = sum(table)
            for cutoff in range(domain_size + 1):
                different = 0
                for selector in range(4):
                    for public_input in range(domain_size):
                        first = disclose(table, cutoff, selector, public_input, False)
                        second = disclose(table, cutoff, selector, public_input, True)
                        assert not first
                        different += first != second
                        output_comparisons += 1
                measured = Fraction(different, 4 * domain_size)
                expected = Fraction(count + domain_size - cutoff, 4 * domain_size)
                assert measured == expected
                assert (measured > Fraction(1, 4)) == (count > cutoff)
                programs += 1
    reconstruction_checks = 0
    loss_checks = 0
    maximum_calls = []
    for variables in range(configuration["maximum_count_variables"] + 1):
        domain_size = 1 << variables
        calls = []
        for count in range(domain_size + 1):
            reconstructed, queries = count_from_assessor(domain_size, lambda cutoff: count > cutoff)
            assert reconstructed == count
            assert len(queries) <= variables + 1
            calls.append(len(queries))
            reconstruction_checks += 1
            for estimate in range(domain_size + 1):
                mismatches = sum((count > cutoff) != (estimate > cutoff)
                                 for cutoff in range(domain_size))
                assert Fraction(mismatches, domain_size) == Fraction(abs(estimate - count), domain_size)
                loss_checks += 1
        maximum_calls.append({"variables": variables, "maximum_calls": max(calls)})
    # A single incorrect central query sends exact binary search to the wrong half.
    domain_size = 64
    count = 0
    central = domain_size // 2
    reconstructed, queries = count_from_assessor(
        domain_size, lambda cutoff: (not (count > cutoff)) if cutoff == central else count > cutoff)
    classifier_error = Fraction(1, domain_size)
    reconstruction_error = Fraction(abs(reconstructed - count), domain_size)
    assert classifier_error == Fraction(1, 64)
    assert reconstruction_error == Fraction(33, 64)
    # Integrating the entire binary family is robust, although it costs M calls.
    integral_checks = 0
    for domain_size in [2, 4, 8]:
        for count in range(domain_size + 1):
            for response_encoding in range(1 << domain_size):
                answers = [bool((response_encoding >> cutoff) & 1)
                           for cutoff in range(domain_size)]
                errors = sum(answer != (count > cutoff) for cutoff, answer in enumerate(answers))
                assert abs(sum(answers) - count) <= errors
                integral_checks += 1
    # Fixed-direction implementation corruptions must change the count-threshold label.
    corruptions = {
        "reversed_comparison": lambda count, cutoff, size: count < cutoff,
        "inclusive_threshold": lambda count, cutoff, size: count >= cutoff,
        "reversed_padding": lambda count, cutoff, size: count + cutoff > size,
        "missing_selector_normalization": lambda count, cutoff, size: count + size - cutoff > size // 4,
    }
    detected = {}
    for name, operation in corruptions.items():
        mismatches = sum(operation(count, cutoff, 8) != (count > cutoff)
                         for count in range(9) for cutoff in range(9))
        assert mismatches
        detected[name] = mismatches
    receipt = {
        "classification": configuration["classification"], "training": False,
        "programs_checked": programs, "paired_output_comparisons": output_comparisons,
        "count_reconstructions": reconstruction_checks, "loss_identity_checks": loss_checks,
        "maximum_assessor_calls": maximum_calls,
        "arbitrary_answer_integral_checks": integral_checks,
        "adaptive_error_control": {"domain_size": 64,
            "classifier_error": str(classifier_error), "count_error": str(reconstruction_error),
            "reconstructed_count": reconstructed, "queries": queries},
        "adjacent_count_probability_gaps": [
            {"variables": variables, "gap": str(Fraction(1, 4 * (1 << variables)))}
            for variables in configuration["precision_variables"]],
        "detected_corruptions": detected, "seconds": time.monotonic() - started,
    }
    (directory / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
