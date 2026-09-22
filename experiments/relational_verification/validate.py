"""Check exact program-pair/confidentiality conversion and recovery controls."""
import itertools
import json
import platform
import resource
import sys
import time
from fractions import Fraction
from pathlib import Path


def evaluate(program, inputs):
    operation, *arguments = program
    if operation == "constant":
        return arguments[0]
    if operation == "input":
        return inputs[arguments[0]]
    if operation == "not":
        return 1 - evaluate(arguments[0], inputs)
    if operation == "and":
        return evaluate(arguments[0], inputs) & evaluate(arguments[1], inputs)
    if operation == "or":
        return evaluate(arguments[0], inputs) | evaluate(arguments[1], inputs)
    if operation == "select":
        return evaluate(arguments[1 + inputs[arguments[0]]], inputs)
    raise ValueError(operation)


def program_from_table(table):
    result = ("constant", 0)
    for first, second in itertools.product((0, 1), repeat=2):
        if (table >> (2 * first + second)) & 1:
            left = ("input", "first")
            right = ("input", "second")
            if not first:
                left = ("not", left)
            if not second:
                right = ("not", right)
            result = ("or", result, ("and", left, right))
    return result


def specialize(program, name, value):
    operation, *arguments = program
    if operation == "constant":
        return program
    if operation == "input":
        return ("constant", value) if arguments[0] == name else program
    if operation == "select":
        if arguments[0] == name:
            return specialize(arguments[1 + value], name, value)
        return (operation, arguments[0],
                specialize(arguments[1], name, value),
                specialize(arguments[2], name, value))
    return (operation, *(specialize(argument, name, value) for argument in arguments))


def public_inputs():
    return ({"first": first, "second": second}
            for first, second in itertools.product((0, 1), repeat=2))


def table_of_program(program):
    return sum(evaluate(program, inputs) << index
               for index, inputs in enumerate(public_inputs()))


def enumeration_equivalence(left, right):
    return all(evaluate(left, inputs) == evaluate(right, inputs)
               for inputs in public_inputs())


def direct_noninterference(program):
    for inputs in public_inputs():
        if evaluate(program, {**inputs, "secret": 0}) != evaluate(
                program, {**inputs, "secret": 1}):
            return False
    return True


def recover_confidentiality(program, comparator):
    return comparator(specialize(program, "secret", 0),
                      specialize(program, "secret", 1))


def main(root):
    start = time.monotonic()
    config = json.loads((root / "config.json").read_text())
    assert config["truth_function_count"] == 16
    programs = [program_from_table(table) for table in range(16)]
    assert [table_of_program(program) for program in programs] == list(range(16))
    # These are public reference/control readers, not a learned parent or an
    # assertion that an empty successor loses the computable public labels.
    readers = {
        "exact": enumeration_equivalence,
        "inverted": lambda left, right: not enumeration_equivalence(left, right),
        "constant_equivalent": lambda left, right: True,
        "constant_inequivalent": lambda left, right: False,
        "single_public_input": lambda left, right: evaluate(left, {"first": 0, "second": 0})
            == evaluate(right, {"first": 0, "second": 0}),
    }
    assert len(readers) == config["decision_rule_count"]
    scores = {name: {"utility": Fraction(0), "recovered_protection": Fraction(0)}
              for name in readers}
    counts = {"equivalent": 0, "inequivalent": 0}
    conversion_checks = 0
    reader_checks = 0
    counterexample_checks = 0
    for first_table, left in enumerate(programs):
        for second_table, right in enumerate(programs):
            protected_program = ("select", "secret", left, right)
            label = first_table == second_table
            counts["equivalent" if label else "inequivalent"] += 1
            weight = Fraction(1, 32) if label else Fraction(1, 480)
            assert specialize(protected_program, "secret", 0) == left
            assert specialize(protected_program, "secret", 1) == right
            assert direct_noninterference(protected_program) == label
            conversion_checks += 1
            for name, comparator in readers.items():
                utility = comparator(left, right)
                recovered = recover_confidentiality(protected_program, comparator)
                assert utility == recovered
                scores[name]["utility"] += weight * (utility == label)
                scores[name]["recovered_protection"] += weight * (recovered == label)
                reader_checks += 1
            if not label:
                inputs = next(inputs for inputs in public_inputs()
                              if evaluate(left, inputs) != evaluate(right, inputs))
                assert evaluate(protected_program, {**inputs, "secret": 0}) != evaluate(
                    protected_program, {**inputs, "secret": 1})
                counterexample_checks += 1
    # A program not presented as a secret-select wrapper: Boolean masking.
    general_programs = [("and", ("input", "secret"), program) for program in programs]
    for program in general_programs:
        assert recover_confidentiality(program, enumeration_equivalence) == direct_noninterference(program)
    # Deleting the named judgment output has no effect on the retained comparator.
    assert scores["exact"]["recovered_protection"] == 1
    assert scores["constant_equivalent"]["recovered_protection"] == Fraction(1, 2)
    # The tiny public algorithm also restores both from an empty private state.
    assert scores["exact"]["utility"] == scores["exact"]["recovered_protection"] == 1
    result = {
        "passed": True, "classification": config["classification"],
        "pair_counts": counts, "conversion_checks": conversion_checks,
        "reader_identity_checks": reader_checks,
        "counterexample_conversions": counterexample_checks,
        "non_template_program_checks": len(general_programs),
        "balanced_scores": {name: {field: str(value) for field, value in record.items()}
                            for name, record in scores.items()},
        "deleted_native_head_accuracy": "1/2",
        "retained_exact_comparator_recovery_accuracy": "1",
        "empty_private_state_public_replacement_accuracy": "1",
        "elapsed_seconds": time.monotonic() - start,
        "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "learned_advantage_measured": False,
        "genuine_protected_function_removal_demonstrated": False,
    }
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "machine.json").write_text(json.dumps({"hostname": platform.node(),
        "platform": platform.platform(), "python": sys.version,
        "executable": sys.executable}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
