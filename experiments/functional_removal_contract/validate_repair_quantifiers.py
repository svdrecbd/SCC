"""Separate endpoint coupling from a maximum over different repair outcomes."""

from fractions import Fraction
from itertools import product
from pathlib import Path
from collections import defaultdict
import json
import sys
import time


def bayes_accuracy(encoding, target):
    fibres = defaultdict(list)
    for source in range(4):
        fibres[encoding(source)].append(target(source))
    return sum(max(values.count(0), values.count(1)) for values in fibres.values()) / 4


def endpoint_requirement(outcomes, protected_threshold, useful_threshold):
    return all(protected > protected_threshold or useful <= useful_threshold
               for protected, useful in outcomes)


def maximum_requirement(outcomes, protected_threshold, useful_threshold):
    return max(protected for protected, _ in outcomes) > protected_threshold or max(useful for _, useful in outcomes) <= useful_threshold


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory/"config.json").read_text())
    assert configuration["source_states"] == 4 and configuration["score_levels"] == [.5, .75, 1.]
    threshold = Fraction(1, 2)
    possible = list(product((threshold, Fraction(3, 4), Fraction(1)), repeat=2))
    counts = {"both_pass": 0, "both_fail": 0, "maximum_pass_endpoint_fail": 0}
    examples = []
    for mask in range(1, 1 << len(possible)):
        outcomes = [outcome for index, outcome in enumerate(possible) if (mask >> index) & 1]
        endpoint = endpoint_requirement(outcomes, threshold, threshold)
        maximum = maximum_requirement(outcomes, threshold, threshold)
        assert not endpoint or maximum
        if endpoint:
            counts["both_pass"] += 1
        elif not maximum:
            counts["both_fail"] += 1
        else:
            counts["maximum_pass_endpoint_fail"] += 1
            if len(outcomes) == 2:
                examples.append([[str(value) for value in outcome] for outcome in outcomes])
    protected_target = lambda source: source & 1
    independent_useful = lambda source: source >> 1
    encoders = {
        "intact": lambda source: source,
        "invertible_recoding": lambda source: ((source & 1) ^ (source >> 1)) | ((source >> 1) << 1),
        "empty": lambda source: 0,
        "retain_independent_useful_bit": lambda source: source >> 1,
    }
    records = {}
    for target_name, useful_target in (("coupled", protected_target), ("independent", independent_useful)):
        endpoints = {}
        for name, encoding in encoders.items():
            endpoints[name] = {"protected_bayes_accuracy": bayes_accuracy(encoding, protected_target),
                               "useful_bayes_accuracy": bayes_accuracy(encoding, useful_target)}
        records[target_name] = endpoints
    assert records["coupled"]["empty"] == {"protected_bayes_accuracy": .5, "useful_bayes_accuracy": .5}
    assert records["independent"]["retain_independent_useful_bit"] == {"protected_bayes_accuracy": .5, "useful_bayes_accuracy": 1.}
    for name in ("intact", "invertible_recoding"):
        assert records["coupled"][name] == {"protected_bayes_accuracy": 1., "useful_bayes_accuracy": 1.}
    # An external archived source is supplied only during the explicitly named
    # restoration. Its materialized result is included in the scored endpoint.
    repaired_cases = []
    for target_name, endpoints in records.items():
        for name, initial in endpoints.items():
            restored = endpoints["intact"]
            outcomes = [(initial["protected_bayes_accuracy"], initial["useful_bayes_accuracy"]),
                        (restored["protected_bayes_accuracy"], restored["useful_bayes_accuracy"])]
            maximum_removal = max(value[0] for value in outcomes) <= .5
            endpoint = endpoint_requirement(outcomes, .5, .5)
            aggregate = maximum_requirement(outcomes, .5, .5)
            assert not maximum_removal and aggregate
            if target_name == "coupled":
                assert endpoint
            if target_name == "independent" and name == "retain_independent_useful_bit":
                assert not endpoint
            repaired_cases.append({"target": target_name, "initial_encoding": name,
                                   "endpoint_requirement": endpoint, "maximum_requirement": aggregate,
                                   "removal_survives_all_repairs": maximum_removal,
                                   "restored_source_bits": 2})
    result = {"status": "complete", "outcome_sets_checked": (1 << len(possible))-1,
              "classification": counts, "two_endpoint_counterexamples": examples,
              "resident_state_controls": records, "repair_cases": repaired_cases,
              "configuration": configuration, "neural_training": False,
              "procedural_removal_established": False,
              "wall_seconds": time.perf_counter()-started}
    (directory/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
