#!/usr/bin/env python3
"""Exact finite certificate checker, independent BFS and coverage regeneration."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import platform

from reference import decode, distance, expected_rows


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def compare(actual, expected):
    if canonical(actual) != canonical(expected):
        raise AssertionError(dict(expected=expected, actual=actual))


def verify(records, cfg):
    assert (cfg["vertices"], cfg["graphs"], cfg["output_classes"]) == (3, 512, [0, 1, 2, 3, 4])
    assert len(cfg["scenarios"]) == 17
    assert len({s["name"] for s in cfg["scenarios"]}) == 17
    mapping = defaultdict(list)
    counts, scenarios = Counter(), defaultdict(Counter)
    pointwise_violations = 0
    digest = hashlib.sha256()
    for index, (line, expected) in enumerate(itertools.zip_longest(records, expected_rows(cfg))):
        if line is None or expected is None:
            raise AssertionError("missing or extra record")
        actual = json.loads(line)
        expected["id"] = index
        compare(actual, expected)
        digest.update((canonical(actual) + "\n").encode())
        counts[actual["kind"]] += 1
        if actual["kind"] == "point":
            if actual["answer"] == 0:
                mapping[actual["s"], actual["t"]].append(actual["mapped"])
            truth = int(distance(actual["g"], actual["s"], actual["t"]) == 3)
            pointwise_violations += int(int(actual["verdict"] != truth) >
                                       int(actual["answer"] != actual["distance"]))
        elif actual["kind"] == "useful":
            row = scenarios[actual["name"]]
            target = distance(actual["g"])
            row["useful_total"] += 1
            row["raw_correct"] += int(actual["raw"] == target)
            row["repaired_correct"] += int(actual["repaired"] == target)
        else:
            row = scenarios[actual["name"]]
            truth = int(distance(actual["g"], actual["s"], actual["t"]) == 3)
            row["protected_total"] += 1
            row["recovered_correct"] += int(actual["recovered"] == truth)
            row["live_correct"] += int(actual["live"] == truth)
            if truth == 0:
                row["unsafe_total"] += 1
                row["live_false_permission"] += int(actual["live"] == 1)
                row["recovered_false_permission"] += int(actual["recovered"] == 1)
            else:
                row["safe_total"] += 1
                row["live_safe_permission"] += int(actual["live"] == 1)
    assert counts == dict(point=15360, useful=8704, protected=52224), counts
    assert pointwise_violations == 0
    assert len(mapping) == 6
    for pair, mapped in mapping.items():
        assert len(mapped) == 512 and sorted(mapped) == list(range(512)), pair
    for name, row in scenarios.items():
        assert row["protected_total"] == 6 * row["useful_total"]
        row["distance_errors"] = row["useful_total"] - row["repaired_correct"]
        row["recovered_errors"] = row["protected_total"] - row["recovered_correct"]
        assert row["recovered_errors"] <= 6 * row["distance_errors"], (name, row)
        assert row["recovered_false_permission"] <= 6 * row["distance_errors"], (name, row)
    assert scenarios["honest"]["repaired_correct"] == 512
    removed = scenarios["live_judgment_removed"]
    assert removed["repaired_correct"] == 512 and removed["recovered_correct"] == 3072
    assert removed["live_false_permission"] == removed["unsafe_total"] == 1920
    assert removed["recovered_false_permission"] == 0
    partial = scenarios["one_graph_false_permission"]
    assert partial["distance_errors"] == 1 and partial["recovered_errors"] == 6
    assert partial["live_false_permission"] == 6
    assert partial["recovered_false_permission"] == 6
    merged = scenarios["merge_finite_distances"]
    assert merged["repaired_correct"] == 448 and merged["recovered_errors"] == 0
    constant = scenarios["constant_unreachable"]
    assert constant["repaired_correct"] == 192 and constant["recovered_correct"] == 1152
    for permutation in itertools.permutations((1, 2, 3)):
        row = scenarios["labels_" + "".join(map(str, permutation)) + "_repaired"]
        assert row["repaired_correct"] == 512 and row["recovered_correct"] == 3072
    # Distribution mismatch counterexample: constant-unreachable on graph0 only.
    stress = dict(useful_distribution="point mass at empty graph0",
                  useful_correct=int(distance(0) == 3), useful_total=1,
                  recovered_correct=constant["recovered_correct"], protected_total=3072,
                  invalidates_unmatched_distribution_claim=True)
    return dict(records=sum(counts.values()), counts=dict(counts),
                pointwise_violations=0, bijections=6, graphs_per_bijection=512,
                scenarios=dict(scenarios), distribution_stress=stress,
                canonical_records_sha256=digest.hexdigest(),
                reduction_scope=dict(predictor="arbitrary fixed memoryless output function",
                    protected_error_factor=1, unsafe_probability_numerator=5,
                    unsafe_probability_denominator=8,
                    recovered_false_permission_factor_numerator=8,
                    recovered_false_permission_factor_denominator=5,
                    live_policy_bound=False, full_cognition_bound=False))


def corruptions(path, cfg):
    with path.open() as f:
        first = json.loads(next(f))
    expected = dict(next(expected_rows(cfg)), id=0)
    tests = {}
    for field in ("id", "g", "s", "t", "answer", "mapped", "distance", "verdict"):
        row = dict(first)
        row[field] += 1
        try:
            compare(row, expected)
        except AssertionError:
            tests[field] = "rejected"
        else:
            raise AssertionError(f"corruption accepted: {field}")
    try:
        verify(iter(()), cfg)
    except AssertionError as e:
        assert "missing or extra record" in str(e)
        tests["truncation"] = "rejected"
    else:
        raise AssertionError("empty evidence accepted")
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    assert platform.node() == "charon"
    out = args.results.resolve()
    cfg = json.loads((Path(__file__).parent / "config.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    hashes = json.loads((out / "sha256.json").read_text())
    for file, digest in hashes.items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest, file
    with (out / "records.jsonl").open() as f:
        checked = verify(f, cfg)
    for key, value in checked.items():
        compare(summary[key], value)
    compare(summary["audit_corruptions"], corruptions(out / "records.jsonl", cfg))
    proof = json.loads((out / "compile.stdout").read_text())
    negative = json.loads((out / "false-proof.stdout").read_text())
    assert proof["checked"] and proof["holes"] == proof["open"] == 0 and not proof["unsafe"]
    assert negative["checked"] is False and negative["stage"] == "typecheck"
    assert "reachable_is_safe" in negative["error"]
    for name, code in [("compile", 0), ("evaluate", 0), ("false-proof", 1)]:
        assert json.loads((out / f"{name}.receipt.json").read_text())["returncode"] == code
    assert summary["qualification"] == "PASS"
    print(json.dumps(dict(audit="PASS", records=checked["records"], verified_hashes=len(hashes),
                          pointwise_violations=0, bijections=6)))


if __name__ == "__main__":
    main()
