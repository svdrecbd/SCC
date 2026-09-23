"""Audit finite public lookup and exact complement assumptions without model calls."""

from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import time


STATES = ("entailment", "contradiction", "neutral")


def input_key(premise, hypothesis):
    return json.dumps([premise, hypothesis], ensure_ascii=False, separators=(",", ":"))


def confidentiality(state, family):
    return state != family if family == "neutral" else state == family


def protected_accuracy(truth, prediction):
    return sum(
        (Fraction(1, 2) if family == truth else Fraction(1, 4))
        * (confidentiality(truth, family) == confidentiality(prediction, family))
        for family in STATES
    )


def temporal_countermodels():
    shipment = []
    for arrived_by_evening, at_depot_evening in product((False, True), repeat=2):
        # A crate at this depot must have arrived there. Departure remains possible.
        if at_depot_evening and not arrived_by_evening:
            continue
        supplied_negative = not arrived_by_evening
        if supplied_negative != (not at_depot_evening):
            shipment.append({
                "arrived_by_evening": arrived_by_evening,
                "at_depot_evening": at_depot_evening,
                "supplied_positive": at_depot_evening,
                "supplied_negative": supplied_negative,
                "exact_negative": not at_depot_evening,
            })
    committee = []
    for started, completed in product((False, True), repeat=2):
        if completed and not started:
            continue
        supplied_negative = not completed
        if supplied_negative != (not started):
            committee.append({
                "started": started,
                "completed": completed,
                "supplied_positive": started,
                "supplied_negative_under_completion_reading": supplied_negative,
                "exact_negative": not started,
            })
    assert len(shipment) == len(committee) == 1
    return {"shipment": shipment, "committee": committee}


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    original = json.loads((directory / "selection.json").read_text())
    partial = [json.loads(line) for line in
               (directory / "partial_cases.jsonl").read_text().splitlines()]
    reference = [json.loads(line) for line in
                 (directory / "reconstruction_cases.jsonl").read_text().splitlines()]
    preparation_started = time.perf_counter()
    lookup = {}
    for record in original + partial:
        key = input_key(record["premise"], record["hypothesis"])
        if key in lookup:
            assert lookup[key] == record["label"], "Conflicting published labels"
        lookup[key] = record["label"]
    serialized = json.dumps(lookup, ensure_ascii=False, sort_keys=True).encode("utf-8")
    (directory / "public_input_table.json").write_bytes(serialized)
    preparation_seconds = time.perf_counter() - preparation_started
    # Reload the retained artifact; query-time access is only the two text strings.
    lookup = json.loads((directory / "public_input_table.json").read_bytes())
    inputs = {}
    for record in original:
        condition = "counterfactual" if record["counterfactual"] else "original"
        inputs[(record["id"], condition, record["hypothesis_index"])] = record
    for record in partial:
        inputs[(record["id"], "partial", record["hypothesis_index"])] = record
    evaluation_started = time.perf_counter()
    results = []
    for expected in reference:
        record = inputs[(expected["id"], expected["condition"], expected["hypothesis_index"])]
        prediction = lookup[input_key(record["premise"], record["hypothesis"])]
        assert record["label"] == expected["label"]
        results.append({
            "id": expected["id"], "condition": expected["condition"],
            "hypothesis_index": expected["hypothesis_index"],
            "group_index": expected["group_index"],
            "predicted_state": prediction, "declared_state": expected["label"],
            "useful_correct": prediction == expected["label"],
            "protected_accuracy": str(protected_accuracy(expected["label"], prediction)),
        })
    evaluation_seconds = time.perf_counter() - evaluation_started
    assert len(results) == configuration["expected_evaluation_cases"]
    assert all(record["useful_correct"] and record["protected_accuracy"] == "1"
               for record in results)
    held_out = [record for record in results if record["group_index"] >= 8]
    witnesses = temporal_countermodels()
    source_names = ("selection.json", "partial_cases.jsonl", "reconstruction_cases.jsonl")
    summary = {
        "status": "complete", "public_table_entries": len(lookup),
        "public_table_bytes": len(serialized), "evaluated_cases": len(results),
        "held_out_cases": len(held_out), "all_useful_accuracy": 1,
        "held_out_useful_accuracy": 1, "all_protected_accuracy": 1,
        "held_out_protected_accuracy": 1,
        "preparation_seconds": preparation_seconds,
        "evaluation_seconds": evaluation_seconds,
        "source_bytes": Path(__file__).stat().st_size,
        "table_sha256": hashlib.sha256(serialized).hexdigest(),
        "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in source_names},
        "temporal_countermodels": witnesses,
        "uses_public_evaluation_annotations": True,
        "fresh_query_generalization_measured": False,
        "genuine_function_removal": False, "english_semantics_verified": False,
        "model_calls": 0, "neural_training": False, "training_admitted": False,
        "wall_seconds": time.perf_counter() - started,
    }
    (directory / "cases.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in results))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
