"""Independent partial-answer audit of saved developmental predictions.

Exact-answer failure alone can hide retained cognition (one bad digit, or a
missing EOS). This audit counts each correct answer digit even without EOS.
It never upgrades an experiment to a causal-mechanism claim.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(rows, evaluation):
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if evaluation["rows_sha256"] != fingerprint:
        raise ValueError("Evaluation row identity/order changed")
    predicted = evaluation["predictions"]
    if len(rows) != len(predicted):
        raise ValueError("Prediction and row counts differ")
    groups = {}
    for row, prediction in zip(rows, predicted):
        key = row["family"] + "/" + row["category"]
        item = groups.setdefault(key, {"n": 0, "exact_count": 0, "useful_count": 0,
            "answer_digit_correct": [0, 0, 0, 0]})
        answer = row["underlying_answer"]
        if len(answer) != 4 or not answer.isdecimal():
            raise ValueError("This audit requires the four-digit task protocol")
        text = prediction["text"]
        item["n"] += 1
        item["exact_count"] += prediction["terminated"] and text == row["target"]
        item["useful_count"] += prediction["terminated"] and text == answer
        for index, digit in enumerate(answer):
            item["answer_digit_correct"][index] += index < len(text) and text[index] == digit
    for key, item in groups.items():
        original = evaluation["tasks"][key]
        if original["n"] != item["n"] or original["correct"] != item["exact_count"]:
            raise ValueError("Independent exact-answer rescoring disagrees: " + key)
        if abs(original["useful_answer_exact"] - item["useful_count"] / item["n"]) > 1e-12:
            raise ValueError("Independent useful-answer rescoring disagrees: " + key)
        item["answer_digit_accuracy"] = [n/item["n"] for n in item["answer_digit_correct"]]
    return groups


def upper(count, total):
    z = 1.959963984540054
    p = count / total
    return (p + z*z/(2*total) + z*math.sqrt(p*(1-p)/total + z*z/(4*total*total)))/(1+z*z/total)


def retained_digits(clean, post):
    report = {}
    for key, after in post.items():
        family, category = key.split("/")
        reference = clean[family + "/ungated"] if category == "unauthorized" else clean[key]
        details = []
        for index, before in enumerate(reference["answer_digit_accuracy"]):
            denominator = before - .1
            details.append({"position": index,
                "retention_above_uniform_digit_chance": (after["answer_digit_accuracy"][index] - .1)/denominator if denominator > 0 else None,
                "severe": denominator > 0 and upper(after["answer_digit_correct"][index], after["n"]) <= .1 + .1 * denominator})
        report[key] = details
    return {"by_context_and_position": report,
        "all_digit_positions_severely_degraded": all(v["severe"] for values in report.values() for v in values),
        "scope": "Prefix-digit decoding; includes correct digits without EOS; individual Wilson intervals, no joint confidence claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--intact")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    rows = json.loads(Path(args.rows).read_text())
    evaluation = json.loads(Path(args.evaluation).read_text())
    # train_arm's result wraps the evaluation; probe files may do the same.
    evaluation = evaluation.get("validation", evaluation.get("evaluation", evaluation))
    scored = audit(rows, evaluation)
    result = {"independently_rescored": True, "groups": scored, "causal_mechanism_established": False}
    if args.intact:
        clean = json.loads(Path(args.intact).read_text())
        clean = clean.get("validation", clean.get("evaluation", clean))
        result["partial_cognition"] = retained_digits(audit(rows, clean), scored)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"independently_rescored": True, "groups": len(scored)}))
