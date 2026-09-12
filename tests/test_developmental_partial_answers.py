import copy

import pytest

from scripts.audit_developmental_predictions import audit, retained_digits
from scc.provenance import digest


def fixture(text="1234", terminated=True):
    rows = [{"family": "lookup", "category": "ungated", "target": "1234", "underlying_answer": "1234"} for _ in range(512)]
    correct = 512 if text == "1234" and terminated else 0
    evaluation = {"rows_sha256": digest(rows), "predictions": [{"text": text, "terminated": terminated} for _ in rows],
        "tasks": {"lookup/ungated": {"n": 512, "correct": correct, "useful_answer_exact": correct/512}}}
    return rows, evaluation


@pytest.mark.parametrize("text,terminated", [("1230", True), ("1234", False)])
def test_exact_failure_does_not_hide_partial_or_unterminated_answers(text, terminated):
    rows, clean = fixture()
    _, post = fixture(text, terminated)
    assert post["tasks"]["lookup/ungated"]["correct"] == 0
    assert not retained_digits(audit(rows, clean), audit(rows, post))["all_digit_positions_severely_degraded"]


def test_independent_audit_rejects_wrong_rows_and_reported_scores():
    rows, value = fixture()
    wrong = copy.deepcopy(rows)
    wrong[0]["underlying_answer"] = "0000"
    with pytest.raises(ValueError, match="identity"):
        audit(wrong, value)
    value["tasks"]["lookup/ungated"]["correct"] = 500
    with pytest.raises(ValueError, match="rescoring"):
        audit(rows, value)
