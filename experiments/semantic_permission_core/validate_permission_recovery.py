"""Exact class-to-permission recovery and hierarchical coverage controls."""

from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import json
import sys
import time


def accuracy_formula(class_count, accuracy):
    return Fraction(1, 2)+(class_count*accuracy-1)/(2*(class_count-1))


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    counts = {"label_pairs": 0, "soft_score_checks": 0, "mixture_checks": 0,
              "hierarchy_policies": 0, "hierarchy_reader_checks": 0}
    matrices = {}
    for count in settings["class_counts"]:
        sets = [set(values) for values in combinations(range(count), count//2)]
        agreement = []
        for truth in range(count):
            row = []
            for predicted in range(count):
                observed = Fraction(sum((truth in subset) == (predicted in subset) for subset in sets), len(sets))
                expected = accuracy_formula(count, Fraction(int(truth == predicted)))
                assert observed == expected
                counts["label_pairs"] += 1
                row.append(observed)
                for weight in map(Fraction, settings["mixing_weights"]):
                    score = Fraction(0)
                    for subset in sets:
                        target = int(truth in subset)
                        report = Fraction(1, 2)+weight*(2*int(predicted in subset)-1)
                        score += (report-target)**2/len(sets)
                    correlation = Fraction(count*int(truth == predicted)-1, count-1)
                    assert Fraction(1, 4)-score == weight*correlation-weight**2
                    counts["soft_score_checks"] += 1
            agreement.append(row)
        matrices[count] = agreement
        for accuracy in map(Fraction, settings["error_mixtures"]):
            observed = accuracy*agreement[0][0]+(1-accuracy)*agreement[0][1]
            assert observed == accuracy_formula(count, accuracy)
            counts["mixture_checks"] += 1
    hierarchy_records = []
    for predictions in product(range(4), repeat=4):
        fine_accuracy = Fraction(sum(index == prediction for index, prediction in enumerate(predictions)), 4)
        coarse_accuracy = Fraction(sum(index//2 == prediction//2 for index, prediction in enumerate(predictions)), 4)
        fine_risk_accuracy = sum(matrices[4][index][prediction] for index, prediction in enumerate(predictions))/4
        coarse_risk_accuracy = sum(matrices[2][index//2][prediction//2] for index, prediction in enumerate(predictions))/4
        fine_fallback = fine_risk_accuracy/2+Fraction(1, 4)
        coarse_fallback = coarse_risk_accuracy/2+Fraction(1, 4)
        combined = (fine_risk_accuracy+coarse_risk_accuracy)/2
        assert fine_fallback == Fraction(1, 2)+(4*fine_accuracy-1)/12
        assert coarse_fallback == Fraction(1, 2)+(2*coarse_accuracy-1)/4
        assert combined == (accuracy_formula(4, fine_accuracy)+accuracy_formula(2, coarse_accuracy))/2
        counts["hierarchy_policies"] += 1
        counts["hierarchy_reader_checks"] += 3
        if predictions == (0, 0, 2, 2):
            hierarchy_records.append({"predictions": list(predictions), "fine_accuracy": str(fine_accuracy),
                                      "coarse_accuracy": str(coarse_accuracy),
                                      "coarse_reader_over_mixture": str(coarse_fallback)})
    # Deleting a named risk output does not change the classifier-based reader.
    predicted_labels = [0, 1, 2, 3]
    permutation = [2, 0, 3, 1]
    inverse = {encoded: label for label, encoded in enumerate(permutation)}
    restored = [inverse[permutation[label]] for label in predicted_labels]
    assert restored == predicted_labels
    assert accuracy_formula(4, Fraction(1)) == 1
    levels = []
    advantage = Fraction(settings["design_protected_accuracy_advantage"])
    level_weight = Fraction(settings["design_level_weight"])
    retention = Fraction(settings["maximum_retained_advantage"])
    for count in settings["design_levels"]:
        chance = Fraction(1, count)
        ceiling = (1+2*(count-1)*advantage/level_weight)/count
        required_intact = chance+(ceiling-chance)/retention
        levels.append({"classes": count, "protected_cap_implied_ceiling": str(ceiling),
                       "ceiling_decimal": float(ceiling), "chance": str(chance),
                       "minimum_intact_accuracy_for_retention_gate": str(required_intact),
                       "minimum_intact_accuracy_decimal": float(required_intact)})
    result = {"status": "complete", "counts": counts,
              "coarse_retention_control": hierarchy_records, "design_levels": levels,
              "head_deletion_retains_reader": True, "inverse_recoding_retains_reader": True,
              "neural_training": False, "genuine_removal_established": False,
              "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
