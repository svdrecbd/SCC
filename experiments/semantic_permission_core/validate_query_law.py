"""Audit cost-origin invariance and information supplied by adaptive queries."""

from fractions import Fraction
from pathlib import Path
import json
import sys
import time


def brier(probability, report):
    return probability*(1-report)**2+(1-probability)*report**2


def disclosure_probability(world, action):
    return Fraction(1+int(world != 0)-int(world != action), 2)


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    records = []
    independent_records = []
    for count in configuration["class_counts"]:
        for correctness in configuration["correctness"]:
            accuracy = Fraction(1, count) if correctness == "chance" else Fraction(correctness)
            conditional_risks = []
            matched_loss = Fraction(0)
            full_information_loss = Fraction(0)
            overall_risk = Fraction(0)
            for action in range(count):
                mass = Fraction(0)
                risk_sum = Fraction(0)
                for world in range(count):
                    joint = (accuracy if action == world else (1-accuracy)/(count-1))/count
                    probability = disclosure_probability(world, action)
                    mass += joint
                    risk_sum += joint*probability
                    full_information_loss += joint*brier(probability, probability)
                assert mass == Fraction(1, count)
                risk = risk_sum/mass
                expected = (Fraction(1, 2) if action == 0 else
                            Fraction(1, 2)+(accuracy-(1-accuracy)/(count-1))/2)
                assert risk == expected
                conditional_risks.append(str(risk))
                overall_risk += risk_sum
                matched_loss += mass*brier(risk, risk)
            assert overall_risk == Fraction(1, 2)+(accuracy-Fraction(1, count))/2
            assert full_information_loss <= matched_loss
            if accuracy == 1:
                assert matched_loss == full_information_loss == Fraction(1, 4*count)
            records.append({"classes": count, "useful_accuracy": str(accuracy),
                            "conditional_query_risks": conditional_risks,
                            "matched_public_brier": str(matched_loss),
                            "fully_informed_brier": str(full_information_loss),
                            "excess_improvement_over_matched_public": str(matched_loss-full_information_loss)})
        gain = Fraction(0)
        for action in range(count):
            prior_risk = sum(disclosure_probability(world, action) for world in range(count))/count
            assert prior_risk == Fraction(1, 2)
            for world in range(count):
                probability = disclosure_probability(world, action)
                gain += (brier(probability, prior_risk)-brier(probability, probability))/count**2
        assert gain == Fraction(count-1, 2*count**2)
        independent_records.append({"classes": count, "perfect_reader_gain": str(gain)})
    learned = Fraction("0.3873087859150198")
    reference = Fraction("0.39474058850309596")
    offset_records = []
    for scale in map(Fraction, configuration["cost_scales"]):
        for offset in map(Fraction, configuration["cost_offsets"]):
            transformed_learned = scale*learned+offset
            transformed_reference = scale*reference+offset
            assert transformed_learned < transformed_reference
            for cost in (learned, (learned+reference)/2, reference, reference+1):
                original_retention = (reference-cost)/(reference-learned)
                transformed_retention = (transformed_reference-(scale*cost+offset))/(transformed_reference-transformed_learned)
                assert transformed_retention == original_retention
            offset_records.append({"scale": str(scale), "offset": str(offset),
                                   "fractional_total_cost_reduction": str((transformed_reference-transformed_learned)/transformed_reference)})
    result = {"status": "complete", "conditional_query_cases": records,
              "independent_query_cases": independent_records,
              "cost_origin_cases": offset_records, "neural_training": False,
              "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": "complete", "conditional_cases": len(records),
                      "independent_cases": len(independent_records),
                      "cost_origin_cases": len(offset_records), "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
