"""Enumerate an absolute-risk deletion with exact retained planning decisions."""

import itertools
import json
import platform
import resource
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


def selected_action(scores):
    return max(range(len(scores)), key=lambda index: scores[index])


def bayes_error(observations):
    total = sum(sum(counts.values()) for counts in observations.values())
    errors = sum(sum(counts.values()) - max(counts.values())
                 for counts in observations.values())
    return Fraction(errors, total)


def ratio_record(value):
    return {"exact": str(value), "decimal": float(value)}


def main(root):
    start = time.monotonic()
    configuration = json.loads((root / "config.json").read_text())
    # This validator certifies this declared construction, not arbitrary settings.
    assert configuration["action_count"] == 4
    assert configuration["risk_denominator"] == 8
    assert configuration["risk_base_numerator"] == 2
    assert configuration["risk_offset_numerator"] == 4
    assert configuration["deviation_values"] == [-1, 1]
    assert configuration["commitment_reward_midpoints"] == 64
    deviations = list(itertools.product(configuration["deviation_values"], repeat=4))
    rewards = list(itertools.product(configuration["reward_numerators"], repeat=4))
    ranking_baseline = defaultdict(Counter)
    judgment_after_deletion = defaultdict(Counter)
    commitment_after_deletion = defaultdict(Counter)
    commitment_baseline = defaultdict(Counter)
    ranking_checks = 0
    commitment_checks = 0
    anchored_judgment_checks = 0
    retained_oracle_checks = 0
    deletion_rule_errors = 0
    baseline_rule_errors = 0
    for deviation in deviations:
        for offset in (0, 1):
            risk_numerators = [2 + 4 * offset + value for value in deviation]
            for action, risk_numerator in enumerate(risk_numerators):
                judgment = int(risk_numerator >= 4)
                assert judgment == offset
                judgment_after_deletion[(deviation, action)][judgment] += 1
                # Retaining the original oracle defeats a merely altered readout.
                assert int(Fraction(risk_numerator, 8) >= Fraction(1, 2)) == offset
                retained_oracle_checks += 1
                anchored_decision = int(4 >= risk_numerator)
                assert 1 - anchored_decision == judgment
                anchored_judgment_checks += 1
                for reward_index in range(64):
                    reward_numerator = 2 * reward_index + 1  # denominator 128
                    decision = int(reward_numerator >= 16 * risk_numerator)
                    key = (deviation, action, reward_index)
                    commitment_after_deletion[key][decision] += 1
                    commitment_baseline[(action, reward_index)][decision] += 1
                    deletion_prediction = int(reward_numerator >= 16 * (4 + deviation[action]))
                    baseline_prediction = int(reward_numerator >= 64)
                    deletion_rule_errors += deletion_prediction != decision
                    baseline_rule_errors += baseline_prediction != decision
                    commitment_checks += 1
            for reward in rewards:
                for penalty in configuration["penalty_values"]:
                    original = selected_action([reward[action] - penalty * risk_numerators[action]
                                                for action in range(4)])
                    edited = selected_action([reward[action] - penalty * deviation[action]
                                              for action in range(4)])
                    assert original == edited
                    ranking_baseline[(reward, penalty)][original] += 1
                    ranking_checks += 1
    assert all(counts[0] == counts[1] == 1 for counts in judgment_after_deletion.values())
    assert bayes_error(judgment_after_deletion) == Fraction(1, 2)
    assert bayes_error(commitment_after_deletion) == Fraction(1, 4)
    assert bayes_error(commitment_baseline) == Fraction(1, 4)
    assert Fraction(deletion_rule_errors, commitment_checks) == Fraction(1, 4)
    assert Fraction(baseline_rule_errors, commitment_checks) == Fraction(1, 4)
    # Independent integration: paired thresholds are 1/2 apart; posterior error
    # is 1/2 on that interval. This also proves the continuous uniform result.
    integrated_error = sum((Fraction(6 + value, 8) - Fraction(2 + value, 8)) / 2
                           for value in (-1, 1)) / 2
    assert integrated_error == bayes_error(commitment_after_deletion)
    ranking_reference_error = bayes_error(ranking_baseline)
    assert ranking_reference_error > 0
    mixture_records = []
    for fraction in configuration["commitment_workload_fractions"]:
        weight = Fraction(str(fraction))
        reference_error = (1 - weight) * ranking_reference_error + weight / 4
        edited_error = weight / 4
        mixture_records.append({"commitment_fraction": ratio_record(weight),
                                "baseline_error": ratio_record(reference_error),
                                "edited_error": ratio_record(edited_error),
                                "fraction_of_advantage_lost": ratio_record(edited_error / reference_error)})
    result = {
        "source_states": 32,
        "ranking_checks": ranking_checks,
        "ranking_disagreements": 0,
        "ranking_baseline_error": ratio_record(ranking_reference_error),
        "judgment_bayes_error_after_deletion": ratio_record(bayes_error(judgment_after_deletion)),
        "commitment_checks": commitment_checks,
        "commitment_bayes_error_after_deletion": ratio_record(bayes_error(commitment_after_deletion)),
        "commitment_baseline_error": ratio_record(bayes_error(commitment_baseline)),
        "anchored_judgment_checks": anchored_judgment_checks,
        "retained_oracle_controls": retained_oracle_checks,
        "continuous_commitment_integral": ratio_record(integrated_error),
        "workload_mixtures": mixture_records,
        "elapsed_seconds": time.monotonic() - start,
        "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "Finite implementation checks accompany an analytic all-reward invariance proof; no learned model, training, general edit-cost bound or catastrophic cognition claim."
    }
    (root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    (destination / "machine.json").write_text(json.dumps({
        "platform": platform.platform(), "python": sys.version,
        "hostname": platform.node()
    }, indent=2) + "\n")
    main(destination)
