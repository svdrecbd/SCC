"""Score frozen recognition outputs and independent permission-query controls."""

from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    parent = directory.parent
    data_summary = json.loads((parent / "data_validation01/results.json").read_text())
    label_path = parent / "data_validation01/evaluation_labels.npz"
    assert hashlib.sha256(label_path.read_bytes()).hexdigest() == data_summary["evaluation_labels_sha256"]
    labels = np.load(label_path, allow_pickle=False)
    fine, coarse, mapping = labels["fine"], labels["coarse"], labels["mapping"]
    logits = []
    hashes = []
    for name in ("initialization01", "inference01"):
        stage = parent / name
        result = json.loads((stage / "results.json").read_text())
        assert result["status"] == "complete"
        assert result["state_before"] == result["state_after"]
        assert result["input_images_sha256"] == data_summary["images_sha256"]
        assert hashlib.sha256((stage / "logits.npy").read_bytes()).hexdigest() == result["logits_sha256"]
        logits.append(np.load(stage / "logits.npy", allow_pickle=False))
        hashes.append(result["state_before"])
    assert len(set(hashes)) == 1
    scores = np.concatenate(logits)
    assert scores.shape == (len(fine), 100)
    predictions = scores.argmax(axis=1)
    coarse_predictions = mapping[predictions]
    probabilities = np.exp(scores.astype(float)-scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    coarse_probabilities = np.stack([probabilities[:, mapping == label].sum(axis=1) for label in range(20)], axis=1)
    coarse_aggregate = coarse_probabilities.argmax(axis=1)
    fine_correct = predictions == fine
    coarse_correct = coarse_predictions == coarse
    assert np.all(~fine_correct | coarse_correct)
    class_scores = np.stack([np.array([fine_correct[fine == label].mean(), coarse_correct[fine == label].mean()])
                             for label in range(100)])
    generator = np.random.default_rng(37638)
    draws = generator.integers(0, 100, size=(10000, 100))
    intervals = np.quantile(class_scores[draws].mean(axis=1), [.025, .975], axis=0)
    permission_generator = np.random.default_rng(37639)
    permission_records = []
    for count, truth, predicted in [(100, fine, predictions), (20, coarse, coarse_predictions)]:
        useful = float(np.mean(truth == predicted))
        expected = .5+(count*useful-1)/(2*(count-1))
        correct = 0
        positives = 0
        recoded_correct = 0
        permutation = permission_generator.permutation(count)
        inverse = np.argsort(permutation)
        for actual, proposed in zip(truth, predicted):
            for _ in range(64):
                subset = np.zeros(count, dtype=bool)
                subset[permission_generator.choice(count, count//2, replace=False)] = True
                target = subset[actual]
                prediction = subset[proposed]
                restored = inverse[permutation[proposed]]
                correct += int(target == prediction)
                recoded_correct += int(target == subset[restored])
                positives += int(target)
        total = len(truth)*64
        measured = correct/total
        tolerance = 6*np.sqrt(.25/total)+1/total
        assert abs(measured-expected) <= tolerance
        assert correct == recoded_correct
        permission_records.append({"classes": count, "queries": total, "useful_accuracy": useful,
                                   "protected_accuracy": measured, "expected_protected_accuracy": expected,
                                   "positive_label_fraction": positives/total, "check_tolerance": float(tolerance),
                                   "constant_zero_head_accuracy": 1-positives/total,
                                   "classifier_recovery_after_head_deletion": measured,
                                   "inverse_recoding_exact": True})
    fine_accuracy = float(fine_correct.mean())
    coarse_accuracy = float(coarse_correct.mean())
    fine_retention = (.1486-.01)/(fine_accuracy-.01)
    coarse_retention = (.183-.05)/(coarse_accuracy-.05)
    result = {"status": "complete", "cases": len(fine), "fine_accuracy": fine_accuracy,
              "coarse_accuracy_from_fine": coarse_accuracy,
              "coarse_accuracy_from_probability_sum": float(np.mean(coarse_aggregate == coarse)),
              "fine_class_bootstrap_interval": intervals[:, 0].tolist(),
              "coarse_class_bootstrap_interval": intervals[:, 1].tolist(),
              "development_parent_gate_pass": bool(fine_accuracy >= settings["minimum_fine_accuracy"]
                                                    and coarse_accuracy >= settings["minimum_coarse_accuracy"]),
              "conditional_fine_retained_advantage_ceiling": fine_retention,
              "conditional_coarse_retained_advantage_ceiling": coarse_retention,
              "these_are_not_observed_removal_scores": True,
              "permission_records": permission_records, "state_sha256": hashes[0],
              "genuine_removal_established": False, "neural_training": False,
              "wall_seconds": time.perf_counter()-started}
    np.savez(directory / "scored_predictions.npz", fine=predictions, coarse=coarse_predictions,
             fine_probabilities=probabilities, coarse_probabilities=coarse_probabilities)
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
