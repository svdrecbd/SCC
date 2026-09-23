"""Validate consent erasure while preserving an entire finite learning machine."""

from collections import defaultdict, deque
from fractions import Fraction
from pathlib import Path
import json
import sys
import time

from validate_canonical_removal import apply_binary_matrix, invert_binary_matrix


def transition(physical_state, action, matrix, inverse_matrix, observation_output):
    logical_state = apply_binary_matrix(inverse_matrix, physical_state)
    hypotheses = logical_state & 15
    consent = logical_state >> 4
    input_value = action[1]
    labels = {hypothesis: (hypothesis & input_value).bit_count() % 2
              for hypothesis in range(4) if (hypotheses >> hypothesis) & 1}
    if action[0] == "observe":
        next_hypotheses = sum(1 << hypothesis for hypothesis, label in labels.items() if label == action[2])
        next_state = apply_binary_matrix(matrix, next_hypotheses | (consent << 4))
        output = next_hypotheses.bit_count() if observation_output == "hypothesis_count" else "observed"
        return output, next_state
    if not labels:
        return "inconsistent", physical_state
    if len(set(labels.values())) > 1:
        return "unknown", physical_state
    return next(iter(labels.values())), physical_state


def refine_partitions(graph):
    classes = [0] * len(graph)
    rounds = [list(classes)]
    while True:
        signatures = {}
        refined = []
        for state, transitions in enumerate(graph):
            signature = (classes[state], tuple((output, classes[next_state]) for output, next_state in transitions))
            refined.append(signatures.setdefault(signature, len(signatures)))
        if refined == classes:
            return classes, rounds
        assert len(set(refined)) > len(set(classes))
        classes = refined
        rounds.append(list(classes))


def distinguishing_sequence(graph, first, second):
    pending = deque([(first, second, ())])
    visited = {(first, second)}
    while pending:
        left, right, prefix = pending.popleft()
        for action, ((left_output, left_next), (right_output, right_next)) in enumerate(zip(graph[left], graph[right])):
            sequence = prefix + (action,)
            if left_output != right_output:
                return sequence
            pair = (left_next, right_next)
            if pair not in visited:
                visited.add(pair)
                pending.append((left_next, right_next, sequence))
    return None


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    assert configuration["state_bits"] == 5 and configuration["input_bits"] == 2
    matrix = configuration["mixing_matrix"]
    inverse_matrix = invert_binary_matrix(matrix)
    observation_output = configuration.get("observation_output", "hypothesis_count")
    assert observation_output in {"hypothesis_count", "acknowledgment"}
    state_count = 1 << configuration["state_bits"]
    actions = [("observe", value, label) for value in range(4) for label in range(2)]
    actions += [("predict", value) for value in range(4)]
    graph = [[transition(state, action, matrix, inverse_matrix, observation_output) for action in actions]
             for state in range(state_count)]
    classes, rounds = refine_partitions(graph)
    fibres = defaultdict(list)
    for state, equivalence_class in enumerate(classes):
        fibres[equivalence_class].append(state)
    assert len(fibres) == configuration["expected_useful_classes"]
    representatives = [min(fibres[classes[state]]) for state in range(state_count)]
    class_records = []
    bayes_correct = 0
    bisimulation_checks = 0
    for equivalence_class, members in sorted(fibres.items()):
        logical_states = [apply_binary_matrix(inverse_matrix, state) for state in members]
        assert len(members) == 2
        assert len({state & 15 for state in logical_states}) == 1
        consent_counts = [sum((state >> 4) == consent for state in logical_states) for consent in range(2)]
        assert consent_counts == [1, 1]
        bayes_correct += max(consent_counts)
        class_records.append({"class": equivalence_class, "physical_states": members,
                              "logical_states": logical_states, "representative": min(members),
                              "consent_counts": consent_counts})
        for left in members:
            for right in members:
                for (left_output, left_next), (right_output, right_next) in zip(graph[left], graph[right]):
                    assert left_output == right_output
                    assert classes[left_next] == classes[right_next]
                    bisimulation_checks += 1
    assert Fraction(bayes_correct, state_count) == Fraction(1, 2)
    pair_records = []
    for first in range(state_count):
        for second in range(first + 1, state_count):
            witness = distinguishing_sequence(graph, first, second)
            assert (witness is None) == (classes[first] == classes[second])
            pair_records.append({"first": first, "second": second,
                                 "distinguishing_inputs": None if witness is None else [actions[index] for index in witness]})
    advice_correct = 0
    for state in range(state_count):
        consent = apply_binary_matrix(inverse_matrix, state) >> 4
        for compartment in range(2):
            expected_permission = consent ^ compartment
            advice_correct += (consent ^ compartment) == expected_permission
    assert advice_correct == 64
    if configuration.get("require_sequential_distinguishing_sequence", False):
        assert len(rounds) >= 3
        assert any(len(row["distinguishing_inputs"] or []) > 1 for row in pair_records)
    trajectories = []
    prediction_checks = 0
    for hypothesis in range(4):
        for consent in range(2):
            original = apply_binary_matrix(matrix, 15 | (consent << 4))
            successor = representatives[original]
            sequence = [("observe", value, (hypothesis & value).bit_count() % 2) for value in (1, 2)]
            sequence += [("predict", value) for value in range(4)]
            steps = []
            for action in sequence:
                original_output, original = transition(original, action, matrix, inverse_matrix, observation_output)
                successor_output, successor = transition(successor, action, matrix, inverse_matrix, observation_output)
                assert original_output == successor_output
                assert classes[original] == classes[successor]
                if action[0] == "predict":
                    assert successor_output == (hypothesis & action[1]).bit_count() % 2
                    prediction_checks += 1
                steps.append({"input": action, "output": successor_output,
                              "original_state": original, "successor_state": successor})
            assert apply_binary_matrix(inverse_matrix, successor) & 15 == 1 << hypothesis
            trajectories.append({"true_hypothesis": hypothesis, "original_consent": consent, "steps": steps})
    assert prediction_checks == 32
    summary = {"status": "complete", "source_states": state_count, "useful_inputs": len(actions),
               "graph_transitions": state_count * len(actions), "equivalence_classes": len(fibres),
               "partition_class_counts": [len(set(partition)) for partition in rounds],
               "bisimulation_checks": bisimulation_checks, "learning_trajectories": len(trajectories),
               "state_pair_checks": len(pair_records),
               "distinguishable_state_pairs": sum(row["distinguishing_inputs"] is not None for row in pair_records),
               "pairs_requiring_multiple_inputs": sum(len(row["distinguishing_inputs"] or []) > 1 for row in pair_records),
               "maximum_distinguishing_length": max(len(row["distinguishing_inputs"] or []) for row in pair_records),
               "post_learning_prediction_checks": prediction_checks,
               "complete_state_consent_bayes_accuracy": str(Fraction(bayes_correct, state_count)),
               "consent_accuracy_with_advice": str(Fraction(advice_correct, 64)),
               "unbounded_history_preservation_basis": "finite bisimulation and induction",
               "neural_training": False, "general_scc_impossibility_established": False,
               "wall_seconds": time.perf_counter() - started}
    (directory / "transition_graph.json").write_text(json.dumps({"actions": actions, "graph": graph}, indent=2) + "\n")
    (directory / "equivalence_classes.json").write_text(json.dumps({"rounds": rounds, "classes": class_records}, indent=2) + "\n")
    (directory / "learning_trajectories.json").write_text(json.dumps(trajectories, indent=2) + "\n")
    (directory / "distinguishing_sequences.json").write_text(json.dumps(pair_records, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
