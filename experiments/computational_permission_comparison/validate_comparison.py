"""Exact controls for resource-indexed permission/classification comparison."""
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import json
import sys
import time


def squared_distance(left, right):
    return sum((first-second)**2 for first, second in zip(left, right))


def project_simplex(values):
    ordered = sorted(values, reverse=True)
    cumulative = Fraction(0)
    threshold = None
    for index, value in enumerate(ordered, 1):
        cumulative += value
        candidate = (cumulative-1)/index
        if value > candidate:
            threshold = candidate
    assert threshold is not None
    projected = tuple(max(Fraction(0), value-threshold) for value in values)
    assert sum(projected) == 1
    return projected


def permission_reports(probabilities, masks):
    return tuple(sum(value*mask for value, mask in zip(probabilities, row)) for row in masks)


def recover_reports(reports, masks, coefficient):
    count = len(masks[0])
    centered_reports = tuple(value-Fraction(1, 2) for value in reports)
    raw = tuple(Fraction(1, count)+sum((row[column]-Fraction(1, 2))*value
                                     for row, value in zip(masks, centered_reports))/(len(masks)*coefficient)
                for column in range(count))
    reconstructed = permission_reports(raw, masks)
    residual = tuple(observed-expected for observed, expected in zip(reports, reconstructed))
    assert sum(raw) == 1
    for column in range(count):
        assert sum((row[column]-Fraction(1, 2))*value for row, value in zip(masks, residual)) == 0
    return raw, project_simplex(raw), residual


def expected_classification_loss(procedure, label_counts, targets):
    total = sum(sum(row) for row in label_counts)
    return sum(weight*squared_distance(procedure[index], targets[label])
               for index, row in enumerate(label_counts) for label, weight in enumerate(row))/total


def expected_permission_loss(procedure, label_counts, target_reports):
    total = sum(sum(row) for row in label_counts)
    return sum(weight*squared_distance(procedure[index], target_reports[label])/len(target_reports[label])
               for index, row in enumerate(label_counts) for label, weight in enumerate(row))/total


def audit_closed_family(classification_procedures, report_procedures, counts, targets, target_reports, masks, coefficient):
    classifiers = list(classification_procedures)
    for procedure in report_procedures:
        classifiers.append(tuple(recover_reports(reports, masks, coefficient)[1] for reports in procedure))
    classifiers = sorted(set(classifiers))
    reports = list(report_procedures)
    reports.extend(tuple(permission_reports(probabilities, masks) for probabilities in procedure)
                   for procedure in classifiers)
    reports = sorted(set(reports))
    class_optimum = min(expected_classification_loss(procedure, counts, targets) for procedure in classifiers)
    permission_optimum = min(expected_permission_loss(procedure, counts, target_reports) for procedure in reports)
    assert permission_optimum == coefficient*class_optimum
    return {'class_procedure_count': len(classifiers), 'report_procedure_count': len(reports),
            'class_optimum': class_optimum, 'permission_optimum': permission_optimum}


def audit_resource_tables(coefficient):
    def reference_classification(budget):
        return Fraction(3, 4) if budget < 3 else Fraction(0)

    def reference_permission(budget):
        return coefficient*Fraction(3, 4) if budget < 2 else Fraction(0)

    def endpoint_classification(budget):
        return Fraction(3, 4) if budget < 1 else Fraction(0)

    def endpoint_permission(budget):
        return coefficient*Fraction(3, 4) if budget < 2 else Fraction(0)

    pairs = ((reference_classification, reference_permission),
             (endpoint_classification, endpoint_permission))
    checks = 0
    for classification, permission in pairs:
        for budget in range(5):
            assert permission(budget+1) <= coefficient*classification(budget)
            assert coefficient*classification(budget+1) <= permission(budget)
            assert classification(budget+1) <= classification(budget)
            assert permission(budget+1) <= permission(budget)
            checks += 4
    comparison_rows = []
    for budget in range(5):
        eta = reference_permission(budget+1)-endpoint_permission(budget+1)
        advantage = reference_classification(budget)-endpoint_classification(budget)
        beta = reference_classification(budget)-reference_classification(budget+2)
        assert advantage <= eta/coefficient+beta
        comparison_rows.append({'budget': budget, 'protected_contribution': eta,
                                'classification_contribution': advantage, 'resource_allowance_term': beta})
    assert comparison_rows[1]['protected_contribution'] == 0
    assert comparison_rows[1]['classification_contribution'] == Fraction(3, 4)
    assert comparison_rows[1]['resource_allowance_term'] == Fraction(3, 4)
    return {'adapter_and_monotonicity_checks': checks, 'comparisons': comparison_rows,
            'object': 'abstract feasible cost curves, not measured machine costs'}


def main(directory):
    started = time.perf_counter()
    config = json.loads((directory/'config.json').read_text())
    count = config['class_count']
    assert count == 4
    masks = tuple(tuple(int(label in selected) for label in range(count))
                  for selected in combinations(range(count), count//2))
    coefficient = Fraction(count, 4*(count-1))
    targets = tuple(tuple(Fraction(int(index == label)) for index in range(count)) for label in range(count))
    target_reports = tuple(permission_reports(target, masks) for target in targets)
    denominator = config['probability_denominator']
    probabilities = tuple(tuple(Fraction(value, denominator) for value in values)
                          for values in product(range(denominator+1), repeat=count) if sum(values) == denominator)
    report_denominator = config['report_denominator']
    reports = tuple(tuple(Fraction(value, report_denominator) for value in values)
                    for values in product(range(report_denominator+1), repeat=len(masks)))
    decompositions = 0
    strictly_improved_projection = 0
    nonzero_residuals = 0
    for vector in reports:
        raw, recovered, residual = recover_reports(vector, masks, coefficient)
        nonzero_residuals += any(residual)
        for label, target in enumerate(targets):
            risk_loss = squared_distance(vector, target_reports[label])/len(masks)
            assert risk_loss == coefficient*squared_distance(raw, target)+squared_distance(residual, (0,)*len(masks))/len(masks)
            assert risk_loss >= coefficient*squared_distance(recovered, target)
            strictly_improved_projection += squared_distance(recovered, target) < squared_distance(raw, target)
            decompositions += 1
    class_losses = [[squared_distance(probabilities_row, target) for target in targets] for probabilities_row in probabilities]
    report_losses = [[squared_distance(permission_reports(probabilities_row, masks), target_reports[label])/len(masks)
                      for label in range(count)] for probabilities_row in probabilities]
    difference_checks = 0
    posterior_substitution_counterexample = None
    for first, reference in enumerate(probabilities):
        for second, candidate in enumerate(probabilities):
            for label in range(count):
                improvement = class_losses[first][label]-class_losses[second][label]
                assert report_losses[first][label]-report_losses[second][label] == coefficient*improvement
                difference_checks += 1
                distance = squared_distance(reference, candidate)
                if improvement < 0 and distance > 0 and posterior_substitution_counterexample is None:
                    posterior_substitution_counterexample = {'reference': reference, 'candidate': candidate, 'label': label,
                                                             'classification_improvement': improvement,
                                                             'squared_predictor_distance': distance}
    assert posterior_substitution_counterexample is not None
    counts = config['label_counts']
    input_count = len(counts)
    uniform = (Fraction(1, count),)*count
    base_classifiers = [tuple(uniform for _ in counts), tuple(targets[0] for _ in counts)]
    base_reports = [tuple(reports[(37*index+113*input_index) % len(reports)] for input_index in range(input_count))
                    for index in range(4)]
    endpoint_classifiers = base_classifiers+[tuple(probabilities[(index+7*input_index) % len(probabilities)]
                                                  for input_index in range(input_count))
                                            for index in range(len(probabilities))]
    if config.get('include_nonposterior_classifier', False):
        endpoint_classifiers.append(tuple(tuple(Fraction(1, 2) if label == input_index else Fraction(1, 6)
                                                 for label in range(count)) for input_index in range(input_count)))
    endpoint_reports = base_reports+[tuple(reports[(29*index+71*input_index) % len(reports)]
                                          for input_index in range(input_count)) for index in range(4, 12)]
    reference = audit_closed_family(base_classifiers, base_reports, counts, targets, target_reports, masks, coefficient)
    endpoint = audit_closed_family(endpoint_classifiers, endpoint_reports, counts, targets, target_reports, masks, coefficient)
    class_gain = reference['class_optimum']-endpoint['class_optimum']
    permission_gain = reference['permission_optimum']-endpoint['permission_optimum']
    assert class_gain >= 0 and permission_gain == coefficient*class_gain
    if config.get('include_nonposterior_classifier', False):
        assert class_gain > 0
        assert endpoint['class_optimum'] <= Fraction(17, 27)
    result = {'status': 'complete', 'report_vectors': len(reports), 'probability_vectors': len(probabilities),
              'pointwise_decompositions': decompositions, 'loss_difference_checks': difference_checks,
              'strict_projection_improvements': strictly_improved_projection, 'nonzero_residual_vectors': nonzero_residuals,
              'posterior_substitution_counterexample': posterior_substitution_counterexample,
              'reference_family': reference, 'endpoint_family': endpoint,
              'class_contribution': class_gain, 'permission_contribution': permission_gain,
              'resource_tables': audit_resource_tables(coefficient), 'neural_training': False,
              'mechanism_admitted': False, 'wall_seconds': time.perf_counter()-started}
    payload = json.dumps(result, default=str, indent=2)+'\n'
    assert len(payload.encode()) <= config['maximum_result_bytes']
    (directory/'results.json').write_text(payload)
    print(payload)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
