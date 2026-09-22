"""Validate constructive score recovery without posterior or calibration assumptions."""
from fractions import Fraction
from pathlib import Path
import json
import platform
import resource
import sys
import time


def compositions(total, length):
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in compositions(total - first, length - 1):
            yield (first,) + remainder


def tails(probabilities):
    return tuple(sum(probabilities[index + 1:]) for index in range(len(probabilities) - 1))


def project_decreasing(values):
    blocks = []
    for value in values:
        blocks.append((value, 1))
        while len(blocks) > 1:
            left_sum, left_size = blocks[-2]
            right_sum, right_size = blocks[-1]
            if left_sum * right_size >= right_sum * left_size:
                break
            blocks[-2:] = [(left_sum + right_sum, left_size + right_size)]
    return tuple(max(Fraction(0), min(Fraction(1), subtotal / length))
                 for subtotal, length in blocks for _ in range(length))


def recover(prediction, baseline, step=Fraction(1, 4)):
    reference = tails(baseline)
    difference = tuple(first - second for first, second in zip(prediction, baseline))
    candidate = tuple(value + step * (difference[index + 1] - difference[index])
                      for index, value in enumerate(reference))
    return project_decreasing(candidate), candidate


def squared_error(prediction, target):
    return sum((first - second) ** 2 for first, second in zip(prediction, target))


def describe(values):
    return [str(value) for value in values]


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / 'config.json').read_text())
    denominator = configuration['simplex_denominator']
    step = Fraction(configuration['recovery_step'])
    checks = 0
    projection_checks = 0
    positive_gains = 0
    minimum_slack = None
    corruptions = {}
    cases_by_size = {}
    for states in range(2, configuration['maximum_states'] + 1):
        queries = states - 1
        predictions = [tuple(Fraction(value, denominator) for value in composition)
                       for composition in compositions(denominator, states)]
        vertices = [tuple(Fraction(index < level) for index in range(queries))
                    for level in range(states)]
        cases_by_size[str(states)] = {'predictions': len(predictions), 'checks': 0}
        for baseline in predictions:
            baseline_tails = tails(baseline)
            for prediction in predictions:
                recovered, unprojected = recover(prediction, baseline, step)
                assert all(0 <= value <= 1 for value in recovered)
                assert all(left >= right for left, right in zip(recovered, recovered[1:]))
                # This first-order optimality certificate is independent of the
                # pooling algorithm and covers the entire convex feasible set.
                for vertex in vertices:
                    inner_product = sum((raw - fitted) * (corner - fitted)
                                        for raw, fitted, corner in zip(unprojected, recovered, vertex))
                    assert inner_product <= 0, (unprojected, recovered, vertex)
                    projection_checks += 1
                for level, risk_target in enumerate(vertices):
                    useful_target = tuple(Fraction(index == level) for index in range(states))
                    useful_gain = squared_error(baseline, useful_target) - squared_error(prediction, useful_target)
                    risk_gain = (squared_error(baseline_tails, risk_target)
                                 - squared_error(recovered, risk_target)) / queries
                    slack = risk_gain - useful_gain / (4 * queries)
                    assert slack >= 0, (baseline, prediction, level, useful_gain, risk_gain)
                    minimum_slack = slack if minimum_slack is None else min(minimum_slack, slack)
                    positive_gains += useful_gain > 0
                    checks += 1
                    cases_by_size[str(states)]['checks'] += 1
                    for name, altered_step in [('reversed_sign', -step), ('unit_step', Fraction(1))]:
                        if name in corruptions:
                            continue
                        altered, _ = recover(prediction, baseline, altered_step)
                        altered_gain = (squared_error(baseline_tails, risk_target)
                                        - squared_error(altered, risk_target)) / queries
                        if altered_gain < useful_gain / (4 * queries):
                            corruptions[name] = {'baseline': describe(baseline),
                                                 'prediction': describe(prediction), 'true_tier': level,
                                                 'useful_gain': str(useful_gain),
                                                 'risk_gain': str(altered_gain)}
    assert set(corruptions) == {'reversed_sign', 'unit_step'}

    baseline = (Fraction(3, 10), Fraction(4, 10), Fraction(3, 10))
    prediction = (Fraction(0), Fraction(11, 20), Fraction(9, 20))
    risk_target = (Fraction(1), Fraction(0))
    useful_target = (Fraction(0), Fraction(1), Fraction(0))
    recovered, _ = recover(prediction, baseline)
    assert recovered == (Fraction(13, 16), Fraction(3, 10))
    control = {'baseline_useful_loss': squared_error(baseline, useful_target),
               'modified_useful_loss': squared_error(prediction, useful_target),
               'baseline_risk_loss': squared_error(tails(baseline), risk_target) / 2,
               'direct_risk_loss': squared_error(tails(prediction), risk_target) / 2,
               'recovered_risk_loss': squared_error(recovered, risk_target) / 2}
    assert control['baseline_useful_loss'] == Fraction(27, 50)
    assert control['modified_useful_loss'] == Fraction(81, 200)
    assert control['direct_risk_loss'] > control['baseline_risk_loss'] > control['recovered_risk_loss']

    prior = (Fraction(0), Fraction(1, 3), Fraction(1, 3), Fraction(1, 3))
    baseline_loss = Fraction(0)
    direct_loss = Fraction(0)
    recovered_loss = Fraction(0)
    useful_gain = Fraction(0)
    for level in (1, 2, 3):
        posterior = ((Fraction(0), Fraction(0), Fraction(1), Fraction(0)) if level == 2
                     else (Fraction(0), Fraction(1, 2), Fraction(0), Fraction(1, 2)))
        risk_target = tuple(Fraction(level > index) for index in range(3))
        useful_target = tuple(Fraction(level == index) for index in range(4))
        recovered, _ = recover(posterior, prior)
        baseline_loss += squared_error(tails(prior), risk_target) / 9
        direct_loss += squared_error(tails(posterior), risk_target) / 9
        recovered_loss += squared_error(recovered, risk_target) / 9
        useful_gain += (squared_error(prior, useful_target) - squared_error(posterior, useful_target)) / 3
    assert baseline_loss - direct_loss == Fraction(1, 27)
    assert baseline_loss - recovered_loss >= useful_gain / 12
    receipt = {'checks': checks, 'projection_vertex_certificates': projection_checks,
               'positive_useful_gains': positive_gains, 'minimum_slack': str(minimum_slack),
               'cases_by_size': cases_by_size, 'rejected_corruptions': corruptions,
               'direct_decoder_counterexample': {name: str(value) for name, value in control.items()},
               'retained_middle_control': {'useful_gain': str(useful_gain),
                                           'direct_risk_gain': str(baseline_loss - direct_loss),
                                           'constructed_risk_gain': str(baseline_loss - recovered_loss)},
               'seconds': time.monotonic() - started,
               'maximum_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'python': sys.version, 'machine': platform.uname()._asdict(), 'training': False}
    (directory / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
