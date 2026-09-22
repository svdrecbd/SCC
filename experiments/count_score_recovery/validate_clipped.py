"""Check the bounded paired score for coordinate-clipped risk recovery."""
from fractions import Fraction
from pathlib import Path
import json
import resource
import sys
import time
from validate import compositions, tails, squared_error


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / 'config.json').read_text())
    denominator = configuration['simplex_denominator']
    checks = 0
    missed_by_direct = 0
    smallest_gain = None
    largest_gain = None
    maximum_displacement = Fraction(0)
    for states in range(2, configuration['maximum_states'] + 1):
        predictions = [tuple(Fraction(value, denominator) for value in composition)
                       for composition in compositions(denominator, states)]
        for baseline in predictions:
            baseline_tails = tails(baseline)
            for prediction in predictions:
                difference = tuple(value - reference for value, reference in zip(prediction, baseline))
                perturbation = tuple((difference[index + 1] - difference[index]) / 4
                                     for index in range(states - 1))
                recovered = tuple(max(Fraction(0), min(Fraction(1), value + adjustment))
                                  for value, adjustment in zip(baseline_tails, perturbation))
                displacement = sum(abs(value - reference) for value, reference in zip(recovered, baseline_tails))
                assert displacement <= sum(map(abs, perturbation)) <= sum(map(abs, difference)) / 2 <= 1
                maximum_displacement = max(maximum_displacement, displacement)
                for level in range(states):
                    useful_target = tuple(Fraction(index == level) for index in range(states))
                    risk_target = tuple(Fraction(index < level) for index in range(states - 1))
                    useful_gain = squared_error(baseline, useful_target) - squared_error(prediction, useful_target)
                    risk_gain = squared_error(baseline_tails, risk_target) - squared_error(recovered, risk_target)
                    direct_gain = squared_error(baseline_tails, risk_target) - squared_error(tails(prediction), risk_target)
                    assert risk_gain >= useful_gain / 4
                    assert abs(risk_gain) <= 2 * displacement <= 2
                    missed_by_direct += useful_gain > 0 and direct_gain <= 0
                    smallest_gain = risk_gain if smallest_gain is None else min(smallest_gain, risk_gain)
                    largest_gain = risk_gain if largest_gain is None else max(largest_gain, risk_gain)
                    checks += 1
    assert missed_by_direct > 0
    receipt = {'checks': checks, 'useful_gains_missed_by_direct_decoder': missed_by_direct,
               'smallest_summed_gain': str(smallest_gain), 'largest_summed_gain': str(largest_gain),
               'maximum_l1_displacement': str(maximum_displacement),
               'seconds': time.monotonic() - started,
               'maximum_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'training': False}
    (directory / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
