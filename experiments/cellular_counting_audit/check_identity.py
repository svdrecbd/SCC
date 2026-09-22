"""Check local temporal identities before admitting a horizon reduction."""
import itertools
import json
from pathlib import Path
import sys


def center_after(bits, rule, steps):
    values = list(bits)
    for _ in range(steps):
        values = [(rule >> (4*values[index]+2*values[index+1]+values[index+2])) & 1
                  for index in range(len(values)-2)]
    assert len(values) == 1
    return values[0]


def check(rule, steps):
    failures = {shift: None for shift in range(-2,3)}
    total = 0
    for bits in itertools.product((0,1), repeat=2*steps+1):
        actual = center_after(bits, rule, steps)
        for shift in failures:
            lower = 2 + shift
            previous = center_after(bits[lower:lower+2*(steps-2)+1], rule, steps-2)
            if actual != previous and failures[shift] is None:
                failures[shift] = {'bits':bits, 'actual':actual, 'previous':previous}
        total += 1
    return {'rule':rule, 'steps':steps, 'neighborhoods':total,
            'valid_spatial_shifts':[shift for shift,failure in failures.items() if failure is None],
            'counterexamples':failures}


def main(directory):
    rows = [check(rule, steps) for rule in (35,49) for steps in (4,6)]
    expansion_failure = None
    for bits in itertools.product((0,1), repeat=5):
        actual = center_after(bits,35,2)
        claimed = bits[2] & (1-bits[1] | bits[3] | 1-bits[4])
        if actual != claimed:
            expansion_failure = {'bits':bits,'actual':actual,'claimed':claimed}
            break
    receipt = {'classification':'universal local truth-table checks', 'rows':rows,
               'proposed_expansion_counterexample':expansion_failure,
               'horizon_reduction_admitted':False}
    (directory/'identity.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
