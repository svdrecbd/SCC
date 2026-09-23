"""Check linear-summary recovery and its nonlinear information boundary."""
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import sys
import time


def multiply(matrix, vector):
    return [sum(coefficient*value for coefficient, value in zip(row, vector)) for row in matrix]


def transpose(matrix):
    return list(zip(*matrix))


def difference(left, right):
    return [first-second for first, second in zip(left, right)]


def squared_norm(vector):
    return sum(value*value for value in vector)


def projected_reader(matrix, prediction, baseline, bound):
    residual = difference(prediction, multiply(matrix, baseline))
    change = [value/bound for value in multiply(transpose(matrix), residual)]
    raw = [value+increment for value, increment in zip(baseline, change)]
    return raw, [min(Fraction(1), max(Fraction(0), value)) for value in raw]


def exact_case(matrix, prediction, baseline, truth, bound, require_identity=False):
    raw, clipped = projected_reader(matrix, prediction, baseline, bound)
    target = multiply(matrix, truth)
    useful_gain = squared_norm(difference(multiply(matrix, baseline), target))-squared_norm(difference(prediction, target))
    raw_gain = squared_norm(difference(baseline, truth))-squared_norm(difference(raw, truth))
    clipped_gain = squared_norm(difference(baseline, truth))-squared_norm(difference(clipped, truth))
    assert raw_gain >= useful_gain/bound
    if require_identity:
        assert raw_gain == useful_gain
    assert clipped_gain >= raw_gain
    # Integrated Brier loss for constant survival forecasts (Bernoulli laws).
    def brier(values):
        return sum(value*value-2*actual*value+actual for value, actual in zip(values, truth))
    protected_gain = (brier(baseline)-brier(clipped))/len(truth)
    assert protected_gain >= useful_gain/(len(truth)*bound)
    return useful_gain > 0, clipped != raw


def singleton_gain(encoding):
    fibres = defaultdict(list)
    for truth in product(range(2), repeat=4):
        fibres[encoding(truth)].append(truth)
    error = Fraction(0)
    for sources in fibres.values():
        means = [Fraction(sum(source[index] for source in sources), len(sources)) for index in range(4)]
        error += sum(sum((Fraction(source[index])-means[index])**2 for index in range(4)) for source in sources)/(16*4)
    return Fraction(1, 4)-error, len(fibres)


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    denomin = configuration['grid_denominator']
    grid = [Fraction(index, denomin) for index in range(denomin+1)]
    half = Fraction(1, 2)
    projections = {'even': [[half, half], [half, half]],
                   'odd': [[half, -half], [-half, half]],
                   'identity': [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]],
                   'zero': [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]]}
    general = {'nonorthogonal': [[Fraction(1), Fraction(2)], [Fraction(-1), Fraction(1)]],
               'scaled': [[Fraction(2), Fraction(0)], [Fraction(0), half]],
               'sum': [[Fraction(1), Fraction(1)]]}
    proposals = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
    checks = positive = clipping = 0
    for truth in product(grid, repeat=2):
        for baseline in product(grid, repeat=2):
            for proposal in proposals:
                proposed = list(map(Fraction, proposal))
                for matrix in projections.values():
                    valid, changed = exact_case(matrix, multiply(matrix, proposed), baseline, truth, Fraction(1), True)
                    checks += 1
                    positive += valid
                    clipping += changed
                for matrix in general.values():
                    bound = sum(squared_norm(row) for row in matrix)
                    valid, changed = exact_case(matrix, proposed[:len(matrix)], baseline, truth, bound)
                    checks += 1
                    positive += valid
                    clipping += changed
    rejected = False
    try:
        exact_case([[Fraction(2)]], [Fraction(2)], [Fraction(0)], [Fraction(1)], Fraction(1))
    except AssertionError:
        rejected = True
    assert rejected
    real_sum_gain, sum_outputs = singleton_gain(lambda source: (source[0]+source[1], source[2]+source[3]))
    parity_gain, parity_outputs = singleton_gain(lambda source: (source[0]^source[1], source[2]^source[3]))
    assert real_sum_gain == Fraction(1, 8) and parity_gain == 0
    for truth in product(range(2), repeat=4):
        reflected = (truth[1], truth[0], truth[3], truth[2])
        assert (truth[0]^truth[1], truth[2]^truth[3]) == (reflected[0]^reflected[1], reflected[2]^reflected[3])
    summary = {'status': 'complete', 'exact_inequality_checks': checks,
               'positive_summary_gains': positive, 'clipped_cases': clipping,
               'underestimated_operator_bound_rejected': rejected,
               'real_pair_sum_singleton_gain': str(real_sum_gain), 'real_sum_outputs': sum_outputs,
               'pair_parity_singleton_gain': str(parity_gain), 'parity_outputs': parity_outputs,
               'parity_sources': 16, 'neural_training': False,
               'wall_seconds': time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
