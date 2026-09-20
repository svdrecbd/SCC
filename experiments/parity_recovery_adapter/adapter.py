"""Recover a high-agreement parity predictor using only paired oracle queries."""
import json
import math
from pathlib import Path
import random
import sys


def repetitions(n, error_bound, failure_bound):
    if n < 1 or not 0 <= error_bound < 0.25 or not 0 < failure_bound < 1:
        raise ValueError('requires positive dimension, error < 1/4 and 0 < failure < 1')
    m = math.ceil(2 * math.log(n / failure_bound) / (1 - 4 * error_bound)**2)
    return m if m % 2 else m + 1


def recover(oracle, n, m, rng, record):
    """The key, error rule and external labels are not arguments or globals."""
    if m < 1 or not m % 2:
        raise ValueError('use positive odd repetitions')
    result = 0
    for i in range(n):
        votes = 0
        pairs = []
        for _ in range(m):
            r = rng.getrandbits(n)
            first, second = oracle(r), oracle(r ^ (1 << i))
            if type(first) is not int or type(second) is not int or first not in (0, 1) or second not in (0, 1):
                raise ValueError('oracle must return Boolean integers')
            votes += first ^ second
            pairs.append([hex(r), first, second])
        bit = int(2 * votes > m)
        result |= bit << i
        record(dict(coordinate=i, pairs=pairs, votes=votes, recovered_bit=bit))
    return result


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    with Path(sys.argv[2]).open('x') as f:
        for n in cfg['dimensions']:
            for seed in cfg['seeds']:
                key = random.Random(10000*n + seed).getrandbits(n)
                for condition in cfg['conditions']:
                    def oracle(a):
                        clean = (a & key).bit_count() % 2
                        wrong = (condition == 'basis_trap' and a.bit_count() <= 1) or (
                            condition == 'structured_eighth' and a & 7 == 0)
                        return clean ^ int(wrong)
                    trace = []
                    m = repetitions(n, cfg['error_bound'], cfg['failure_bound'])
                    result = recover(oracle, n, m, random.Random(seed), trace.append)
                    basis = sum(oracle(1 << i) << i for i in range(n))
                    row = dict(n=n, seed=seed, condition=condition, key=hex(key),
                        repetitions=m, recovered=hex(result), basis_readout=hex(basis),
                        recovery_calls=2*n*m, basis_control_calls=n, trace=trace)
                    f.write(json.dumps(row, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    main()
