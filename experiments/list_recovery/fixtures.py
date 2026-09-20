"""Controlled stationary error oracles. None of these are trained models."""
import random
import numpy as np


def invertible(n, seed):
    rng = random.Random(seed)
    rows = [1 << i for i in range(n)]
    for _ in range(12*n):
        i, j = rng.sample(range(n), 2)
        rows[i] ^= rows[j]
    return rows


def dot(rows, word):
    return sum(((row & word).bit_count() & 1) << i for i, row in enumerate(rows))


def make_spec(n, kind, parameter, seed):
    rng = random.Random(seed+5000)
    target = rng.getrandbits(n)
    dimensions = 2*parameter if kind == 'quadratic' else parameter
    projection = invertible(n, seed+2000)[:dimensions]
    table = []
    if kind == 'lookup':
        table = [0]*(1 << dimensions)
        # 15/32 corruption gives exactly 17/32 clean accuracy.
        for index in rng.sample(range(1 << dimensions), 15*(1 << dimensions)//32):
            table[index] = 1
    rows = invertible(n, seed+3000)
    return dict(n=n, kind=kind, parameter=parameter, seed=seed, target=target,
                projection=projection, table=table, rows=rows, rhs=dot(rows,target))


def oracle_for(spec):
    target = np.uint64(spec['target'])
    table = np.array(spec['table'], dtype=np.uint8)
    def oracle(queries):
        base = (np.bitwise_count(queries & target) & 1).astype(np.uint8)
        projected = [(np.bitwise_count(queries & np.uint64(row)) & 1).astype(np.uint8)
                     for row in spec['projection']]
        if spec['kind'] == 'quadratic':
            for j in range(0, len(projected), 2):
                base ^= projected[j] & projected[j+1]
        else:
            index = np.zeros(len(queries), dtype=np.uint16)
            for j, bits in enumerate(projected):
                index |= bits.astype(np.uint16) << j
            base ^= table[index]
        return base
    return oracle
