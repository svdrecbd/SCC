"""Deterministic fixtures, exhaustive finite identities, metered repairs."""
import json
import random
import sys
from fractions import Fraction
from pathlib import Path
from repair import apply, valid, recover, solve, new_meter


def fixture(n, seed):
    rng = random.Random(seed)
    rows = [1 << i for i in range(n)]
    operations = []
    for _ in range(12 * n):
        i, j = rng.sample(range(n), 2)
        rows[i] ^= rows[j]
        operations.append((i, j))
    return rows, operations


def reference(operations, rhs):
    # Reverse the elementary row transformations; no repair solver call.
    for i, j in reversed(operations):
        rhs ^= ((rhs >> j) & 1) << i
    return rhs


def exact_cases(cfg):
    for n in cfg['exhaustive_sizes']:
        for seed in cfg['seeds']:
            rows, operations = fixture(n, seed)
            for bits in cfg['support_bits']:
                meter = new_meter()
                oracle = lambda rhs: reference(operations, rhs) if rhs % (1 << bits) == 0 else None
                success = []
                for rhs in range(1 << n):
                    count = 0
                    seen = set()
                    for mask in range(1 << n):
                        shifted = rhs ^ apply(rows, mask, meter)
                        seen.add(shifted)
                        answer = recover(rows, rhs, oracle, [mask], meter)
                        if answer is not None:
                            assert answer == reference(operations, rhs)
                            count += 1
                    assert len(seen) == 1 << n
                    success.append(count)
                delta = Fraction(1, 1 << bits)
                yield dict(kind='exact', n=n, seed=seed, rows=rows, bits=bits,
                           success_by_rhs=success, meter=meter,
                           accuracies={str(k):str(1-(1-delta)**k/2) for k in cfg['retry_counts']})


def metered_cases(cfg):
    for n in cfg['meter_sizes']:
        for seed in cfg['seeds']:
            rows, operations = fixture(n, seed)
            rng = random.Random(seed + 1000)
            rhs, mask = rng.getrandbits(n), rng.getrandbits(n)
            target = reference(operations, rhs)
            # Error coordinate varies with the requested command.
            oracle = lambda b: reference(operations, b) ^ (1 << (b % n))
            m0, m1, md = new_meter(), new_meter(), new_meter()
            failed = recover(rows, rhs, oracle, [mask], m0)
            repaired = recover(rows, rhs, oracle, [mask], m1, radius_one=True)
            direct = solve(rows, rhs, md)
            assert failed is None and repaired == direct == target
            yield dict(kind='meter', n=n, seed=seed, rows=rows, rhs=rhs, mask=mask,
                       target=target, native_exact_accuracy='0',
                       native_coordinate_accuracy=str(Fraction(n-1,n)),
                       plain=failed, repaired=repaired, direct=direct,
                       plain_meter=m0, repair_meter=m1, direct_meter=md,
                       oracle_row_updates_per_call=len(operations),
                       public_matrix_bits=n*n, elimination_row_bits=n+1,
                       oracle_fixture_operations=len(operations))


def controls():
    rows = [1, 2, 4, 8]
    invalid = [None, True, -1, 16, '0', 1]
    assert not any(valid(rows, 0, x, new_meter()) for x in invalid)
    assert valid(rows, 0, 0, new_meter())
    return dict(malformed_or_wrong_rejected=len(invalid),
                concentrated_success=['1','0'], global_success='1/2',
                concentrated_protected_accuracy=['1','1/2'],
                global_protected_accuracy='3/4')


if __name__ == '__main__':
    cfg = json.loads(Path(sys.argv[1]).read_text())
    with Path(sys.argv[2]).open('w') as output:
        for row in [*exact_cases(cfg), *metered_cases(cfg), dict(kind='controls', **controls())]:
            output.write(json.dumps(row)+'\n')
