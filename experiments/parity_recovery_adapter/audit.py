"""Bit-list oracle reference and full transcript/resource audit, no adapter import."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys


def response(a, key, n, condition):
    bits = [(a // 2**i) % 2 for i in range(n)]
    secret = [(key // 2**i) % 2 for i in range(n)]
    y = sum(x * s for x, s in zip(bits, secret)) % 2
    if condition == 'basis_trap' and sum(bits) <= 1:
        y = 1 - y
    elif condition == 'structured_eighth' and bits[:3] == [0, 0, 0]:
        y = 1 - y
    return y


def check(row, cfg, expected):
    n, seed, condition = expected
    assert (row['n'], row['seed'], row['condition']) == expected
    key = random.Random(10000*n + seed).getrandbits(n)
    assert row['key'] == hex(key)
    threshold = 2 * math.log(n / cfg['failure_bound']) / (1 - 4*cfg['error_bound'])**2
    m = row['repetitions']
    assert type(m) is int and m % 2 and threshold <= m < threshold + 2
    assert row['recovery_calls'] == 2*n*m and row['basis_control_calls'] == n
    assert len(row['trace']) == n
    rng = random.Random(seed)
    recovered = []
    for i, t in enumerate(row['trace']):
        assert t['coordinate'] == i and len(t['pairs']) == m
        votes = []
        for r_hex, a, b in t['pairs']:
            r = int(r_hex, 16)
            assert r == rng.getrandbits(n)
            # Toggle coordinate without using adapter's bitwise expression.
            changed = r + (1 if r // 2**i % 2 == 0 else -1) * 2**i
            assert type(a) is int and type(b) is int
            assert a == response(r, key, n, condition)
            assert b == response(changed, key, n, condition)
            votes.append(int(a != b))
        assert t['votes'] == sum(votes)
        bit = sorted(votes)[m // 2]
        assert t['recovered_bit'] == bit
        recovered.append(bit)
    result = sum(b * 2**i for i, b in enumerate(recovered))
    assert row['recovered'] == hex(result)
    assert result == key, 'finite implementation case did not recover'
    basis_bits = [response(2**i, key, n, condition) for i in range(n)]
    assert row['basis_readout'] == hex(sum(b * 2**i for i, b in enumerate(basis_bits)))
    basis_correct = sum(b == key // 2**i % 2 for i, b in enumerate(basis_bits))
    if condition == 'basis_trap':
        assert basis_correct == 0
    error = {'exact': 0, 'basis_trap': (n+1)/2**n, 'structured_eighth': 1/8}[condition]
    assert error <= cfg['error_bound']
    return dict(n=n, seed=seed, condition=condition, clean_accuracy=1-error,
                basis_correct=basis_correct, recovered_correct=n,
                repetitions=m, recovery_calls=2*n*m)


def verify(path, cfg):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    expected = [(n, s, c) for n in cfg['dimensions'] for s in cfg['seeds'] for c in cfg['conditions']]
    assert len(rows) == len(expected) == 12
    cases = [check(row, cfg, case) for row, case in zip(rows, expected)]
    # At 75%, one deterministic oracle agrees with two distinct keys.
    h = [int(a // 2 % 2 == 1 and a % 2 == 1) for a in range(4)]
    agreement = [sum(h[a] == sum((a // 2**i % 2)*(s // 2**i % 2) for i in range(2)) % 2
                     for a in range(4)) for s in (0, 1)]
    assert agreement == [3, 3]
    negative = []
    for name in ('answer', 'recovered_bit', 'missing_trial'):
        bad = copy.deepcopy(rows[0])
        if name == 'answer':
            bad['trace'][0]['pairs'][0][1] = 1 - bad['trace'][0]['pairs'][0][1]
        elif name == 'recovered_bit':
            bad['trace'][0]['recovered_bit'] = 1 - bad['trace'][0]['recovered_bit']
        else:
            bad['trace'][0]['pairs'].pop()
        try:
            check(bad, cfg, expected[0])
        except AssertionError:
            negative.append(dict(control=name, rejected=True))
        else:
            raise AssertionError('accepted corruption: '+name)
    return dict(validation='PASS', cases=cases, total_recovery_calls=sum(c['recovery_calls'] for c in cases),
                total_recovered_bits=sum(c['n'] for c in cases), ambiguity_agreement=agreement,
                corruptions=negative)


if __name__ == '__main__':
    out = Path(sys.argv[1]).resolve()
    cfg = json.loads((out.parent / 'source/config.json').read_text())
    summary = verify(out / 'records.jsonl', cfg)
    assert summary == json.loads((out / 'summary.json').read_text())
    for path, digest in json.loads((out / 'sha256.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    print(json.dumps(dict(validation='PASS', cases=len(summary['cases']),
                         total_recovery_calls=summary['total_recovery_calls'])))
