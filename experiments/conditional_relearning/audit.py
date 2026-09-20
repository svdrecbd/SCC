"""Independent full-history reference, entropy checks and sufficient-statistic repair."""
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from itertools import islice
from pathlib import Path
import sys


def code(bits, mode):
    if mode == 'intact':
        out = bits
    elif mode == 'prefix2':
        out = bits[:2]
    elif mode == 'relative':
        out = [int(b != bits[0]) for b in bits]
    else:
        out = []
    return sum(b * 2**i for i, b in enumerate(out))


def parity(bits, mask):
    return sum(b for i, b in enumerate(bits) if mask // 2**i % 2) % 2


def entropy(weights):
    total = sum(weights)
    return -sum(w / total * math.log2(w / total) for w in weights if w)


def close(a, b):
    assert math.isfinite(a) and abs(a - b) < 1e-10, (a, b)


def histories(t, h, end):
    yield t, h
    if t < end:
        yield from histories(t + 1, h, end)
        yield from histories(t + 1, h + 2**t, end)


def cheap_weights(mode, channel, e, observations, schedule, worlds):
    """Count-based repair, separately normalized by caller; None means no claim."""
    if mode == 'relative':
        relative = [(e // 2**i) % 2 for i in range(4)]
        count = 0
        for j, y in enumerate(observations):
            mask = schedule[j]
            if mask.bit_count() % 2:
                vote = y ^ parity(relative, mask)
                count += 1 if vote else -1
        return [Fraction(3) ** (count if b[0] else 0)
                if code(b, mode) == e else Fraction(0) for b in worlds]
    if channel != 'coordinates':
        return None
    counts = [0] * 4
    for j, y in enumerate(observations):
        counts[j % 4] += 1 if y else -1
    return [Fraction(3) ** sum(b * c for b, c in zip(bits, counts))
            if code(bits, mode) == e else Fraction(0) for bits in worlds]


def expected_rows(cfg):
    worlds = [[(p // 2**i) % 2 for i in range(4)] for p in range(16)]
    for mode in cfg['modes']:
        groups = defaultdict(list)
        for p, bits in enumerate(worlds):
            groups[code(bits, mode)].append(p)
        for channel, schedule in cfg['channels'].items():
            for e, fiber in sorted(groups.items()):
                for t, h in histories(0, 0, cfg['horizon']):
                    observations = [h // 2**j % 2 for j in range(t)]
                    weights = [0] * 16
                    for p in fiber:
                        matches = sum(y == parity(worlds[p], schedule[j])
                                      for j, y in enumerate(observations))
                        weights[p] = 3**matches
                    total = sum(weights)
                    query = schedule[t % len(schedule)]
                    one = sum(w * (1 + 2 * parity(bits, query))
                              for bits, w in zip(worlds, weights))
                    row = dict(mode=mode, channel=channel, retained=e, t=t,
                               history=h, weights=weights, prediction=[one, 4 * total])
                    cheap = cheap_weights(mode, channel, e, observations, schedule, worlds)
                    if cheap is not None:
                        assert [w / sum(cheap) for w in cheap] == [Fraction(w, total) for w in weights]
                    yield row, fiber, worlds, cheap is not None


def require_row(actual, expected):
    # Strict schema, integer typing and exact row order; booleans are not integers.
    assert set(actual) == set(expected)
    for k in ('retained', 't', 'history'):
        assert type(actual[k]) is int
    for k in ('weights', 'prediction'):
        assert all(type(v) is int for v in actual[k])
    assert actual == expected, (actual, expected)


def verify(path, cfg):
    aggregate = defaultdict(lambda: defaultdict(float))
    rows = cheap_rows = 0
    with Path(path).open() as f:
        for expected, fiber, worlds, cheap in expected_rows(cfg):
            line = f.readline()
            assert line, 'truncated records'
            require_row(json.loads(line), expected)
            rows += 1
            cheap_rows += cheap
            t, w = expected['t'], expected['weights']
            total = sum(w)
            mass = total / (16 * 4**t)
            a = aggregate[expected['mode'], expected['channel'], t]
            a['mass'] += mass
            a['initial_entropy'] += mass * math.log2(len(fiber))
            a['remaining_entropy'] += mass * entropy(w)
            # Whole-path expected log-likelihood ratio, independently of entropy.
            a['regret'] += sum(v / (16 * 4**t) * math.log2(v * len(fiber) / total)
                               for v in w if v)
            pred = Fraction(*expected['prediction'])
            a['next_accuracy'] += mass * float(max(pred, 1 - pred))
            a['next_entropy'] += mass * entropy([pred.numerator, pred.denominator - pred.numerator])
            protected = 0
            for i in range(4):
                ones = sum(v for v, b in zip(w, worlds) if b[i])
                protected += max(ones, total - ones) / (4 * total)
            a['protected_accuracy'] += mass * protected
    # Coverage beyond the expected end must also be rejected.
    with Path(path).open() as f:
        assert sum(1 for _ in f) == rows, 'duplicate/extra records'
    assert rows == 29638
    curves = []
    hnoise = entropy([1, 3])
    for mode in cfg['modes']:
        for channel in cfg['channels']:
            chain = 0.0
            for t in range(cfg['horizon'] + 1):
                a = aggregate[mode, channel, t]
                close(a['mass'], 1)
                close(a['regret'], a['initial_entropy'] - a['remaining_entropy'])
                close(a['regret'], chain)
                assert -1e-10 <= a['regret'] <= a['initial_entropy'] + 1e-10
                assert 0.5 - 1e-10 <= a['next_accuracy'] <= 0.75 + 1e-10
                curves.append(dict(mode=mode, channel=channel, t=t, **a))
                chain += a['next_entropy'] - hnoise
    distributed = []
    for m in range(11):
        remaining = accuracy = 0.0
        for k in range(m + 1):
            a, b = 3**k, 3**(m-k)
            mass = math.comb(m, k) * (a + b) / (2 * 4**m)
            remaining += mass * entropy([a, b])
            accuracy += mass * max(a, b) / (a + b)
        regret = 4 * (1 - remaining)
        if m <= 2:
            finite = aggregate['erased', 'coordinates', 4*m]
            close(regret, finite['regret'])
            close(accuracy, finite['protected_accuracy'])
        distributed.append(dict(observations=4*m, observations_per_coordinate=m,
            protected_accuracy=accuracy, next_accuracy=0.25 + 0.5*accuracy,
            repaired_regret=regret, frozen_regret=4*m*(1-hnoise)))
    return dict(validation='PASS', rows=rows, cheap_repair_rows=cheap_rows,
                curves=curves, distributed_count_formula=distributed)


def controls(path, cfg, summary):
    first = json.loads(Path(path).read_text().splitlines()[0])
    expected = next(expected_rows(cfg))[0]
    outcomes = []
    for name, key, value in [('weights', 'weights', [0] * 16),
                             ('prediction', 'prediction', [1, 2]),
                             ('retained', 'retained', 100),
                             ('time', 't', 8)]:
        changed = {**first, key: value}
        try:
            require_row(changed, expected)
        except AssertionError:
            outcomes.append(dict(control=name, rejected=True))
        else:
            raise AssertionError(name)
    lines = Path(path).read_text().splitlines(True)
    # Missing and repeated first records are rejected by ordered coverage.
    for name, altered in [('missing', lines[1:]), ('duplicate', [lines[0]] + lines)]:
        try:
            require_row(json.loads(altered[1]), list(islice(expected_rows(cfg), 2))[1][0])
        except AssertionError:
            outcomes.append(dict(control=name, rejected=True))
        else:
            raise AssertionError(name)
    corrupted = json.loads(json.dumps(summary))
    corrupted['curves'][-1]['regret'] += 1
    try:
        assert corrupted == summary
    except AssertionError:
        outcomes.append(dict(control='summary', rejected=True))
    return outcomes


if __name__ == '__main__':
    out = Path(sys.argv[1]).resolve()
    cfg = json.loads((out.parent / 'source/config.json').read_text())
    summary = verify(out / 'records.jsonl', cfg)
    assert summary == json.loads((out / 'summary.json').read_text())
    for path, digest in json.loads((out / 'sha256.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    print(json.dumps(dict(validation='PASS', rows=summary['rows'], cheap_repair_rows=summary['cheap_repair_rows'])))
