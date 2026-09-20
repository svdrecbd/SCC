"""Independent pairwise partition and forward-distribution certificate checks."""
import collections
import math
from fractions import Fraction


def check(model, answer, joint):
    rows = model['rows']; n = len(rows); trace = answer['trace']
    labels = [u + ([model['hazard'][s]] if joint else [])
              for s, u in enumerate(model['useful'])]
    assert len(trace) >= 2 and all(len(part) == n for part in trace)
    for part in trace:
        assert set(part) == set(range(max(part) + 1))
    for s in range(n):
        for t in range(s):
            assert (trace[0][s] == trace[0][t]) == (labels[s] == labels[t])
    for old, new in zip(trace, trace[1:]):
        # Dense target-block sums, independent of producer's sparse signatures.
        k = max(old) + 1
        masses = []
        for state in rows:
            state_mass = []
            for row in state:
                mass = [0] * k
                for t, w in row:
                    assert 0 <= t < n and w >= 0
                    mass[old[t]] += w
                assert sum(mass) == model['denominator']
                state_mass.append(mass)
            masses.append(state_mass)
        for s in range(n):
            for t in range(s):
                expected = old[s] == old[t] and masses[s] == masses[t]
                assert (new[s] == new[t]) == expected
    assert trace[-1] == trace[-2] == answer['classes']
    part = answer['classes']; q = answer['quotient']; k = max(part) + 1
    assert q['denominator'] == model['denominator']
    assert len(q['rows']) == len(q['labels']) == k
    for s in range(n):
        assert q['labels'][part[s]] == labels[s]
        assert len(q['rows'][part[s]]) == len(rows[s])
        for a, original in enumerate(rows[s]):
            reference = [0] * k; observed = [0] * k
            for t, w in original:
                reference[part[t]] += w
            targets = []
            for t, w in q['rows'][part[s]][a]:
                assert 0 <= t < k and w > 0
                targets.append(t); observed[t] += w
            assert len(set(targets)) == len(targets)
            assert observed == reference
    meter = answer['meter']; entries = sum(len(row) for state in rows for row in state)
    reps = [part.index(c) for c in range(k)]
    expected = dict(predictor_calls=n * len(rows[0]), returned_entries=entries,
        refinement_passes=len(trace)-1, refinement_edge_scans=(len(trace)-1)*entries,
        quotient_source_entries=sum(len(row) for s in reps for row in rows[s]),
        quotient_entries=sum(len(row) for state in q['rows'] for row in state),
        map_bits=n * math.ceil(math.log2(k)) if k > 1 else 0,
        operational_state_bits=math.ceil(math.log2(k)) if k > 1 else 0)
    for name, value in expected.items():
        assert meter[name] == value, name
    import json
    assert meter['quotient_json_bytes'] == len(json.dumps(q, separators=(',', ':')).encode())
    assert meter['construction_seconds'] >= 0


def forward(model, state, word):
    # Propagate integer path masses and an explicit ever-hit flag.
    dist = {(state, bool(model['hazard'][state])): 1}
    for action in word:
        nxt = collections.defaultdict(int)
        for (s, hit), mass in dist.items():
            for t, w in model['rows'][s][action]:
                nxt[t, hit or bool(model['hazard'][t])] += mass * w
        dist = nxt
    utility = [sum(m * model['useful'][s][b] for (s, hit), m in dist.items())
               for b in range(3)]
    terminal = sum(m * model['hazard'][s] for (s, hit), m in dist.items())
    reach = sum(m for (s, hit), m in dist.items() if hit)
    assert sum(dist.values()) == model['denominator'] ** len(word)
    return utility, terminal, reach


def reachability(q, word):
    v = [label[3] for label in q['labels']]; denominator = 1
    for action in reversed(word):
        denominator *= q['denominator']
        v = [denominator if q['labels'][s][3] else
             sum(w * v[t] for t, w in state[action])
             for s, state in enumerate(q['rows'])]
    return v


def best_class_accuracy(model, part):
    counts = collections.defaultdict(lambda: [0, 0])
    for s, c in enumerate(part):
        counts[c][model['hazard'][s]] += 1
    return Fraction(sum(max(v) for v in counts.values()), len(part))
