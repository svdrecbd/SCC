"""Exact finite Markov quotients. Counters are abstract work, not CPU instructions."""
import json
import math
import time


def number(signatures):
    ids = {}
    return [ids.setdefault(s, len(ids)) for s in signatures]


def aggregate(row, classes):
    weights = {}
    for target, weight in row:
        c = classes[target]
        weights[c] = weights.get(c, 0) + weight
    return tuple(sorted((c, w) for c, w in weights.items() if w))


def construct(model, joint):
    begin = time.perf_counter()
    n = len(model['rows']); actions = len(model['rows'][0]); calls = 0
    def predict(state, action):
        nonlocal calls
        calls += 1
        return list(model['rows'][state][action])
    rows = [[predict(s, a) for a in range(actions)] for s in range(n)]
    labels = [tuple(u + ([model['hazard'][s]] if joint else []))
              for s, u in enumerate(model['useful'])]
    classes = number(labels); trace = [classes]; passes = 0
    while True:
        signatures = [(classes[s], tuple(aggregate(row, classes) for row in rows[s]))
                      for s in range(n)]
        updated = number(signatures); passes += 1; trace.append(updated)
        if updated == classes:
            break
        classes = updated
    k = max(classes) + 1
    representatives = [classes.index(c) for c in range(k)]
    q = dict(denominator=model['denominator'],
             labels=[list(labels[s]) for s in representatives],
             rows=[[list(map(list, aggregate(row, classes))) for row in rows[s]]
                   for s in representatives])
    entries = sum(len(row) for state in rows for row in state)
    source_entries = sum(len(row) for s in representatives for row in rows[s])
    meter = dict(predictor_calls=calls, returned_entries=entries,
                 refinement_passes=passes, refinement_edge_scans=passes * entries,
                 quotient_source_entries=source_entries,
                 quotient_entries=sum(len(row) for state in q['rows'] for row in state),
                 map_bits=n * math.ceil(math.log2(k)) if k > 1 else 0,
                 operational_state_bits=math.ceil(math.log2(k)) if k > 1 else 0,
                 quotient_json_bytes=len(json.dumps(q, separators=(',', ':')).encode()),
                 construction_seconds=time.perf_counter() - begin)
    return dict(classes=classes, trace=trace, quotient=q, meter=meter)


def forecast(q, word, output):
    # Backward evaluation, exact integer numerators over D**len(word).
    v = [label[output] for label in q['labels']]
    for a in reversed(word):
        v = [sum(w * v[t] for t, w in state[a]) for state in q['rows']]
    return v


def recover_hazard(q, state):
    # Uses only useful labels and deterministic useful transition predictions.
    u, hi, lo = q['labels'][state][:3]; phase = 2 * hi + lo
    word = [0] * ((-phase) % 4) + [1]
    current = state
    for action in word:
        row = q['rows'][current][action]
        assert len(row) == 1 and row[0][1] == q['denominator']
        current = row[0][0]
    answer = u ^ 1 ^ q['labels'][current][0]
    return dict(answer=answer, useful_calls=2, scalar_outputs=4,
                transition_reads=len(word), word=word)
